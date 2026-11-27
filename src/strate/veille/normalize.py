"""Étape 2 de la veille : le normaliseur (aucune IA, uniquement des règles explicables).

    raw_docs (status='new') ──▶ formats vérifiés ──▶ doublons écartés ──▶ mine(s) reconnue(s) ──▶ type
                                    │ sinon 'rejected' + motif                                   │
                                    ▼                                                            ▼
                               raw_docs.status                                   events (une ligne par mine)

Pourquoi ces choix :
- Rattacher un article à une mine est le point faible de toute veille automatique : « Victoria », « Eagle »
  ou « Gold Mine » désignent des dizaines de sites, et « Felix » est aussi un président. On exige donc le
  nom EN ENTIER (mots distinctifs, numéros de puits qui séparent), on écarte les noms génériques, et les
  homonymes sont tranchés par la géographie GDELT (lieu à moins de 50 km, sinon même pays), puis par
  la seule fiche restante qui publie sa production (à défaut, la seule qui a une emprise mesurée) :
  « priorité », force réduite ; sinon aucun rattachement.
  Mieux vaut rater un article que l'attribuer à la mauvaise mine : le corrélateur mesure des liens.
- L'index des noms (~80 000 sites, plusieurs noms chacun, toutes langues) est construit une fois par
  processus : chaque document est comparé par recherche dans un dictionnaire (séquences de mots, ou
  bigrammes pour le chinois, le japonais, le coréen, le thaï), jamais par une boucle sur les sites.
- Le classement combine les thèmes GDELT (calculés sur l'article traduit en anglais) et des mots-clés
  dans le titre d'origine, en 14 langues au moins : le titre dit souvent ce que les thèmes noient.
- Chaque événement garde sa confiance (force du rattachement × force du classement) et la preuve en JSON :
  on peut toujours expliquer pourquoi un article a été attribué à une mine.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import pandas as pd

from strate import catalog, entities
from strate.veille import collect, store

log = logging.getLogger(__name__)

STAGE = "normalisation"
NAMES_PATH = store.DB_PATH.parent / "names.parquet"   # écrit par collect.fetch_names
MIN_DATE = datetime(2015, 1, 1, tzinfo=UTC)          # GDELT 2.0 commence en 2015 : avant, date fausse
FUTURE_SLACK = timedelta(hours=26)                   # fuseaux horaires mal déclarés par les sites
DEDUP_WINDOW = timedelta(days=3)
DEDUP_MIN_WORDS = 4      # titres trop courts (« Mining news ») : pas de dédoublonnage par titre
NEAR_KM = 50.0           # un lieu GDELT à moins de 50 km confirme la mine
CLUSTER_KM = 10.0        # mêmes noms à moins de 10 km : la même mine en double fiche, pas des homonymes
MAX_HOMONYMS = 25        # au-delà, le nom est trop commun pour être tranché par la géographie
MAX_SITES_PER_DOC = 10   # un article qui cite 30 mines est une liste, pas une nouvelle sur chacune
CITY_TYPES = {3, 4}      # types de lieux GDELT ponctuels (ville US / ville du monde) ; 1, 2, 5 = centroïdes

# Force du rattachement (0–1) : multipliée par la force du classement (0,7 à 1) pour donner `confidence`.
# Seuil du corrélateur : 0,5. Un nom complet de plusieurs mots sans lieu passe (0,75 × 0,7 ≈ 0,53). Un nom
# d'un seul mot distinctif (`_weak_key` : « Fekola », mais aussi « Kibali gold », « Canadian Malartic »)
# n'atteint 0,5 QUE confirmé par un lieu GDELT à moins de 50 km : sans lieu (0,4) ou avec le pays
# seulement (0,45), il reste sous le seuil même avec le classement le plus fort (× 1,0).
STRENGTH = {
    "nom+lieu": 0.95, "nom+pays": 0.85, "nom": 0.75,
    "mot+lieu": 0.9, "mot+pays": 0.45, "mot": 0.4,      # nom d'un seul mot (plus risqué)
    "contredit": 0.45,                                   # nom présent mais l'article parle d'un autre pays
    "requête+nom": 0.9, "requête": 0.5,
    # homonymes sans lieu ni pays qui tranche : la seule fiche documentée (emprise ou production). Un mot
    # seul (« Escondida ») reste sous le seuil 0,5 comme « mot » et « mot+pays » : rattaché, pas compté.
    "nom+priorité": 0.7, "mot+priorité": 0.4,
}
ORG_FACTOR = 0.9         # nom trouvé seulement dans les organisations GDELT (extraites par machine)


# == repli des textes ===================================================================================
_APOS = str.maketrans({"'": "", "’": "", "‘": "", "ʼ": "", "`": "", "´": ""})
_LATIN = str.maketrans({"ł": "l", "ø": "o", "æ": "ae", "œ": "oe", "đ": "d", "ı": "i", "þ": "th", "ð": "d"})
_WORD = re.compile(r"[^\W_]+")
_AR_PREFIX = re.compile(r"^(?:وال|بال|فال|كال|لل|ال)(?=\w\w)")
# écritures sans espaces entre les mots : comparées caractère à caractère (bigrammes)
_CJK = re.compile("[฀-໿က-႟ᄀ-ᇿក-៿぀-ヿ㄰-㆏"
                  "㐀-䶿一-鿿가-힯豈-﫿]")
_HAN_KANA_HANGUL = re.compile("[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-䶿一-鿿"
                              "가-힯豈-﫿]")
_ARABIC = re.compile("[؀-ۿ]")
_ROMAN = catalog._ROMAN


def fold(s: str) -> str:
    """Minuscules, sans accents ni apostrophes, toutes écritures gardées (cyrillique, arabe, grec…).

    `catalog._fold` passe en ASCII et effacerait « Распадская » : ici on retire seulement les marques
    combinantes. Appliqué aux noms ET aux titres, il suffit qu'il soit cohérent avec lui-même.
    """
    s = unicodedata.normalize("NFKD", s.translate(_APOS).casefold()).translate(_LATIN)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def words(s: str) -> list[str]:
    """Mots repliés dans l'ordre ; l'article arabe collé (« ال », « وال ») est retiré."""
    out = []
    for t in _WORD.findall(fold(s)):
        if _ARABIC.match(t):
            t = _AR_PREFIX.sub("", t)
        out.append(t)
    return out


def _cjk_fold(s: str) -> str:
    """Pour les écritures sans espaces : NFKC (pleine chasse → normale), minuscules, sans blancs."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s).casefold())


def _fset(ws) -> frozenset[str]:
    return frozenset(t for w in ws for t in words(w))


# Mots du TYPE d'ouvrage, toutes langues : retirés des noms comme des titres (« mina Spence » = « Spence
# mine » = « Spence ») et servant d'indice de contexte pour les noms d'un seul mot.
MINE_WORDS = _fset("""
mine mines mining miner miners colliery collieries quarry quarries pit pits shaft shafts deposit deposits
project projects operation operations minier miniere minieres gisement projet carriere mina minas minera
minero mineros mineria yacimiento proyecto faena cantera tajo mineracao mineradora jazida projeto garimpo
bergwerk bergbau grube zeche tagebau schacht steinbruch miniera giacimento progetto cava kopalnia kopalni
kopalnie kopalnię kwk odkrywka рудник рудника руднике шахта шахты шахте шахту шахтёры шахтеры разрез
карьер прииск месторождение месторождения гок комбинат рудник tambang pertambangan penambang maden madeni
madencilik ocagi ocağı mijn groeve gruva gruvan gruvor kaivos dul doly bánya منجم مناجم المنجم تعدين
""".split())
STOP = frozenset(catalog._STOP) | MINE_WORDS

# Mots NON distinctifs (gardés dans la séquence mais insuffisants seuls) : ceux du catalogue + autres langues
GENERIC = frozenset(catalog._GENERIC) | _fset("""
золото золотой золотая золотое медный медная угольный угольная железный железная северный северная южный
южная восточный западный новый новая старый старая большой малый гора имени им
gold kupfer kohle eisen salz silber nord süd ost west neu alt gross groß klein berg sankt
złoto miedź węgiel węgla srebro północ południe nowa nowy stara stary
altın bakır kömür demir gümüş kuzey güney yeni eski büyük küçük
emas tembaga batubara nikel bijih timah perak utara selatan timur barat baru
oro rame carbone ferro argento nuovo nuova vecchio
ذهب نحاس فحم حديد فضة شمال جنوب
""".split())

# Substances : indice de contexte, et mots qu'on peut retirer en fin de nom
# (« Escondida copper » → « Escondida »)
COMMODITY = _fset("""
gold copper coal iron ore silver zinc nickel tin uranium lithium diamond diamonds platinum cobalt
manganese chrome chromite bauxite potash phosphate graphite tungsten molybdenum vanadium rare earth
oro plata cobre carbon hierro ouro prata ferro carvao niquel argent cuivre charbon fer
kupfer kohle eisen silber gold braunkohle steinkohle rame carbone argento miedź węgiel złoto
золото золотой медь медный уголь угольный железо никель алмаз алмазы
altın bakır kömür demir emas tembaga batubara nikel timah ذهب نحاس فحم
""".split())


def _forms(bases: str, endings: str) -> set[str]:
    """Mot de base + chacune des terminaisons (« - » = mot nu) : « canadien » → canadienne, canadiens…"""
    ends = ["" if e == "-" else e for e in endings.split()]
    return {b + e for b in bases.split() for e in ends}


# Nationalités / gentilés : JAMAIS distinctifs. Sans cela, « American Gold Mine », « Indian Copper
# Complex » ou « Canadian Coal » deviennent des noms de plusieurs mots que tout titre sur le cours de l'or
# ou la demande indienne de cuivre « contient » en entier. Principales langues de la veille ; les formes
# fléchies sont générées (une forme de trop ne coûte rien : elle n'est qu'un mot non distinctif de plus).
DEMONYMS = _fset(" ".join([
    # anglais (invariables)
    """american canadian australian chinese russian indian african brazilian chilean peruvian mexican
    argentine argentinian argentinean bolivian colombian ecuadorian venezuelan guyanese surinamese british
    english scottish welsh irish french german spanish portuguese italian dutch belgian swiss austrian
    swedish norwegian finnish danish icelandic polish czech slovak hungarian romanian bulgarian serbian
    croatian bosnian slovenian montenegrin macedonian albanian greek turkish ukrainian belarusian kazakh
    kazakhstani uzbek kyrgyz tajik turkmen mongolian japanese korean vietnamese thai lao laotian burmese
    cambodian malaysian indonesian filipino philippine papuan fijian kiwi pakistani afghan iranian persian
    iraqi saudi emirati omani yemeni jordanian israeli egyptian moroccan algerian tunisian libyan sudanese
    ethiopian eritrean somali kenyan tanzanian ugandan rwandan burundian congolese zambian zimbabwean
    malawian mozambican angolan namibian botswanan motswana batswana malagasy ghanaian ivorian malian
    burkinabe nigerien nigerian senegalese mauritanian guinean liberian sierra leonean cameroonian gabonese
    chadian european asian latin arab arabian nordic scandinavian siberian alaskan nevadan texan
    quebecois quebecer ontarian zealander""",
    # français : -e -s -es (-ne -nes pour « -ien »)
    " ".join(_forms("""americain canadien australien chinois russe indien africain bresilien chilien
    peruvien mexicain argentin bolivien colombien equatorien venezuelien britannique anglais francais
    allemand espagnol portugais italien neerlandais belge suisse suedois norvegien finlandais danois
    polonais tcheque hongrois roumain bulgare serbe croate grec turc ukrainien kazakh ouzbek mongol
    japonais coreen vietnamien thailandais indonesien philippin pakistanais afghan iranien irakien saoudien
    egyptien marocain algerien tunisien libyen soudanais ethiopien kenyan tanzanien ougandais rwandais
    burundais congolais zambien zimbabween malawite mozambicain angolais namibien ghaneen ivoirien malien
    burkinabe nigerien nigerian senegalais mauritanien guineen liberien camerounais gabonais tchadien
    europeen asiatique arabe quebecois malgache""", "- e s es ne nes")),
    # espagnol, portugais, italien : -o -a -os -as -i -e ; -és/-ês → -esa -eses
    " ".join(_forms("""american canadiens australian chin rus russ indi african brasilen brasileir chilen
    peruan mexican argentin bolivian colombian ecuatorian venezolan britanic britanico ingles frances
    aleman espanol portugues italian holandes suec polac polones turc ucranian japones corean sudafrican
    sul-african european asiatic arab congolen angolan mocambican cinese canadese giappones ingles frances
    olandes svedes portoghes tedesc spagnol cilen peruvian messican brasilian sudafrican""",
                    "- o a os as i e es esa eses")),
    # allemand : -e -en -er -es
    " ".join(_forms("""amerikanisch kanadisch australisch chinesisch russisch indisch afrikanisch
    brasilianisch chilenisch peruanisch mexikanisch argentinisch britisch englisch franzosisch deutsch
    spanisch portugiesisch italienisch niederlandisch schwedisch norwegisch finnisch polnisch tschechisch
    turkisch ukrainisch kasachisch mongolisch japanisch koreanisch indonesisch sudafrikanisch europaisch
    asiatisch arabisch kongolesisch""", "- e en er es em")),
    # russe : adjectifs en -ский / -цкий
    " ".join(_forms("""американск канадск австралийск китайск российск русск индийск африканск бразильск
    чилийск перуанск мексиканск британск английск французск испанск японск корейск казахстанск казахск
    монгольск украинск южноафриканск европейск азиатск немецк турецк""",
                    "ий ая ое ие ого ой ую их им ом ому ими")),
    # portugais, turc, polonais (formes irrégulières)
    """alemao alemaes alema tedeschi tedesche amerikan kanadali avustralyali rus
    amerykanski amerykanska kanadyjski kanadyjska australijski chinski chinska rosyjski rosyjska polski
    polska polskie polskiej niemiecki niemiecka""",
]).split())
NON_CORE = GENERIC | COMMODITY | DEMONYMS   # mots qui ne rendent jamais un nom distinctif

# Noms d'un seul mot qui sont aussi des mots courants de titres : seule une confirmation par un lieu
# à moins de 50 km permet de les retenir (« Premier », « Hope », « Victoria », « Felix »…).
COMMON = _fset("""
premier victory victoria champion pioneer discovery fortune progress liberty union independence future
summit standard crown royal empire national general golden world global america american africa europe
asia pacific atlantic energy power star sun moon eagle hope phoenix success prosperity excelsior
enterprise alliance freedom peace justice president king queen prince princess lady emperor jupiter mars
venus mercury titan atlas apollo orion genesis nova delta alpha beta omega gamma sigma echo bonanza
jackpot eureka paradise eldorado dorado grande lucky chance home tiger lion bear wolf fox buffalo jewel
crystal ruby emerald sapphire opal pearl trump news update market prices report record porphyry skarn
epithermal orogenic breccia exploration drilling resource reserve tailings
esperanza fortuna libertad porvenir progreso paz gloria providencia concordia dolores rosario carmen
maria jesus cristo trinidad asuncion soledad mercedes pilar luz sol rica rico boa vista esperança pedra
cruz rosa clara elena lucia isabel teresa juana juan jose pedro pablo miguel antonio francisco felix
january february march april may june july august september october november december monday tuesday
wednesday thursday friday saturday sunday janvier fevrier mars avril mai juin juillet aout septembre
octobre novembre decembre enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre
diciembre janeiro fevereiro marco maio junho julho setembro outubro novembro dezembro januar februar
marz juni juli oktober dezember gennaio febbraio aprile maggio giugno luglio settembre ottobre dicembre
government state states bank banks city stock stocks park party trade young federal republic national
international universal unknown unnamed none other anything long quick model motor museum radio transport
target trend rail rain relief rescue recovery reform refuse resolution rise metal metals milling morning
midnight millennium mission monster pipeline page panel prospectus super superior surprise spirit strip
stone tech tower trail tunnel valentine thanksgiving vicinity wonder wood dream faith harmony horizon
frontier gateway giant galaxy cloud comet cosmos dragon falcon castle camp chase contact content
combination congress defender doctor exhibition historical hunt hazard inland majestic imperial sovereign
regent pilot patch portal pride ranger rambler rolling savage sentinel shine spot starlight stellar sunrise
sunset sunshine sunlight oversight foresight matchless peerless optimum oracle mammoth jumbo jubilee karma
triad trio veteran warrior viking cowboy outlaw prospector adventure advance adjacent affinity anchor arena
battery bulletin cable cargo development epoch fence grinding industry century center centre commonwealth
continental cosmopolitan metropolitan oriental highland midland midway landmark navigation boundary
climax century centurion challenger champion commodore cardinal monarch citizen pioneer premier
""".split())

# Composés germaniques/nordiques : « Goldmine », « Kohlegrube » portent le contexte « mine » dans le mot
_COMPOUND_TAILS = ("mine", "minen", "grube", "gruben", "bergwerk", "zeche", "tagebau", "mijn", "gruva",
                   "gruvan", "kaivos")
_NOT_COMPOUND = {"determine", "examine", "famine", "undermine", "jasmine", "ermine", "bromine", "chlorine"}

# Suffixes CJK du type d'ouvrage, retirés en fin de nom (« 大柳塔煤矿 » → « 大柳塔煤 », trouvé dans le titre)
_CJK_SUFFIX = sorted(set("""矿山 矿区 矿井 矿场 矿床 矿 礦 礦山 鉱山 鉱床 鉱区 鉱 炭鉱 炭坑 광산 탄광 광업소
광구 광상 项目 專案 プロジェクト 露天 露天矿""".split()), key=len, reverse=True)
_CJK_GENERIC = set("金银銀铜銅铁鐵鉄煤炭锌鋅铅鉛镍鎳锡錫钨鎢钼鉬锂鋰稀土磷钾鉀盐鹽石油气氣矿礦鉱山区區井场場"
                   "号號第一二三四五六七八九十东東西南北中新老大小上下前后後公司集团集團有限责任股份露天地"
                   "금은동철석탄광산")
_CJK_MINE_CHARS = set("矿礦鉱광坑炭탄")


# == pays des lieux GDELT ===============================================================================
# GDELT code les pays en FIPS 10-4, PAS en ISO (AS = Australie, RS = Russie, SF = Afrique du Sud). Le
# collecteur lit le pays dans le nom complet du lieu, mais les lieux « adjectifs » (« Russian », « Dutch »)
# ne portent que le code FIPS : cette table les rattrape.
_FIPS = dict(x.split(" ", 1) for x in """AA Aruba;AC Antigua and Barbuda;AE United Arab Emirates
AF Afghanistan;AG Algeria;AJ Azerbaijan;AL Albania;AM Armenia;AN Andorra;AO Angola;AQ American Samoa
AR Argentina;AS Australia;AU Austria;AV Anguilla;BA Bahrain;BB Barbados;BC Botswana;BD Bermuda
BE Belgium;BF Bahamas;BG Bangladesh
BH Belize;BK Bosnia and Herzegovina;BL Bolivia;BM Myanmar;BN Benin;BO Belarus;BP Solomon Islands;BR Brazil
BT Bhutan;BU Bulgaria;BX Brunei;BY Burundi;CA Canada;CB Cambodia;CD Chad;CE Sri Lanka;CF Republic of the Congo
CG Democratic Republic of the Congo;CH China;CI Chile;CJ Cayman Islands;CM Cameroon;CN Comoros;CO Colombia
CS Costa Rica;CT Central African Republic;CU Cuba;CV Cape Verde;CY Cyprus;DA Denmark;DJ Djibouti;DO Dominica
DR Dominican Republic;EC Ecuador;EG Egypt;EI Ireland;EK Equatorial Guinea;EN Estonia;ER Eritrea;ES El Salvador
ET Ethiopia;EZ Czech Republic;FG French Guiana;FI Finland;FJ Fiji;FM Federated States of Micronesia;FR France
GA Gambia;GB Gabon;GG Georgia;GH Ghana;GJ Grenada;GL Greenland;GM Germany;GR Greece;GT Guatemala;GV Guinea
GY Guyana;HA Haiti;HK Hong Kong;HO Honduras;HR Croatia;HU Hungary;IC Iceland;ID Indonesia;IN India;IR Iran
IS Israel;IT Italy;IV Côte d'Ivoire;IZ Iraq;JA Japan;JM Jamaica;JO Jordan;KE Kenya;KG Kyrgyzstan
KN North Korea;KR Kiribati;KS South Korea;KU Kuwait;KV Kosovo;KZ Kazakhstan;LA Laos;LE Lebanon;LG Latvia
LH Lithuania;LI Liberia;LO Slovakia;LS Liechtenstein;LT Lesotho;LU Luxembourg;LY Libya;MA Madagascar
MD Moldova;MG Mongolia;MI Malawi;MJ Montenegro;MK North Macedonia;ML Mali;MN Monaco;MO Morocco;MP Mauritius
MR Mauritania;MT Malta;MU Oman;MV Maldives;MX Mexico;MY Malaysia;MZ Mozambique;NC New Caledonia;NG Niger
NH Vanuatu;NI Nigeria;NL Netherlands;NO Norway;NP Nepal;NR Nauru;NS Suriname;NU Nicaragua;NZ New Zealand
OD South Sudan;PA Paraguay;PE Peru;PK Pakistan;PL Poland;PM Panama;PO Portugal;PP Papua New Guinea;PS Palau
PU Guinea-Bissau;QA Qatar;RB Serbia;RI Serbia;RO Romania;RP Philippines;RQ Puerto Rico;RS Russia;RW Rwanda
SA Saudi Arabia;SE Seychelles;SF South Africa;SG Senegal;SI Slovenia;SL Sierra Leone;SN Singapore;SO Somalia
SP Spain;SU Sudan;SW Sweden;SY Syria;SZ Switzerland;TD Trinidad and Tobago;TH Thailand;TI Tajikistan;TO Togo
TS Tunisia;TT East Timor;TU Turkey;TW Taiwan;TX Turkmenistan;TZ Tanzania;UG Uganda;UK United Kingdom
UP Ukraine;US United States;UV Burkina Faso;UY Uruguay;UZ Uzbekistan;VE Venezuela;VM Vietnam;WA Namibia
WI Western Sahara;WZ Eswatini;YM Yemen;ZA Zambia;ZI Zimbabwe""".replace("\n", ";").split(";"))


def loc_country(loc: dict) -> str | None:
    """Pays canonique (comme le catalogue) d'un lieu GDELT : nom lisible, sinon code FIPS (`cc`)."""
    c = entities.norm_country(loc.get("country"))
    if c in entities.COUNTRIES:
        return c
    cc = loc.get("cc") or (c if isinstance(c, str) and len(c) == 2 else None)
    name = _FIPS.get(str(cc).upper()) if cc else None
    return entities.norm_country(name) if name else None


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


# == index des noms =====================================================================================
_SPLIT = re.compile(r"\s+(?:and|und|et|y|&)\s+|[/;|()\[\]（）、]|\s+[-–]\s+", re.I)


def _seq(s: str) -> tuple[str, ...]:
    """Séquence des mots utiles, dans l'ordre : sans « mine », « the »… ; chiffres romains → arabes."""
    return tuple(_ROMAN.get(t, t) for t in words(s) if t not in STOP)


def _is_core(t: str) -> bool:
    """Mot distinctif : ni générique (relief, orientation…), ni substance, ni nationalité, ni numéro."""
    return t not in NON_CORE and not t.isdigit() and len(t) >= 3


def _all_common(key: tuple[str, ...]) -> bool:
    """Nom dont tous les mots distinctifs sont des mots courants de titres (« Premier », « State Coal »,
    « Golden Eagle ») : comme un mot seul courant, il lui faut un lieu à moins de 50 km."""
    return all(t in COMMON or not _is_core(t) for t in key)


# Mots non distinctifs qui forment pourtant un nom de lieu avec UN mot distinctif : relief, eau, orientation,
# taille, couleur, articles et saints (« Mount Isa », « Olympic Dam », « Los Bronces », « Timmins West »).
# Les substances, nationalités et mots d'ouvrage ou de société, eux, ne font que décrire : « Kibali gold »,
# « Canadian Malartic », « Rossing uranium » restent des noms d'UN mot.
TOPONYM = _fset("""
mount mt mont monte montana cerro sierra mountain mountains hill hills ridge peak valley creek river lake
lakes gulch canyon spring springs bay island field fields flat flats point rock rocks well wells dam north
south east west northern southern eastern western central norte sur este oeste upper lower new old big
little great grand deep main nueva nuevo novo nova black white red blue green el los las lo da do das dos
di du des del der die den van von san santa santo sao saint st ste sainte nuestra senora
гора северный северная южный южная восточный западный новый новая старый старая большой малый
nord süd ost west neu alt gross groß klein berg sankt północ południe nowa nowy stara stary
kuzey güney yeni eski büyük küçük utara selatan timur barat baru nuovo nuova vecchio شمال جنوب
""".split())


def _weak_key(key: tuple[str, ...]) -> bool:
    """Nom traité comme un nom d'UN mot (lieu à moins de 50 km exigé pour atteindre 0,5) : un seul mot
    distinctif, entouré seulement de substances, nationalités, numéros ou mots génériques non toponymiques
    (« Kibali gold », « Canadian Malartic », « Laizhou Gold »). Deux mots distinctifs, ou un mot distinctif
    avec un mot de lieu (« Mount Isa », « Olympic Dam »), font un vrai nom de plusieurs mots."""
    core = sum(_is_core(t) for t in key)
    return core < 2 and not (core == 1 and any(t in TOPONYM for t in key))


def word_key(name: str) -> tuple[str, ...] | None:
    """Clé d'un nom en écriture à mots, ou None s'il est trop générique pour être cherché seul.

    Refusés : aucun mot distinctif (« Gold Mine », « Mine No 2 », « North Pit », « American Gold »,
    « Indian Copper Complex » : nationalités et substances ne distinguent rien), un nom de pays
    (alias « Peru » d'une mine péruvienne), un mot seul de moins de 4 lettres. Pour les noms latins, on
    exige en plus un mot distinctif au sens du catalogue (`catalog._core`) : même définition partout.
    """
    seq = _seq(name)
    if not seq or len(seq) > 8 or not any(_is_core(t) for t in seq):
        return None
    if len(seq) == 1 and len(seq[0]) < 4:
        return None
    if re.search(r"[A-Za-z]", name) and not catalog._core(catalog._tokens(name)):
        return None
    if entities.fold(" ".join(seq)) in entities.COUNTRY_ALIASES:
        return None
    if len(seq) >= 2 and re.search(r"[A-Za-z]", name) and collect.region_like(name):
        return None      # « Ural Mountains », « Lake Victoria » : un lieu (voir `place_key`)
    return seq


def place_key(name: str) -> tuple[str, ...] | None:
    """Clé d'un nom PRINCIPAL (catalogue, liste suivie) fait seulement de mots de lieu, de pays ou
    génériques (« Red Lake », « Copper Mountain ») : les noms que le collecteur accepte en phrase exacte
    pour l'API documentaire (`collect._place_name_key`), sans quoi les articles de cette requête ne
    deviendraient jamais des événements alors que la couverture de la mine serait comptée ; et les noms
    de région + relief (`collect.region_like` : « Ural Mountains », « Cerro Colorado »), que le collecteur
    ne cherche pas.

    Ces noms sont aussi des noms de lieux (« Red Lake » est une ville de l'Ontario, « Ural Mountains » une
    chaîne de montagnes) : `match` les traite comme un nom d'un mot de mots courants, donc seulement
    confirmés par un lieu GDELT ponctuel à moins de 50 km ou par la requête de la mine (« requête+nom »)."""
    if not isinstance(name, str) or word_key(name) is not None:
        return None
    if collect._place_name_key(name) is None and not collect.region_like(name):
        return None
    seq = _seq(name)
    return seq if 2 <= len(seq) <= 8 else None


def _tail_variant(seq: tuple[str, ...]) -> tuple[str, ...] | None:
    """« Daliuta coal », « Escondida copper » → « Daliuta », « Escondida » (la substance décrit, comme
    « mine »). Jamais si le reste perd tout mot distinctif ou si le dernier mot est un numéro."""
    s = seq
    while len(s) > 1 and s[-1] in COMMODITY:
        s = s[:-1]
    return s if s != seq and any(_is_core(t) for t in s) and not (len(s) == 1 and len(s[0]) < 4) else None


def cjk_key(name: str) -> tuple[str, bool] | None:
    """Clé d'un nom chinois/japonais/coréen/thaï : nom sans suffixe d'ouvrage. (clé, faible) ou None.

    Il faut 2 caractères distinctifs (hors substances, relief, points cardinaux, « mine », « n° »), ou 1
    dans un nom d'au moins 3 caractères : « 金矿 » (mine d'or) ou « 第一煤矿 » ne désignent aucune mine
    en particulier. Un nom de 2 caractères est « faible » : il lui faut « 矿 » juste après, ou un lieu.
    """
    s = re.sub(r"[()（）\[\]【】「」『』\"'·・,，.。\-]", "", _cjk_fold(name))
    changed = True
    while changed:
        changed = False
        for suf in _CJK_SUFFIX:
            if s.endswith(suf) and len(s) > len(suf):
                s, changed = s[: -len(suf)], True
                break
    core = [ch for ch in s if _CJK.match(ch) and ch not in _CJK_GENERIC]
    if _HAN_KANA_HANGUL.search(s):
        # « 大柳塔煤 » (2 caractères distinctifs) ou « 紫金山金铜 » (1 distinctif dans un nom long)
        if not (len(core) >= 2 or (core and len(s) >= 3)):
            return None
        return s, len(s) <= 2
    if len(core) < 4:      # thaï, khmer… : des lettres, pas des idéogrammes
        return None
    return s, False


MAX_ALIASES = 20   # au-delà, la liste d'alias est un inventaire historique (sociétés, puits), pas des noms


@dataclass
class NameIndex:
    """Index construit une fois par processus. `sites` : id → (nom, lat, lon, pays canonique)."""

    sites: dict[str, tuple[str | None, float | None, float | None, str | None]] = field(default_factory=dict)
    words: dict[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)
    cjk: dict[str, list[tuple[str, tuple[str, ...], bool]]] = field(default_factory=dict)
    twin_of: dict[str, str] = field(default_factory=dict)   # id catalogue → id de la liste suivie
    place_keys: frozenset[tuple[str, ...]] = frozenset()    # clés de `place_key` : lieu ou requête exigés
    same_as: dict[str, str] = field(default_factory=dict)   # fiche en double (< 10 km, même nom) → gardée
    # toutes les clés de chaque fiche, AVANT les filtres (fiche étayée, homonymes, doubles) : servent
    # seulement à confirmer qu'un article renvoyé par la requête d'une mine cite bien l'un de ses noms
    keys_of: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)
    cjk_of: dict[str, set[str]] = field(default_factory=dict)
    notable: frozenset[str] = frozenset()  # fiche étayée : liste suivie, ICMM, Wikipedia, emprise, production
    documented: frozenset[str] = frozenset()  # données propres : emprise ou production, ou liste suivie
    producing: frozenset[str] = frozenset()   # production publiée (ou mine suivie au stade production)
    maxlen: int = 1
    dropped_generic: int = 0              # clés écartées : trop d'homonymes
    catalog_rows: int = 0                 # fiches du catalogue indexées (0 : catalogue absent)

    def canonical(self, site_id: str | None) -> str | None:
        return self.twin_of.get(site_id, site_id) if site_id else None


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def build_index(cat: pd.DataFrame | None, watch: list | None = None, names: pd.DataFrame | None = None,
                twins: dict[str, str] | None = None) -> NameIndex:
    """Construit l'index des noms. `cat` : catalogue (colonnes id, name, aliases, lat, lon, country) ;
    `watch` : objets Site de la liste suivie ; `names` : noms multilingues (site_id, name) ;
    `twins` : mine suivie → sa jumelle du catalogue (les deux ids désignent la même mine).

    Deux rangs de clés : le nom ou l'alias complet (rang 0), et ses morceaux (« Pongkor and Cibaliung » →
    « Pongkor ») ou sa forme sans substance finale (rang 1). Une clé qui est le nom complet d'au moins une
    mine ne pointe que vers ces mines-là : « Mount Isa » désigne Mount Isa, pas « Ernest Henry and Mount
    Isa ».
    """
    idx = NameIndex()
    idx.twin_of = {c: w for w, c in (twins or {}).items()}
    prio: dict[str, int] = {}
    notable: set[str] = set()
    documented: set[str] = set()
    producing: set[str] = set()
    raw_w: dict[tuple[str, ...], tuple[set[str], set[str]]] = {}
    raw_c: dict[str, tuple[set[str], set[str], list[bool]]] = {}
    places: set[tuple[str, ...]] = set()

    def put_w(key, site_id, rank):
        raw_w.setdefault(key, (set(), set()))[rank].add(site_id)
        idx.keys_of.setdefault(site_id, set()).add(key)

    def add(site_id: str, name, main: bool = False) -> None:
        if not isinstance(name, str) or not name.strip():
            return
        name = html.unescape(" ".join(name.split()))
        if main and (pk := place_key(name)):
            # nom principal fait de mots de lieu, cherché tel quel par le collecteur : même nom ici
            put_w(pk, site_id, 0)
            places.add(pk)
        parts = [(name, 0)] + [(p.strip(), 1) for p in _SPLIT.split(name) if p and p.strip() != name]
        for part, rank in parts:
            if not part:
                continue
            if _CJK.search(part):
                k = cjk_key(part)
                if k:
                    e = raw_c.setdefault(k[0], (set(), set(), [True]))
                    idx.cjk_of.setdefault(site_id, set()).add(k[0])
                    e[rank].add(site_id)
                    e[2][0] = e[2][0] and k[1]
                continue
            key = word_key(part)
            if key:
                put_w(key, site_id, rank)
                tail = _tail_variant(key)
                if tail:
                    put_w(tail, site_id, 1)

    for s in watch or []:
        idx.sites[s.id] = (s.name, _num(s.lat), _num(s.lon), entities.norm_country(s.country))
        prio[s.id] = 100
        notable.add(s.id)
        documented.add(s.id)        # mine suivie : analysée par satellite (emprise mesurée)
        if getattr(s, "stage", None) == "production":
            producing.add(s.id)
        add(s.id, s.name, main=True)
    if cat is not None and len(cat):
        idx.catalog_rows = len(cat)
        cols = {c: (cat[c].tolist() if c in cat else [None] * len(cat))
                for c in ("id", "name", "aliases", "lat", "lon", "country", "footprint_km2",
                          "has_production", "wikidata", "icmm", "wikipedia")}
        for i, cid in enumerate(cols["id"]):
            cid = str(cid)
            sid = idx.twin_of.get(cid, cid)
            nm = cols["name"][i]
            fp, prod = _num(cols["footprint_km2"][i]) is not None, cols["has_production"][i] is True
            if fp or prod:                 # sous l'id suivi pour une jumelle : c'est la même mine
                documented.add(sid)
            if prod:
                producing.add(sid)
            if sid == cid:
                idx.sites[cid] = (nm if isinstance(nm, str) else None, _num(cols["lat"][i]),
                                  _num(cols["lon"][i]), entities.norm_country(cols["country"][i]))
                prio[cid] = 2 * fp + prod + isinstance(cols["wikidata"][i], str)
                if fp or prod or isinstance(cols["icmm"][i], str) or isinstance(cols["wikipedia"][i], str):
                    notable.add(cid)
            add(sid, nm, main=True)
            al = cols["aliases"][i]
            if isinstance(al, str):
                aliases = [a for a in dict.fromkeys(x.strip() for x in al.split(";")) if a]
                if len(aliases) > MAX_ALIASES:  # inventaire : seuls les alias qui ressemblent au nom
                    aliases = [a for a in aliases if catalog._similar(nm, a)]
                for a in aliases:
                    add(sid, a)
    if names is not None and len(names):
        for sid, nm in zip(names["site_id"].astype(str), names["name"], strict=False):
            sid = idx.twin_of.get(sid, sid)
            if sid in idx.sites:
                add(sid, nm)

    def distinct(ids: set[str]) -> tuple[str, ...] | None:
        """Homonymes réels : les fiches à moins de CLUSTER_KM d'une fiche gardée sont la même mine."""
        if len(ids) == 1:
            return tuple(ids)
        kept: list[str] = []
        for sid in sorted(ids, key=lambda x: (-prio.get(x, 0), x)):
            _, la, lo, _ = idx.sites.get(sid, (None, None, None, None))
            near = next((o for o in kept if la is not None and lo is not None
                         and (k := idx.sites.get(o)) and k[1] is not None and k[2] is not None
                         and _km(la, lo, k[1], k[2]) <= CLUSTER_KM), None)
            if near is not None:
                # même mine en double fiche : la requête de la fiche écartée reconnaît le nom (`match`)
                idx.same_as.setdefault(sid, near)
                continue
            kept.append(sid)
            if len(kept) > MAX_HOMONYMS:
                return None
        return tuple(kept)

    for key, (full, part) in raw_w.items():
        if _weak_key(key) and key not in places:
            # nom d'un seul mot distinctif (« Kibali gold » compris) : seulement les fiches étayées. Wikidata
            # compte des milliers de petites mines américaines « States Mine », « High Mine », « November
            # Mine » : sans ce filtre, chaque « United States » ou « November 25 » d'un titre deviendrait
            # un événement.
            full, part = full & notable, part & notable
            if not (full or part):
                continue
        d = distinct(full or part)
        if d is None:
            idx.dropped_generic += 1
            continue
        idx.words[key] = d
        idx.maxlen = max(idx.maxlen, len(key))
    for key, (full, part, weak) in raw_c.items():
        d = distinct(full or part)
        if d is None:
            idx.dropped_generic += 1
            continue
        idx.cjk.setdefault(key[:2], []).append((key, d, weak[0]))
    for lst in idx.cjk.values():
        lst.sort(key=lambda x: -len(x[0]))
    idx.notable = frozenset(notable)
    idx.documented = frozenset(documented)
    idx.producing = frozenset(producing)
    idx.place_keys = frozenset(k for k in places if k in idx.words)
    return idx


_cache: dict = {}
_cache_lock = threading.Lock()


def _mtime(p) -> int | None:
    try:
        return Path(str(p)).stat().st_mtime_ns
    except OSError:
        return None


def get_index() -> NameIndex:
    """Index partagé par le processus, reconstruit seulement si le catalogue, la liste suivie ou
    names.parquet ont changé (le démon tourne des semaines : il doit voir un catalogue reconstruit)."""
    sites_toml = files("strate.data").joinpath("sites.toml")
    key = (_mtime(catalog.OUT / "mines.parquet"), _mtime(NAMES_PATH), _mtime(sites_toml))
    with _cache_lock:
        if _cache.get("key") == key:
            return _cache["index"]
        catalog.load.cache_clear()
        entities.load.cache_clear()
        cat = catalog.load()
        try:
            watch = list(entities.load().sites.values())
        except Exception as e:  # liste suivie illisible : le catalogue suffit
            log.warning("liste suivie illisible (%s)", e)
            watch = []
        try:
            from strate import screener
            screener.twins.cache_clear()
            twins = screener.twins()
        except Exception as e:
            log.warning("jumelles liste suivie ↔ catalogue indisponibles (%s)", e)
            twins = {}
        names = None
        if NAMES_PATH.exists():
            try:
                names = pd.read_parquet(NAMES_PATH, columns=["site_id", "name"])
            except Exception as e:  # fichier en cours d'écriture ou abîmé : on fera sans ce cycle-ci
                log.warning("names.parquet illisible (%s)", e)
        idx = build_index(cat, watch, names, twins)
        _cache.update(key=key, index=idx)
        return idx


# == vérification des formats ===========================================================================
_ISO639_1 = frozenset("""aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs
cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia
id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh
mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc
sd se sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi
vo wa wo xh yi yo za zh zu""".split())
# ISO 639-2 (B et T) → 639-1 : les ~65 langues du flux GDELT traduit et de l'API documentaire
_LANG3 = dict(x.split(":") for x in """eng:en fra:fr fre:fr spa:es por:pt deu:de ger:de rus:ru zho:zh chi:zh
jpn:ja kor:ko ind:id tur:tr pol:pl ara:ar ita:it nld:nl dut:nl ukr:uk vie:vi afr:af sqi:sq alb:sq hye:hy
arm:hy aze:az bel:be ben:bn bos:bs bul:bg cat:ca hrv:hr ces:cs cze:cs dan:da est:et fin:fi glg:gl kat:ka
geo:ka ell:el gre:el guj:gu heb:he hin:hi hun:hu isl:is ice:is kan:kn kaz:kk lav:lv lit:lt mkd:mk mac:mk
msa:ms may:ms mal:ml mar:mr mon:mn nep:ne nor:no nob:nb nno:nn fas:fa per:fa pan:pa ron:ro rum:ro srp:sr
sin:si slk:sk slo:sk slv:sl som:so swa:sw swe:sv tam:ta tel:te tha:th urd:ur uzb:uz tgl:tl fil:tl amh:am
khm:km mya:my bur:my lao:lo bod:bo tib:bo mlt:mt kin:rw gle:ga cym:cy wel:cy eus:eu baq:eu kur:ku pus:ps
tgk:tg tuk:tk kir:ky hau:ha yor:yo ibo:ig zul:zu xho:xh mlg:mg hat:ht ltz:lb jav:jv sun:su epo:eo lat:la
fao:fo gla:gd fry:fy cos:co smo:sm mri:mi mao:mi sna:sn sot:st nya:ny orm:om tir:ti uig:ug yid:yi div:dv
tat:tt bak:ba che:ce lug:lg lin:ln wol:wo tsn:tn tso:ts ssw:ss ven:ve kon:kg""".split())
# Codes à 3 lettres sans équivalent à 2 lettres, mais réels (ISO 639-2/3) : gardés tels quels
_LANG3_ONLY = frozenset("""ceb haw hmn yue cmn nan ast nds arz apc ary arb kok mai bho sat mni doi ckb
nso gsw sco scn vec war ilo pam bcl hil min ace ban bug mad bjn tet tpi pap""".split())
_LANG_NAMES = {"english": "en", "french": "fr", "spanish": "es", "portuguese": "pt", "german": "de",
               "russian": "ru", "chinese": "zh", "japanese": "ja", "korean": "ko", "indonesian": "id",
               "turkish": "tr", "polish": "pl", "arabic": "ar", "italian": "it", "dutch": "nl",
               "swedish": "sv", "finnish": "fi", "norwegian": "no", "danish": "da", "czech": "cs",
               "slovak": "sk", "hungarian": "hu", "romanian": "ro", "bulgarian": "bg", "greek": "el",
               "ukrainian": "uk", "hebrew": "he", "hindi": "hi", "vietnamese": "vi", "thai": "th",
               "persian": "fa", "farsi": "fa", "urdu": "ur", "bengali": "bn", "tamil": "ta", "telugu": "te",
               "malay": "ms", "serbian": "sr", "croatian": "hr", "bosnian": "bs", "slovenian": "sl",
               "lithuanian": "lt", "latvian": "lv", "estonian": "et", "albanian": "sq", "macedonian": "mk",
               "georgian": "ka", "armenian": "hy", "azerbaijani": "az", "kazakh": "kk", "mongolian": "mn",
               "nepali": "ne", "swahili": "sw", "afrikaans": "af", "catalan": "ca", "galician": "gl",
               "icelandic": "is", "gujarati": "gu", "punjabi": "pa", "marathi": "mr", "malayalam": "ml",
               "kannada": "kn", "sinhala": "si", "sinhalese": "si", "somali": "so", "tagalog": "tl",
               "filipino": "tl", "belarusian": "be", "uzbek": "uz", "burmese": "my", "khmer": "km",
               "lao": "lo", "amharic": "am", "tibetan": "bo"}


def norm_lang(v, source: str | None = None) -> str | None:
    """Code ISO 639 (1 de préférence). Le flux anglais GDELT n'a pas d'info de traduction : 'en'."""
    if not isinstance(v, str) or not v.strip():
        return "en" if source == "gdelt_gkg" else None
    s = v.strip().lower().replace("_", "-").split("-")[0]
    if s in _ISO639_1:
        return s
    if s in _LANG3:
        return _LANG3[s]
    if s in _LANG_NAMES:
        return _LANG_NAMES[s]
    return s if s in _LANG3_ONLY else None   # « xyz » : pas un code ISO


def parse_date(v) -> datetime | None:
    """ISO 8601 (avec ou sans fuseau, « Z »), « AAAAMMJJHHMMSS » (GDELT) ou « AAAAMMJJTHHMMSSZ » (API)."""
    if not isinstance(v, str) or not v.strip():
        return None
    s = v.strip()
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})(\d{2})Z?", s)
    if m:
        s = "{}-{}-{}T{}:{}:{}".format(*m.groups())
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (d.replace(tzinfo=UTC) if d.tzinfo is None else d).astimezone(UTC)


def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _domain(url: str, given) -> str:
    d = given if isinstance(given, str) and given.strip() else (urlsplit(url).hostname or "")
    d = d.strip().lower().split(":")[0]
    return d[4:] if d.startswith("www.") else d


_TRACKING = re.compile(r"^(utm_|fbclid|gclid|ocid|cmpid|mc_|icid|ref$|src$|source$|feature$)")


def canon_url(url: str) -> str:
    """URL comparable : sans schéma, « www. », fragment, paramètres de suivi ni « / » final."""
    u = urlsplit(url.strip())
    host = (u.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    q = urlencode(sorted((k, v) for k, v in parse_qsl(u.query, keep_blank_values=True)
                         if not _TRACKING.match(k.lower())))
    return f"{host}{u.path.rstrip('/')}" + (f"?{q}" if q else "")


_LABEL = re.compile(r"[^\W_](?:[\w-]{0,61}[^\W_])?")


def _valid_host(host: str) -> bool:
    """Nom de domaine propre : au moins deux étiquettes non vides (« a..com », « .com » refusés), chacune
    de 1 à 63 caractères alphanumériques ou tirets (Unicode admis : domaines internationalisés), et une
    extension qui contient une lettre (une adresse IP n'est pas un site d'actualité)."""
    labels = host.split(".")
    return (len(labels) >= 2 and len(host) <= 253 and all(_LABEL.fullmatch(x) for x in labels)
            and not labels[-1].isdigit() and any(ch.isalpha() for ch in labels[-1]))


def _split_list(v) -> list[str]:
    return [x.strip() for x in v.split(";") if x.strip()] if isinstance(v, str) else []


def validate(row, now: datetime | None = None) -> tuple[dict | None, str | None]:
    """Document brut → (document propre, None) ou (None, motif de rejet)."""
    now = now or datetime.now(UTC)
    r = dict(row)
    url = (r.get("url") or "").strip()
    try:
        u = urlsplit(url)
        host = u.hostname or ""
        u.port  # noqa: B018 — lève ValueError si le port est hors bornes (« :99999 »)
    except ValueError:
        return None, "URL invalide"
    if u.scheme not in ("http", "https") or not _valid_host(host) or re.search(r"\s", url) \
            or len(url) > 2048:
        return None, "URL invalide"
    pub = parse_date(r.get("published_at"))
    if pub is None:
        return None, "date illisible" if r.get("published_at") else "date absente"
    if pub > now + FUTURE_SLACK:
        return None, "date future"
    if pub < MIN_DATE:
        return None, "date avant 2015"
    t = r.get("title")
    title = " ".join(html.unescape(t if isinstance(t, str) else "").split())
    if not title:
        return None, "titre vide"
    lang = norm_lang(r.get("lang"), r.get("source"))
    if lang is None:
        return None, "langue non ISO" if r.get("lang") else "langue absente"
    locs = []
    if r.get("locations"):
        try:
            locs = json.loads(r["locations"]) if isinstance(r["locations"], str) else r["locations"]
        except (TypeError, ValueError):
            return None, "lieux illisibles"
        if not isinstance(locs, list) or not all(isinstance(x, dict) for x in locs):
            return None, "lieux illisibles"
        for loc in locs:
            # champs texte attendus : un nom numérique (source future, bogue) ne doit rien faire planter
            for k in ("name", "country", "cc"):
                if loc.get(k) is not None and not isinstance(loc[k], str):
                    loc[k] = str(loc[k])
            try:
                loc["type"] = int(loc["type"]) if loc.get("type") is not None else None
            except (TypeError, ValueError):
                loc["type"] = None
            la, lo = loc.get("lat"), loc.get("lon")
            if la is None and lo is None:
                continue
            la, lo = _num(la), _num(lo)
            if la is None or lo is None or not (-90 <= la <= 90 and -180 <= lo <= 180):
                return None, "lat/lon hors bornes"
    themes = [t.split(",")[0].strip() for t in (r.get("themes") or "").split(";") if t.strip()]
    return {
        "id": r["id"], "source": r.get("source"), "url": url, "domain": _domain(url, r.get("domain")),
        "lang": lang, "title": title, "published": pub, "published_at": _iso(pub), "themes": themes,
        "locations": locs, "organizations": _split_list(r.get("organizations")),
        "persons": _split_list(r.get("persons")), "hint_site": r.get("hint_site") or None,
    }, None


def title_key(title: str) -> str | None:
    t = words(title)
    return " ".join(t) if len(t) >= DEDUP_MIN_WORDS else None


# == classement =========================================================================================
TYPE_PRIORITY = ["accident", "securite", "environnement", "social", "juridique", "finance", "operations"]

# Thèmes GDELT (préfixes) → (type, poids). Calculés sur tout l'article : poids modestes, un seul thème
# fréquent (KILL, LEGISLATION…) ne suffit pas à typer ; sans mot-clé, le seuil est THEME_ONLY_MIN (1,5).
THEMES: list[tuple[str, str, float]] = [
    ("accident", "UNSAFE_WORK_ENVIRONMENT", 1.0), ("accident", "CRISISLEX_T08_MISSINGFOUNDTRAPPED", 1.0),
    ("accident", "MANMADE_DISASTER_IMPLIED", 0.4), ("accident", "MANMADE_DISASTER", 0.8),
    ("accident", "KILL", 0.7), ("accident", "CRISISLEX_T03_DEAD", 0.6),
    ("accident", "CRISISLEX_T02_INJURED", 0.6), ("accident", "WOUND", 0.4),
    ("accident", "NATURAL_DISASTER_LANDSLIDE", 0.8), ("accident", "EVACUATION", 0.4),
    ("social", "STRIKE", 1.0), ("social", "PROTEST", 0.8), ("social", "ECON_UNIONS", 0.8),
    ("social", "TAX_FNCACT_PROTESTER", 0.6), ("social", "TAX_FNCACT_STRIKER", 0.8),
    ("social", "BLOCKADE", 0.8),
    ("juridique", "LEGISLATION", 0.5), ("juridique", "ARREST", 0.8), ("juridique", "TRIAL", 0.8),
    ("juridique", "SANCTIONS", 0.8), ("juridique", "WB_849", 0.8), ("juridique", "WB_840_JUSTICE", 0.4),
    ("juridique", "WB_2531_INSPECTIONS_LICENSING_AND_PERMITS", 0.8), ("juridique", "SEIZE", 0.6),
    ("juridique", "ECON_NATIONALIZ", 1.0), ("juridique", "WB_1014_CRIMINAL_JUSTICE", 0.4),
    ("juridique", "WB_2019_ANTI_CORRUPTION_LEGISLATION", 0.5), ("juridique", "WB_2020_BRIBERY_FRAUD", 0.5),
    ("environnement", "ENV_WATERWAYS", 0.8), ("environnement", "ENV_DEFORESTATION", 0.8),
    ("environnement", "ENV_SPECIESENDANGERED", 0.6), ("environnement", "ENV_CLIMATECHANGE", 0.3),
    ("environnement", "POLLUTION", 1.0), ("environnement", "WATER_SECURITY", 0.5),
    ("environnement", "WB_1790_ENVIRONMENTAL_HEALTH", 0.6),
    ("finance", "ECON_BANKRUPTCY", 1.0), ("finance", "ECON_IPO", 1.0), ("finance", "ECON_MERGER", 1.0),
    ("finance", "ECON_ACQUISITION", 1.0), ("finance", "ECON_TAKEOVER", 1.0),
    ("finance", "ECON_EARNINGSREPORT", 0.8),
    ("finance", "ECON_STOCKMARKET", 0.5), ("finance", "ECON_DEBT", 0.4),
    ("securite", "TERROR", 0.8), ("securite", "KIDNAP", 1.0), ("securite", "ARMEDCONFLICT", 0.7),
    ("securite", "TAX_FNCACT_MILITANT", 0.8), ("securite", "TAX_FNCACT_REBEL", 0.8),
    ("securite", "REBELLION", 0.8),
    ("securite", "WB_2445_NON_STATE_SECURITY_ACTORS", 0.7), ("securite", "WB_2462_POLITICAL_VIOLENCE", 0.5),
    ("securite", "ASSASSINATION", 0.8), ("securite", "TAX_TERROR_GROUP", 0.8), ("securite", "MILITARY", 0.3),
]
MINING_THEMES = ("ENV_MINING", "TAX_FNCACT_MINER", "WB_1698_COAL_MINING")   # précis (WB_1699 : trop large)
_ENV_DIRTY = ("POLLUT", "TOXIC", "SPILL", "CONTAMIN", "WASTE")   # ENV_* de pollution, pas d'énergie

# Mots-clés du titre d'origine, par type et par langue (radicaux : « effondr » couvre « effondrement »).
# Les articles de l'API documentaire n'ont pas de thèmes GDELT (KILL, CRISISLEX_T03_DEAD…) : les verbes
# « mourir » (die, dies, died, meurt, muere, morre, stirbt, muore, zmarł, умер…) doivent être dans le titre.
# Exclus à dessein : turc « öldü » (replié « oldu », aussi « il est devenu »), « meurtre » (« meurt= »
# exact), indonésien « mati » (aussi « en panne, à l'arrêt ») ; « tewas » et « meninggal » suffisent.
# Arabe « توفي= » en mot entier : en radical, il prenait « توفير » (fournir) et « توفيق » (un prénom).
# Les verbes qui meurent aussi au figuré (« strike dies down », « deal dies », « muere el acuerdo »)
# ne comptent qu'avec une personne dans le titre : voir _NEEDS_PERSON.
# Mots de 3 lettres ou moins : mot exact. Allemand/néerlandais : aussi au milieu d'un composé.
KEYWORDS: dict[str, dict[str, str]] = {
    "accident": {
        "en": "accident collaps cave-in explosion blast fatal fatality killed killing dead deadly death "
              "die dies died dying injur trapped rescue flood landslide mudslide rockfall rockburst victim "
              "bodies tragedy disaster",
        "fr": "accident effondr eboul explosion incendie mort deces decede meurt= meurent tue tues tuee "
              "blesse victime disparu piege sauvet inondation grisou drame catastrophe glissement",
        "es": "accidente derrumbe colapso explosion incendio muert muere murio murieron fallec "
              "herid atrapad rescate victima desaparecid inundacion deslave deslizamiento tragedia "
              "siniestro",
        "pt": "acidente desabamento desmoronamento colapso explosao incendio mort morre falec ferid "
              "soterrad resgate vitima desaparecid inundacao deslizamento tragedia rompimento",
        "de": "unfall ungluck einsturz eingesturzt explosion brand tot= tote toten getotet stirbt sterben "
              "gestorben ums_leben verletzt verschuttet eingeschlossen rettung vermisst flutung uberflut",
        "it": "incidente crollo esplosione incendio morti morto morta muore muoiono decedut decesso "
              "vittim ferit intrappolat soccors dispersi frana alluvione tragedia",
        "pl": "wypadek wypadk katastrof zawal wybuch pozar zgin zmarl smierc smiertel nie_zyje ofiar "
              "ranny ranni zasypan uwiezi ratownic tapniecie metan",
        "ru": "авари обрушен взрыв пожар погиб гибел умер= умерл умира скончал смерт жертв пострада "
              "ранен заблокирован спасат затоплен трагеди катастроф выброс",
        "tr": "kaza gocuk patlama yangin olu olum oldurul hayatini yarali mahsur kurtarma facia felaket "
              "grizu",
        "id": "kecelakaan longsor ambruk runtuh ledakan kebakaran tewas meninggal korban terjebak "
              "tertimbun evakuasi banjir tragedi",
        "ar": "حادث انهيار انفجار حريق مقتل قتلى وفاة مصرع مات يموت توفي= توفيت= حتفه جرحى "
              "إصابة محاصرين إنقاذ فيضان",
        "zh": "事故 坍塌 塌方 冒顶 爆炸 火灾 起火 死亡 遇难 身亡 罹难 丧生 致死 殉职 受伤 被困 救援 透水 "
              "瓦斯 矿难 失踪",
        "ja": "事故 崩落 落盤 爆発 火災 死亡 死者 死去 亡くな 犠牲 負傷 閉じ込め 救助 浸水 遭難 行方不明",
        "ko": "사고 붕괴 매몰 폭발 화재 사망 숨져 숨진 숨졌 목숨 부상 고립 구조 침수 실종",
    },
    "social": {
        "en": "strike striker walkout protest demonstrat blockade roadblock unionist trade_union "
              "labour_union "
              "labor_union trade_unions industrial_action wage wages picket lockout layoff unrest riot",
        "fr": "greve greviste manifest blocage syndica conflit_social revendication licenciement colere "
              "debrayage piquet",
        "es": "huelga paro protesta manifestacion manifestante bloqueo sindica conflicto_laboral despido "
              "comuner",
        "pt": "greve grevista paralisacao protesto manifestacao manifestante bloqueio sindica demissao "
              "demissoes",
        "de": "streik protest demonstr blockade gewerkschaft arbeitskampf tarifstreit kundgebung entlassung",
        "it": "sciopero scioperi protest manifestazion presidio blocco sindacat vertenza licenziament",
        "pl": "strajk protest manifestac blokad zwiazkow spor_zbiorowy pikiet zwolnien",
        "ru": "забастов стачк протест митинг пикет профсоюз блокад бастуют бастующ голодовк сокращен",
        "tr": "grev eylem protesto direnis sendika yuruyus",
        "id": "mogok unjuk_rasa demo demonstrasi protes blokade serikat_pekerja phk buruh",
        "ar": "إضراب احتجاج مظاهرة تظاهر اعتصام نقابة",
        "zh": "罢工 抗议 示威 工会 维权 讨薪 游行 封锁",
        "ja": "ストライキ 抗議 デモ 労働組合 労組 封鎖",
        "ko": "파업 시위 항의 노조 노동조합 농성 봉쇄",
    },
    "juridique": {
        "en": "court lawsuit sue sues sued judge ruling verdict fined penalt permit licence license "
              "regulator regulat ban bans banned sanction investigat probe charged arrest illegal "
              "injunction appeal "
              "tribunal legislation law laws royalt tax nationaliz expropriat revok revoc moratorium decree "
              "inspection corruption bribe fraud",
        "fr": "tribunal justice juge proces plainte condamn amende permis licence autorisation regulation "
              "reglementation interdi sanction enquete arrestation illegal illicite decret loi code_minier "
              "redevance fiscal retrait expropri nationalis moratoire convention_miniere corruption",
        "es": "tribunal juzgado juez demanda fallo sentencia multa permiso licencia concesion regulac "
              "normativ prohibi sancion investigac arrest detenid ilegal decreto ley regalia impuesto "
              "expropiac nacionaliz moratoria fiscalia denuncia clausura corrupcion",
        "pt": "tribunal justica juiz processo acao_judicial condena multa licenca licenciamento autoriza "
              "regulac proib sancao sancoes investiga prisao preso ilegal decreto lei royalties imposto "
              "cfem embargo interdi fiscaliza denuncia corrupcao ministerio_publico",
        "de": "gericht klage urteil richter strafe bussgeld genehmigung lizenz verbot sanktion ermittl "
              "festnahme verhaftet illegal gesetz verordnung steuer enteign verstaatlich behorde",
        "it": "tribunale giudice sentenza causa denuncia multa sanzion permesso licenza autorizzazion "
              "concession divieto indagin inchiesta arrest illegal abusiv decreto legge sequestr corruzion",
        "pl": "sad sadu sadow wyrok prokurat kara grzywn koncesj pozwolen zezwolen zakaz sankcj sledztw "
              "zatrzyman aresztow nielegaln ustaw rozporzadzen podatek nacjonaliz wywlaszcz kontrol nik",
        "ru": "суд иск приговор штраф лиценз разрешен запрет санкци расследован арест задержан незаконн "
              "закон постановлен налог прокурат национализ изъят ростехнадзор проверк",
        "tr": "mahkeme dava ceza ruhsat izin yasak yaptirim sorusturma gozalti tutuklan kacak yasa= kanun "
              "vergi kamulastir iptal",
        "id": "pengadilan gugatan vonis hakim denda izin iup larangan sanksi penyelidikan penyidikan "
              "ditangkap tersangka ilegal peti undang regulasi pajak royalti kpk korupsi kejaksaan "
              "nasionalisasi",
        "ar": "محكمة دعوى غرامة ترخيص رخصة حظر عقوبات تحقيق اعتقال قانون ضريبة تأميم",
        "zh": "法院 诉讼 起诉 判决 罚款 处罚 许可证 采矿权 禁令 制裁 调查 逮捕 拘留 "
              "非法 违法 法律 条例 税 国有化 整改 监管",
        "ja": "裁判 訴訟 提訴 判決 罰金 許可 禁止 制裁 調査 逮捕 違法 法律 規制 税",
        "ko": "법원 소송 판결 벌금 허가 인허가 금지 제재 조사 체포 불법 법률 규제 세금",
    },
    "environnement": {
        "en": "tailings pollut contaminat spill leak toxic cyanide mercury acid_drainage environmental "
              "deforestation emissions dust waste rehabilitat reclamat biodiversity wildlife sludge "
              "sediment arsenic poisoning",
        "fr": "residus digue pollu contamin deversement fuite toxique cyanure mercure environnement "
              "deforestation rejets boues rehabilitation poussiere dechets arsenic",
        "es": "relave tranque contamina derrame fuga toxic cianuro mercurio ambiental medioambient "
              "deforestac residuos vertido polvo glaciar arsenico",
        "pt": "rejeito barragem poluic polui contamin vazamento toxic cianeto mercurio ambiental "
              "desmatamento residuos lama poeira arsenio",
        "de": "abraum ruckstand schlamm absetzbecken umwelt verschmutz kontamin giftig zyanid cyanid "
              "quecksilber abwasser staub rodung renaturier",
        "it": "sterili discarica inquinament contamina sversament tossic cianuro mercurio ambiental rifiuti "
              "polveri bonifica diga",
        "pl": "odpady hald skazen zanieczyszcz wyciek toksyczn cyjan rtec srodowisk ekolog poflotac pyl",
        "ru": "хвост загрязн разлив утечк токсичн цианид ртут экологи отход вырубк",
        "tr": "atik siyanur civa kirlil kirlet zehir sizinti cevre ekolojik toz",
        "id": "limbah tailing pencemaran tercemar cemar bocor beracun sianida merkuri lingkungan deforestasi "
              "reklamasi debu",
        "ar": "مخلفات نفايات تلوث ملوث تسرب سام سيانيد زئبق بيئ",
        "zh": "尾矿 污染 泄漏 有毒 氰化物 汞 环保 环境 生态 废水 粉尘 废石 溃坝",
        "ja": "鉱滓 尾鉱 汚染 流出 有毒 シアン 水銀 環境 廃水 粉じん 鉱毒",
        "ko": "광미 오염 유출 유독 시안 수은 환경 폐수 분진",
    },
    "finance": {
        "en": "acquisit acquire takeover merger merge buyout bid bids deal deals stake= stakes divest "
              "financing loan loans funding raises= raised= offering ipo shares= stock= stocks= earnings "
              "profit revenue results dividend bankrupt insolven creditor debt writedown impairment "
              "valuation investment invests "
              "joint_venture jv private_placement sells sold sale",
        "fr": "rachat rachete acquisition acquiert fusion cession cede opa participation financement emprunt "
              "pret levee bourse resultat benefice chiffre_daffaires dividende faillite redressement "
              "liquidation dette creanc investiss coentreprise valorisation ipo",
        "es": "adquisicion adquiere compra venta vende fusion oferta participacion financiamiento "
              "financiacion prestamo credito bolsa acciones resultados utilidad ganancia beneficio "
              "ingresos dividendo quiebra insolvencia concurso deuda inversion invierte empresa_conjunta ipo",
        "pt": "aquisicao adquire compra venda vende fusao oferta participacao financiamento emprestimo "
              "credito bolsa acoes resultado lucro receita dividendo falencia recuperacao_judicial "
              "insolvencia divida investimento investe joint_venture ipo",
        "de": "ubernahme ubernimmt kauft verkauf fusion beteiligung finanzierung kredit anleihe borse aktie "
              "gewinn umsatz quartalszahlen bilanz dividende insolvenz pleite konkurs schulden investition "
              "investiert ipo",
        "it": "acquisizion acquista cessione cede vendita vende fusione offerta partecipazion finanziament "
              "prestito credito borsa azioni utile ricavi fatturato dividend fallimento insolvenz debito "
              "investiment opa ipo",
        "pl": "przejec przejmie kupi zakup sprzedaz sprzeda fuzj udzial finansowan kredyt pozyczk obligacj "
              "gield akcj zysk przychod wynik dywidend upadlosc restrukturyzac zadluz dlug inwestyc ipo",
        "ru": "поглощен приобрет покупк купит купил продаж продаст продал слиян доля долю доли "
              "финансирован кредит займ облигац бирж акци прибыл выручк дивиденд банкрот долг инвестиц "
              "сделк ipo",
        "tr": "satin_al devral birlesme satis hisse finansman kredi borc borsa kar gelir temettu iflas "
              "konkordato yatirim ortak_girisim halka_arz",
        "id": "akuisisi mengakuisisi caplok merger divestasi saham pendanaan pinjaman kredit obligasi bursa "
              "laba rugi pendapatan dividen pailit bangkrut utang investasi ipo menjual dijual membeli",
        "ar": "استحواذ اندماج صفقة بيع شراء حصة تمويل قرض أسهم بورصة أرباح إيرادات إفلاس ديون استثمار",
        "zh": "收购 并购 兼并 合并 出售 转让 股权 融资 贷款 债券 上市 股价 股票 "
              "利润 净利 营收 业绩 财报 分红 破产 债务 投资 合资 ipo",
        "ja": "買収 合併 売却 譲渡 出資 株式 融資 上場 株価 利益 決算 配当 破産 債務 投資 合弁 ipo",
        "ko": "인수 합병 매각 지분 투자 융자 대출 상장 주가 실적 이익 배당 파산 부채 합작 ipo",
    },
    "operations": {
        "en": "production produce producing output expansion expand shutdown shut_down shut shuts closure "
              "close= closes= closed= suspend suspension halt halts halted restart resume reopen ramp_up "
              "commissioning first_gold "
              "first_ore gold_pour throughput shipment care_and_maintenance force_majeure guidance "
              "mine_life reserves drill exploration discovery feasibility construction development "
              "outage stoppage idle",
        "fr": "production produire produit= extraction expansion agrandissement extension arret fermeture "
              "suspension suspend reprise redemarrage relance mise_en_service mise_en_production premier_or "
              "tonnage minerai exploration forage decouverte faisabilite construction developpement "
              "exportation maintenance",
        "es": "produccion producir produjo extraccion expansion ampliacion cierre cierra paraliza "
              "suspension suspende detiene reanuda reinicio reapertura puesta_en_marcha toneladas "
              "exploracion perforacion descubrimiento factibilidad construccion desarrollo exportacion "
              "mantenimiento",
        "pt": "producao produz extracao expansao ampliacao fechamento encerramento paralisa suspensao "
              "suspende retoma retomada reinicio reabertura toneladas exploracao sondagem descoberta "
              "viabilidade construcao exportacao manutencao",
        "de": "produktion forderung fordert ausbau erweiterung stilllegung schliessung schliesst stillgelegt "
              "stopp wiederaufnahme neustart inbetriebnahme tonnen exploration bohrung entdeckung "
              "machbarkeit bau export",
        "it": "produzion estrazion espansion ampliamento chiusura chiude sospension sospende fermo "
              "riapertura ripresa riavvio tonnellat esplorazion perforazion scoperta fattibilita costruzion",
        "pl": "produkcj wydobyc wydobyw rozbudow ekspansj zamkniec zamknie likwidac wstrzym wznowien wznow "
              "uruchomien poszukiwan odwiert odkryc budow eksport",
        "ru": "добыч производств выпуск расширен закрыт закро остановк останов приостанов возобнов запуск "
              "запуст ввод тонн разведк бурен строительств экспорт отгрузк модерниз мощност",
        "tr": "uretim uret genisleme kapat kapan durdur askiya yeniden faaliyet ton tonluk sondaj kesif "
              "insaat ihracat kapasite",
        "id": "produksi ekspansi perluasan penutupan ditutup tutup penghentian dihentikan berhenti "
              "beroperasi operasi dibuka ton eksplorasi pengeboran penemuan konstruksi pembangunan ekspor "
              "smelter hilirisasi kapasitas",
        "ar": "إنتاج توسعة توسيع إغلاق وقف توقف تعليق استئناف تشغيل طن استكشاف تنقيب اكتشاف تصدير",
        "zh": "产量 生产 投产 扩产 扩建 停产 关闭 关停 停工 复产 复工 重启 开工 "
              "产能 吨 勘探 钻探 发现 建设 出口 达产",
        "ja": "生産 産出 増産 減産 拡張 操業 停止 閉山 再開 稼働 着工 探鉱 試掘 発見 建設 輸出 トン",
        "ko": "생산 증산 감산 확장 증설 가동 중단 폐쇄 폐광 재개 착공 탐사 시추 발견 건설 수출 톤",
    },
    "securite": {
        "en": "attack attacker gunmen gunman armed militant rebel insurgent terror jihadist kidnap abduct "
              "hostage ambush bandit raid raids militia shooting shot gunfire insecurity army soldier "
              "violence clash looting robbery",
        "fr": "attaque assaillant armes terroris jihadist djihadist enlev otage embuscade bandit rebel "
              "milice fusillade tirs insecurite violence affrontement pillage m23",
        "es": "ataque atacan armado terroris secuestr rehen emboscada bandid rebelde guerrill milicia "
              "tiroteo disparos inseguridad violencia enfrentamiento narcotrafic crimen_organizado "
              "extorsion cartel saqueo robo",
        "pt": "ataque armado terroris sequestr refem emboscada bandid rebelde guerrilh milicia tiroteio "
              "disparos inseguranca violencia confronto faccao roubo assalto",
        "de": "angriff uberfall bewaffnet terror entfuhr geisel hinterhalt rebellen miliz schusse gewalt "
              "unsicherheit raub plunder",
        "it": "attacco assalto armati terroris rapiment rapit ostagg imboscata banditi ribelli milizi "
              "sparator violenz scontri rapina",
        "pl": "atak napad uzbrojen terroryst porwan porwal zakladni zasadzk bandyt rebeli milicj strzelanin "
              "przemoc kradziez",
        "ru": "нападен атак вооружен террорист теракт похищ заложник засад бандит боевик повстанц стрельб "
              "обстрел насили хищен грабеж ограблен",
        "tr": "saldiri silahli teror kacir rehine pusu haydut isyanci militan catisma siddet soygun",
        "id": "serangan menyerang bersenjata teroris penculikan diculik sandera penyergapan bandit "
              "pemberontak kkb milisi penembakan ditembak kekerasan bentrok perampokan",
        "ar": "هجوم مسلح إرهاب خطف اختطاف رهائن كمين عصابة متمردين ميليشيا عنف اشتباك سرقة",
        "zh": "袭击 武装 恐怖 绑架 劫持 人质 伏击 土匪 叛军 民兵 枪击 暴力 冲突 抢劫",
        "ja": "襲撃 武装 テロ 誘拐 拉致 人質 待ち伏せ 反政府 民兵 銃撃 暴力 強盗",
        "ko": "습격 공격 무장 테러 납치 인질 매복 반군 민병 총격 폭력 강도",
    },
}
_COMPOUND_LANGS = {"de", "nl", "sv", "da", "no", "fi"}


def _compile_keywords() -> dict[str, list[tuple[str, str, str]]]:
    """lang → [(type, mode, mot replié)].

    Modes : cjk (sous-chaîne), ar (mot arabe, avec ou sans préfixe و ب ل ف ك), phrase (mots exacts
    consécutifs), exact (mot entier : 4 lettres ou moins, ou suffixe « = »), sub (dans un composé
    allemand/néerlandais, 5 lettres ou plus), prefix (radical : « effondr » → « effondrement »).
    """
    out: dict[str, list[tuple[str, str, str]]] = {}
    for typ, by_lang in KEYWORDS.items():
        for lang, txt in by_lang.items():
            for raw in txt.split():
                exact = raw.endswith("=")
                kw = raw.rstrip("=").replace("_", " ").replace("-", " ")
                if _CJK.search(kw):
                    out.setdefault(lang, []).append((typ, "cjk", _cjk_fold(kw)))
                    continue
                f = " ".join(words(kw))
                if not f:
                    continue
                if _ARABIC.search(f):
                    mode = "ar_mot" if exact else "ar"
                elif " " in f:
                    mode = "phrase"
                elif exact or len(f) <= 4:
                    mode = "exact"
                elif lang in _COMPOUND_LANGS:
                    mode = "sub"
                else:
                    mode = "prefix"
                out.setdefault(lang, []).append((typ, mode, f))
    return out


_KW = _compile_keywords()
THEME_ONLY_MIN = 1.5     # sans mot-clé dans le titre, il faut plusieurs thèmes concordants
# mot-clé à double sens : ignoré si l'un de ces mots est dans le titre (« earthquake strikes »)
_FALSE_SENSE = {"strike": {"earthquake", "quake", "lightning", "drone", "missile", "air", "airstrike",
                           "military", "tornado", "storm", "hurricane", "typhoon"}}


# Verbes « mourir » qui meurent aussi au figuré : « Mine strike dies down », « Gold rally dies »,
# « Muere el acuerdo », « Projekt stirbt ». Ils ne comptent que si le titre nomme une personne (mineur,
# ouvrier, homme…) ou un nombre (« 3 die »). Repli : « muoiono », « murieron », « умер » restent libres,
# leur sujet est presque toujours humain.
_PERSON: dict[str, frozenset[str]] = {lang: frozenset(txt.split()) for lang, txt in {
    "en": "worker workers miner miners man men woman women people person persons employee employees "
          "contractor contractors boy boys girl girls child children teen teenager driver operator "
          "operators digger diggers labourer labourers laborer laborers artisanal one two three four five "
          "six seven eight nine ten dozen dozens several",
    "fr": "mineur mineurs ouvrier ouvriers travailleur travailleurs homme hommes femme femmes personne "
          "personnes employe employes salarie salaries enfant enfants jeune jeunes orpailleur orpailleurs "
          "sous-traitant deux trois quatre cinq six sept huit neuf dix plusieurs",
    "es": "minero mineros trabajador trabajadores obrero obreros operario operarios persona personas "
          "hombre hombres mujer mujeres joven jovenes nino nina empleado empleados dos tres cuatro cinco "
          "seis siete ocho nueve diez varios",
    "pt": "mineiro mineiros trabalhador trabalhadores operario operarios pessoa pessoas homem homens "
          "mulher jovem crianca funcionario funcionarios garimpeiro garimpeiros dois duas tres quatro "
          "cinco seis sete oito nove dez",
    "de": "bergmann bergleute kumpel arbeiter mann frau menschen person personen mitarbeiter kind "
          "jugendlicher zwei drei vier funf sechs sieben acht neun zehn",
    "it": "minatore minatori operaio operai lavoratore lavoratori uomo donna persona persone giovane "
          "dipendente dipendenti due tre quattro cinque sei sette otto nove dieci",
}.items()}
_NEEDS_PERSON: dict[str, frozenset[str]] = {
    "en": frozenset({"die", "dies", "dying"}), "fr": frozenset({"meurt", "meurent"}),
    "es": frozenset({"muere", "murio"}), "pt": frozenset({"morre"}),
    "de": frozenset({"stirbt", "sterben"}), "it": frozenset({"muore", "decedut"}),
}


_PERSON_WINDOW = 2  # « Muere el acuerdo minero » : « minero » y est un adjectif, trois mots plus loin


def _is_person(lang: str, t: str) -> bool:
    # composés allemands : « Bergarbeiter », « Grubenmann »
    return t.isdigit() or t in _PERSON.get(lang, frozenset()) or (
        lang == "de" and t.endswith(("arbeiter", "leute", "mann")))


def _verb_with_person(lang: str, mode: str, kw: str, toks: list[str]) -> bool:
    """Le verbe figure-t-il à deux mots au plus d'une personne ou d'un nombre ?"""
    for i, t in enumerate(toks):
        hit = t == kw if mode == "exact" else t.startswith(kw) if mode == "prefix" else kw in t
        near = range(max(0, i - _PERSON_WINDOW), min(len(toks), i + _PERSON_WINDOW + 1))
        if hit and any(_is_person(lang, toks[j]) for j in near if j != i):
            return True
    return False


def _kw_hit(mode: str, kw: str, toks: list[str], tokset: set[str], joined: str, cjk_text: str) -> bool:
    if mode == "cjk":
        return kw in cjk_text
    if mode == "exact":
        return kw in tokset
    if mode == "prefix":
        return any(t.startswith(kw) for t in toks)
    if mode == "phrase":
        return f" {kw} " in f" {joined} "
    if mode in ("ar", "ar_mot"):
        # « بيع » est aussi dans « الطبيعي » : mot entier pour les mots courts et les mots marqués « = »
        short = mode == "ar_mot" or len(kw) <= 3
        return any(t == kw or (t[0] in "وبلفك" and t[1:] == kw)
                   or (not short and (t.startswith(kw) or (t[0] in "وبلفك" and t[1:].startswith(kw))))
                   for t in toks)
    return any(kw in t for t in toks)      # sub : composés germaniques


def classify(title: str, themes: list[str], lang: str | None) -> tuple[str, float, dict]:
    """Type d'événement (le plus fort), force du classement (0–1) et détail.

    Score = mots-clés du titre (2 par mot, 2 mots au plus) + thèmes GDELT (plafonnés à 2). Sans mot-clé,
    il faut au moins 1,5 de thèmes ; sinon « autre ». Égalité : le type qui a un mot-clé dans le titre
    (preuve plus précise que les thèmes de tout l'article), puis l'ordre de gravité. Mots-clés de la
    langue du titre seulement ; langue sans dictionnaire (vietnamien, hindi…) : thèmes seuls, calculés
    par GDELT sur la traduction anglaise — les faux amis d'un autre dictionnaire (« bán » vietnamien lu
    « ban ») feraient plus de mal que de bien.
    """
    toks = words(title)
    tokset = set(toks)
    joined = " ".join(toks)
    cjk_text = _cjk_fold(title)
    hits: dict[str, set[str]] = {}
    needs = _NEEDS_PERSON.get(lang or "", frozenset())
    for typ, mode, kw in _KW.get(lang or "", ()):
        if kw in _FALSE_SENSE and tokset & _FALSE_SENSE[kw]:
            continue
        if kw in needs and not _verb_with_person(lang or "", mode, kw, toks):
            continue
        if _kw_hit(mode, kw, toks, tokset, joined, cjk_text):
            hits.setdefault(typ, set()).add(kw)
    th_score: dict[str, float] = {}
    th_hits: dict[str, list[str]] = {}
    for t in themes:
        for typ, prefix, w in THEMES:
            if t.startswith(prefix):
                th_score[typ] = th_score.get(typ, 0.0) + w
                th_hits.setdefault(typ, []).append(t)
                break
        else:
            if t.startswith("ENV_") and any(x in t for x in _ENV_DIRTY):
                th_score["environnement"] = th_score.get("environnement", 0.0) + 0.8
                th_hits.setdefault("environnement", []).append(t)
    scores = {typ: round(2.0 * min(len(hits.get(typ, ())), 2) + min(th_score.get(typ, 0.0), 2.0), 2)
              for typ in TYPE_PRIORITY}
    # égalité de score : le type appuyé par un mot du titre l'emporte sur un type venu des seuls thèmes
    # (calculés sur tout l'article), puis l'ordre de gravité
    best = max(TYPE_PRIORITY, key=lambda typ: (scores[typ], bool(hits.get(typ)), -TYPE_PRIORITY.index(typ)))
    detail = {"scores": {k: v for k, v in scores.items() if v}, "mots": sorted(hits.get(best, ())),
              "themes": sorted(set(th_hits.get(best, ())))}
    if not hits.get(best) and th_score.get(best, 0.0) < THEME_ONLY_MIN:
        return "autre", 0.7, {**detail, "mots": [], "themes": []}
    kw, th = bool(hits.get(best)), th_score.get(best, 0.0) > 0
    strength = 1.0 if kw and th else 0.9 if kw else 0.7 + 0.075 * min(th_score[best], 2.0)
    return best, round(strength, 3), detail


# == rattachement à une mine ============================================================================
def _is_context(t: str) -> bool:
    """Mot qui confirme qu'on parle d'une mine : « mine », « mina », « шахта », une substance, un composé
    germanique (« Goldmine », « Kohlegrube »)."""
    if t in MINE_WORDS or t in COMMODITY:
        return True
    return t not in _NOT_COMPOUND and len(t) > 6 and t.endswith(_COMPOUND_TAILS)


def _contains(big: tuple[str, ...], small: tuple[str, ...]) -> bool:
    n = len(small)
    return any(big[i:i + n] == small for i in range(len(big) - n + 1))


@dataclass
class _Occ:
    key: str                  # texte de la clé (affichage)
    ids: tuple[str, ...]
    where: str                # 'titre' | 'organisations'
    weak: bool                # nom d'un seul mot distinctif (`_weak_key`)
    common: bool              # tous ses mots distinctifs sont des mots courants (« Premier », « State Coal »)
    context: bool             # mot « mine » / substance juste à côté, ou organisation = nom exact
    seq: tuple[str, ...] = ()
    lower: bool = False       # écrit en minuscules dans un titre qui a des majuscules : pas un nom propre
    unit: bool = False        # suivi d'un numéro d'unité accolé (« Kolubara 2 ») : une autre unité


@dataclass
class _Title:
    """Titre découpé : mots repliés, mêmes mots avec leur casse d'origine, positions dans le texte replié."""
    toks: list[str]
    raw: list[str] | None = None          # None si le découpage ne s'aligne pas (on renonce aux contrôles)
    text: str = ""
    spans: list[tuple[int, int]] | None = None

    @classmethod
    def of(cls, title: str) -> _Title:
        toks = words(title)
        text = fold(title)
        spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
        keep_case = "".join(ch for ch in unicodedata.normalize("NFKD", title.translate(_APOS))
                            if not unicodedata.combining(ch))
        raw = _WORD.findall(keep_case)
        ok = len(spans) == len(toks) == len(raw)
        return cls(toks, raw if ok else None, text, spans if ok else None)


# désignateurs d'unité : « No 2 », « Nr. 2 », « N° 2 », « Unit 2 », « Shaft 3 », « Pit 4 » (+ « #2 »)
_UNIT_WORDS = _fset("""no nr n nos number num numero nomer номер unit units shaft pit schacht szyb puits pozo
poco""".split())
# mots qui suivent un NOMBRE DE PERSONNES ou une mesure, pas un numéro d'unité : « Grasberg 10 miners
# trapped », « Olympic Dam 3 workers injured », « Fekola 12 June » (principales langues de la veille)
_COUNT_NOUNS = _fset("""
workers worker miners miner men women people persons employees staff contractors dead deaths death killed
dies die died injured hurt wounded trapped missing bodies victims rescued feared evacuated
morts mort mineurs mineur ouvriers ouvrier blesses blesse tues tue travailleurs personnes disparus victimes
employes salaries muertos muerto mineros minero trabajadores heridos personas fallecidos atrapados
desaparecidos obreros victimas mortos mineiros trabalhadores feridos pessoas vitimas soterrados
tote toten bergleute arbeiter verletzte kumpel menschen vermisste
погибли погибших горняков шахтеров шахтёров человек рабочих пострадавших
years year days day hours months weeks percent per cent km tonnes tons mt kt oz koz moz million billion
january february march april may june july august september october november december janvier fevrier
avril mai juin juillet aout septembre octobre novembre decembre enero febrero marzo abril mayo junio julio
agosto septiembre octubre noviembre diciembre
""".split())


def _unit_after(t: _Title, b: int) -> bool:
    """Un numéro d'unité est-il ACCOLÉ au nom (dernier mot en position b) ? « Kolubara 2 », « Kolubara II »,
    « Kolubara #2 », « Kolubara No. 2 », « Kolubara (mine) Unit 2 », « Kolubara shaft 3 », « Kolubara-B » :
    une autre unité que la mine nommée.

    Pas un numéro : un nombre séparé du nom par une ponctuation (« Kibali gold mine: 2 dead »,
    « Fekola, 3 killed »), un nombre suivi de personnes ou d'une mesure (« Grasberg 10 miners trapped »,
    « Olympic Dam 3 workers injured »), une année (« Fekola 2026 »), un pourcentage, une décimale.
    """
    toks, n = t.toks, len(t.toks)
    if b + 1 >= n:
        return False

    def gap(i: int, j: int) -> str | None:
        """Texte entre les mots i et j (None si le découpage n'est pas aligné : on suppose une espace)."""
        return t.text[t.spans[i][1]:t.spans[j][0]] if t.spans else None

    def number(i: int, allowed: str) -> bool:
        """Mot i = numéro d'unité, séparé du mot précédent seulement par des espaces ou `allowed`."""
        g = gap(i - 1, i)
        if g is not None and g.strip(" \t" + allowed):
            return False
        w = toks[i]
        if w.isdigit() and len(w) <= 2:
            if t.spans:
                end = t.spans[i][1]
                after = t.text[end:end + 2]
                if after[:1] == "%" or (after[:1] in ".," and after[1:2].isdigit()):
                    return False
            return True
        if w in _ROMAN and len(w) >= 2:                 # « II », « IV » ; pas « i », « v », « x »
            return t.raw is None or t.raw[i].isupper()
        return False

    g = gap(b, b + 1)
    if g == "-" and len(toks[b + 1]) == 1:
        return True                                     # « Kolubara-B »
    # « Kolubara #2 » / « Kolubara 2 mine » : nombre nu juste après le nom, sans ponctuation, et qui n'est
    # pas un nombre de personnes (« Kolubara 12 workers trapped »)
    if number(b + 1, "#"):
        return not (b + 2 < n and toks[b + 2] in _COUNT_NOUNS)
    # « Kolubara No. 2 », « Kolubara mine Unit 2 » : désignateur (après au plus un mot « mine ») + numéro
    i = b + 1
    if toks[i] in MINE_WORDS and toks[i] not in _UNIT_WORDS and i + 1 < n and (gap(b, i) or " ").isspace():
        i += 1
    if toks[i] in _UNIT_WORDS and (gap(i - 1, i) or " ").isspace() and i + 1 < n:
        return number(i + 1, ".°º#:")
    return False


def _scan_words(toks: list[str], idx: NameIndex, where: str, title: _Title | None = None) -> list[_Occ]:
    pos = [(i, _ROMAN.get(t, t)) for i, t in enumerate(toks) if t not in STOP]
    vals = [v for _, v in pos]
    has_caps = bool(title and title.raw and any(w[:1].isupper() for w in title.raw))
    out, j = [], 0
    while j < len(vals):
        for n in range(min(idx.maxlen, len(vals) - j), 0, -1):
            key = tuple(vals[j:j + n])
            ids = idx.words.get(key)
            if not ids:
                continue
            a, b = pos[j][0], pos[j + n - 1][0]
            # « Kolubara No 2 » quand on cherche « Kolubara » : une autre unité, pas la mine nommée
            if not any(t.isdigit() for t in key) and _unit_after(title or _Title(toks), b):
                out.append(_Occ(" ".join(key), ids, where, True, False, False, key, unit=True))
                j += n
                break
            # nom fait de mots de lieu (« Red Lake ») : comme un mot seul courant, lieu ou requête exigés
            place = key in idx.place_keys
            weak = place or _weak_key(key)
            # « Kibali gold » : la substance du nom vaut le mot « mine » juste à côté
            window = toks[max(0, a - 3):a] + toks[b + 1:b + 4] + [t for t in key if not _is_core(t)]
            ctx = any(_is_context(t) for t in window)
            if where == "organisations":
                # l'organisation EST la mine et se dit minière (« Minera Escondida », « Mount Isa Mines ») ;
                # « Victoria Gold Corp » ou une organisation « Young » ne prouvent rien
                ctx = tuple(vals) == key and any(_is_context(t) for t in toks)
            # un nom propre prend une majuscule : « South African bank raises… » ne parle pas de « Bank »
            lower = has_caps and all(title.raw[pos[k][0]][:1].islower() for k in range(j, j + n))
            out.append(_Occ(" ".join(key), ids, where, weak, place or _all_common(key), ctx, key, lower))
            j += n
            break
        else:
            j += 1
    return out


def _scan_cjk(title: str, idx: NameIndex) -> list[_Occ]:
    if not idx.cjk or not _CJK.search(title):
        return []
    text = _cjk_fold(title)
    out, i = [], 0
    while i < len(text) - 1:
        for key, ids, weak in idx.cjk.get(text[i:i + 2], ()):
            if text.startswith(key, i):
                after = text[i + len(key):i + len(key) + 3]
                ctx = any(ch in _CJK_MINE_CHARS for ch in after)
                out.append(_Occ(key, ids, "titre", weak, False, ctx))
                i += len(key)
                break
        else:
            i += 1
    return out


def _resolve(ids: tuple[str, ...], idx: NameIndex, pts: list[tuple[float, float, str]],
             countries: set[str]) -> tuple[str | None, str, float | None]:
    """Homonymes et confirmation géographique → (site, niveau, distance km).

    Niveaux : 'lieu' (lieu GDELT à < 50 km), 'pays' (seule mine de ce nom dans un pays de l'article),
    'nom' (nom sans homonyme, article sans lieu), 'priorite', 'contredit' (article situé ailleurs),
    'ambigu'. Homonymes, comme docs/veille.md : lieu à moins de 50 km, sinon une seule mine de ce nom
    dans un pays de l'article, sinon — parmi les homonymes des pays de l'article, ou tous si l'article
    n'a pas de pays — la seule fiche qui a une production publiée, ou à défaut d'aucune, la seule qui a
    une emprise mesurée (« Escondida » : la mine de BHP, seule à publier sa production parmi quatre
    fiches, dont deux petites emprises). Une fiche sans ces données est le plus souvent une entrée
    mineure ou un doublon ; la force réduite (`STRENGTH`) dit que ce choix n'est pas une preuve. Deux
    fiches de même rang : AUCUN rattachement.
    """
    best, best_d = None, None
    for sid in ids:
        _, la, lo, _ = idx.sites.get(sid, (None, None, None, None))
        if la is None or lo is None:
            continue
        for pla, plo, _ in pts:
            d = _km(la, lo, pla, plo)
            if d < NEAR_KM and (best_d is None or d < best_d):
                best, best_d = sid, d
    if best:
        return best, "lieu", round(best_d, 1)
    same = [sid for sid in ids if idx.sites.get(sid, (None,) * 4)[3] in countries]
    if len(ids) == 1:
        if same:
            return ids[0], "pays", None
        return ids[0], ("contredit" if countries else "nom"), None
    if len(same) == 1:
        return same[0], "pays", None
    pool = same if countries else list(ids)
    if len(pool) > 1:
        for tier in (idx.producing, idx.documented):   # production publiée d'abord, puis emprise
            doc = [sid for sid in pool if sid in tier]
            if len(doc) == 1:
                return doc[0], "priorite", None
            if doc:
                break                                   # plusieurs fiches de même rang : rien ne tranche
    return None, "ambigu", None


# niveau géographique → (méthode pour un nom de plusieurs mots, pour un nom d'un mot, facteur)
_LEVELS = {
    "lieu": ("nom+lieu", "mot+lieu", 1.0), "pays": ("nom+pays", "mot+pays", 1.0),
    "nom": ("nom", "mot", 1.0), "contredit": ("contredit", None, 1.0),
    "priorite": ("nom+priorité", "mot+priorité", 1.0),
}


def _named(title: _Title, idx: NameIndex, sid: str) -> bool:
    """Le titre cite-t-il l'un des noms de la fiche `sid` (même absent de l'index de recherche : trop
    d'homonymes, fiche non étayée, double d'une autre fiche) ? Sert seulement à la requête par mine : la
    requête a déjà choisi la mine, le nom présent confirme que l'article en parle (« requête+nom »).
    Un numéro d'unité accolé (« Kolubara No 2 ») reste une autre unité."""
    keys = idx.keys_of.get(sid, ())
    if keys:
        pos = [(i, _ROMAN.get(t, t)) for i, t in enumerate(title.toks) if t not in STOP]
        vals = [v for _, v in pos]
        for key in keys:
            n = len(key)
            for j in range(len(vals) - n + 1):
                if tuple(vals[j:j + n]) == key and (any(t.isdigit() for t in key)
                                                    or not _unit_after(title, pos[j + n - 1][0])):
                    return True
    cjk = idx.cjk_of.get(sid, ())
    return bool(cjk) and any(k in _cjk_fold(" ".join(title.raw or title.toks)) for k in cjk)


def match(doc: dict, idx: NameIndex) -> tuple[list[dict], dict]:
    """Mines citées par un document : [{site_id, method, strength, evidence}] et compteurs de rejet."""
    locs = doc.get("locations") or []
    pts = [(la, lo, str(x.get("name") or "")) for x in locs if x.get("type") in CITY_TYPES
           and (la := _num(x.get("lat"))) is not None and (lo := _num(x.get("lon"))) is not None]
    countries = {c for c in (loc_country(x) for x in locs) if c}
    places = {_seq(str(x.get("name") or "").split(",")[0]) for x in locs}
    persons = [_seq(p) for p in doc.get("persons") or []]

    title = _Title.of(doc["title"])
    toks = title.toks
    # l'article parle-t-il de mines ? mot « mine »/substance dans le titre, ou thème GDELT précis
    title_mining = any(_is_context(t) for t in toks) or any(ch in _CJK_MINE_CHARS for ch in doc["title"])
    theme_mining = any(t.startswith(MINING_THEMES) for t in doc.get("themes") or ())
    occs = _scan_words(toks, idx, "titre", title) + _scan_cjk(doc["title"], idx)
    for org in doc.get("organizations") or []:
        occs += _scan_words(words(org), idx, "organisations")

    found: dict[str, dict] = {}
    stats = {"ambigus": 0, "ecartes": 0}
    raw_ids: set[str] = set()
    for o in occs:
        if o.unit:
            stats["ecartes"] += 1          # « Kolubara No 2 » : une autre unité que la mine nommée
            continue
        raw_ids.update(o.ids)
        if o.seq and any(_contains(p, o.seq) for p in persons):
            stats["ecartes"] += 1          # « Felix » dans « Felix Tshisekedi » : une personne
            continue
        if o.lower:
            stats["ecartes"] += 1          # « bank », « state » en minuscules : un mot, pas un nom
            continue
        sid, level, dist = _resolve(o.ids, idx, pts, countries)
        if sid is None:
            stats["ambigus"] += 1
            continue
        if o.seq and o.seq in places and level != "lieu":
            stats["ecartes"] += 1          # le nom est celui d'un lieu de l'article, loin de la mine
            continue
        multi, single, factor = _LEVELS[level]
        if o.common and level != "lieu":
            stats["ecartes"] += 1          # mots courants (« Hope », « State Coal ») : lieu exigé
            continue
        if o.weak:
            # nom d'un seul mot : il faut un titre qui parle de mines (ou une organisation qui EST la mine),
            # jamais contre la géographie ; sans lieu ni pays, un mot « mine » juste à côté. Sans lieu à
            # moins de 50 km, sa force (0,4–0,45) le laisse sous le seuil 0,5 du corrélateur.
            exact_org = o.where == "organisations" and o.context
            if not (title_mining or exact_org) or (o.where == "organisations" and not exact_org) \
                    or single is None or (level in ("nom", "priorite") and not o.context):
                stats["ecartes"] += 1
                continue
            method = single
        else:
            method = multi
            if not (title_mining or theme_mining):
                if sid not in idx.notable and level != "lieu":
                    stats["ecartes"] += 1      # « Black Friday Mines » dans une page de soldes
                    continue
                factor *= 0.8                  # nom présent, mais l'article ne parle pas de mines
        s = STRENGTH[method] * factor * (ORG_FACTOR if o.where == "organisations" else 1.0)
        ev = {"nom": o.key, "trouve_dans": o.where, "homonymes": len(o.ids), "niveau": level}
        if dist is not None:
            ev["distance_km"] = dist
        if s > found.get(sid, {}).get("strength", -1):
            found[sid] = {"site_id": sid, "method": method, "strength": round(s, 3), "evidence": ev}

    # requêtes par mine : celle du document, plus celles héritées de ses doublons (`extra_hints`)
    seen: set[str] = set()
    for raw_hint in [doc.get("hint_site"), *(doc.get("extra_hints") or ())]:
        hint = idx.canonical(raw_hint)
        if not hint or hint in seen:
            continue
        seen.add(hint)
        cur = found.get(hint)
        rep = idx.same_as.get(hint)        # requête d'une fiche en double : le nom indexé est celui du double
        if cur is None and rep is not None and rep in raw_ids:
            cur = found.pop(rep, None)
        if cur or hint in raw_ids or (rep is not None and rep in raw_ids) or _named(title, idx, hint):
            ev = (cur or {}).get("evidence") or {"nom": "présent (homonymes tranchés par la requête)"}
            found[hint] = {"site_id": hint, "method": "requête+nom",
                           "strength": max(STRENGTH["requête+nom"], (cur or {}).get("strength", 0)),
                           "evidence": {**ev, "requete": raw_hint}}
        else:
            found[hint] = {"site_id": hint, "method": "requête", "strength": STRENGTH["requête"],
                           "evidence": {"nom": None, "requete": raw_hint}}
    out = sorted(found.values(), key=lambda m: (-m["strength"], m["site_id"]))[:MAX_SITES_PER_DOC]
    if countries:
        for m in out:
            m["evidence"]["pays_article"] = sorted(countries)[:8]
    return out, stats


# == étape complète =====================================================================================
def _event_id(doc_id: str, site_id: str) -> str:
    return hashlib.sha1(f"{doc_id}|{site_id}".encode()).hexdigest()


def _recent_done(con: sqlite3.Connection, docs: list[dict]) -> list[sqlite3.Row]:
    """Documents déjà traités des mêmes domaines autour des dates du lot (fenêtre de dédoublonnage)."""
    if not docs:
        return []
    lo = (min(d["published"] for d in docs) - DEDUP_WINDOW).strftime("%Y-%m-%d")
    hi = (max(d["published"] for d in docs) + DEDUP_WINDOW + timedelta(days=1)).strftime("%Y-%m-%d")
    domains = sorted({d["domain"] for d in docs if d["domain"]})
    rows: list = []
    for i in range(0, len(domains), 400):
        chunk = domains[i:i + 400]
        variants = chunk + ["www." + d for d in chunk]
        q = ",".join("?" * len(variants))
        rows += con.execute(
            f"SELECT id, url, domain, title, published_at FROM raw_docs WHERE status='done' "
            f"AND published_at >= ? AND published_at < ? AND domain IN ({q})", (lo, hi, *variants)).fetchall()
    return rows


def _event_rows(doc: dict, matches: list[dict], idx: NameIndex, created: str) -> list[tuple]:
    """Lignes `events` d'un document (une par mine), avec la preuve en JSON."""
    typ, cstrength, cdetail = classify(doc["title"], doc["themes"], doc["lang"])
    rows = []
    for m in matches:
        name, la, lo, country = idx.sites.get(m["site_id"], (None, None, None, None))
        conf = round(m["strength"] * cstrength, 3)
        ev = {**m["evidence"], "methode": m["method"], "force_rattachement": m["strength"],
              "type": {"retenu": typ, **cdetail}, "force_classement": cstrength}
        rows.append((
            _event_id(doc["id"], m["site_id"]), doc["id"], m["site_id"], name, typ, doc["published_at"],
            doc["published_at"][:7], doc["title"], doc["url"], doc["domain"] or None, doc["lang"],
            country, la, lo, conf, m["method"], json.dumps(ev, ensure_ascii=False), created))
    return rows


def flag_doc_names(con: sqlite3.Connection, idx: NameIndex) -> tuple[int, int]:
    """Pour chaque mine suivie dont l'historique documentaire est fait : l'un de ses noms de requête
    (`collect.mine_names`, ceux que l'API a cherchés) est-il reconnu dans un titre ? Écrit
    meta « doc_reconnu:<mine> » = '1' ou '0' ; rend (reconnues, non reconnues).

    Le corrélateur ne compte la couverture de l'historique documentaire que pour les mines reconnues :
    sinon des années d'articles renvoyés par la requête, jamais devenus des événements faute de nom
    reconnu, compteraient comme des mois « sans événement ». Recalculé à chaque passage (l'index change
    avec le catalogue et names.parquet)."""
    rows = con.execute("SELECT site_id, site_name FROM tracked WHERE backfill_status = 'fait'").fetchall()
    flags = []
    for sid, name in rows:
        target = idx.canonical(sid)
        ok = any(_named(_Title.of(n), idx, target) for n in collect.mine_names(sid, name))
        flags.append((f"doc_reconnu:{sid}", "1" if ok else "0"))
    con.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", flags)
    n_ok = sum(v == "1" for _, v in flags)
    return n_ok, len(flags) - n_ok


_INSERT_EVENT = ("INSERT OR REPLACE INTO events(id, doc_id, site_id, site_name, event_type, published_at, "
                 "month, title, url, domain, lang, country, lat, lon, confidence, match_method, evidence, "
                 "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")


def _late_hints(con: sqlite3.Connection, late: dict[str, list[str]], idx: NameIndex, now: datetime,
                created: str) -> list[tuple]:
    """Requêtes par mine portées par des doublons d'un document DÉJÀ traité (status 'done') : on relit
    l'original et on n'écrit que les événements des mines demandées (même id d'événement : idempotent).
    Sans cela, « https://www.x.com/a » (flux GKG, traité hier) puis « https://x.com/a/ » (requête sur une
    mine suivie) ferait perdre à la mine suivie son événement."""
    rows: list[tuple] = []
    for orig_id, hints in late.items():
        r = con.execute("SELECT * FROM raw_docs WHERE id=? AND status='done'", (orig_id,)).fetchone()
        if r is None:
            continue
        try:
            doc, _ = validate(r, now)
            if doc is None:
                continue
            doc["extra_hints"] = hints
            wanted = {idx.canonical(h) for h in hints}
            matches = [m for m in match(doc, idx)[0] if m["site_id"] in wanted]
            rows += _event_rows(doc, matches, idx, created)
        except Exception:  # un original abîmé ne doit pas bloquer le lot
            log.exception("requête héritée non appliquée à %s", orig_id)
    return rows


def run_once(con: sqlite3.Connection, batch: int = 5000, *, index: NameIndex | None = None,
             now: datetime | None = None) -> dict:
    """Traite au plus `batch` documents `new` : formats, doublons, mines, type → events + statuts.

    Chaque document est isolé : une erreur imprévue en le lisant le classe « rejected » (motif
    « erreur de traitement ») au lieu de bloquer le lot — sinon, resté « new » et trié en tête, il ferait
    échouer tous les cycles suivants.
    """
    con.row_factory = sqlite3.Row
    now = now or datetime.now(UTC)
    counters = {"lus": 0, "rejetes": 0, "doublons": 0, "sans_mine": 0, "rattaches": 0, "evenements": 0,
                "requete_seule": 0, "ambigus": 0, "ecartes": 0, "erreurs": 0, "requetes_heritees": 0,
                "motifs": {}}
    try:
        idx = index or get_index()
    except Exception as e:  # catalogue illisible : on ne touche à rien, on réessaie au prochain cycle
        log.exception("index des noms indisponible")
        store.set_status(con, STAGE, False, f"index des noms indisponible : {e}", **counters)
        con.commit()
        return counters
    if not idx.catalog_rows:
        # sans catalogue (clone neuf, catalogue pas encore construit), seule la liste suivie serait
        # cherchée : presque tout finirait « sans mine », marqué 'done' et jamais relu. On attend.
        store.set_status(con, STAGE, False, "catalogue des mines absent : documents laissés en attente",
                         **counters)
        con.commit()
        return counters

    rows = con.execute("SELECT * FROM raw_docs WHERE status='new' ORDER BY published_at, id LIMIT ?",
                       (batch,)).fetchall()
    counters["lus"] = len(rows)
    updates: list[tuple[str, str | None, str]] = []
    valid: list[dict] = []

    def reject(doc_id: str, why: str) -> None:
        updates.append(("rejected", why, doc_id))
        counters["rejetes"] += 1
        counters["motifs"][why] = counters["motifs"].get(why, 0) + 1

    def failed(doc_id: str, e: Exception) -> None:
        log.warning("document %s : erreur de traitement (%s: %s)", doc_id, type(e).__name__, e)
        counters["erreurs"] += 1
        reject(doc_id, f"erreur de traitement ({type(e).__name__})")

    for r in rows:
        try:
            doc, why = validate(r, now)
        except Exception as e:
            failed(r["id"], e)
            continue
        if doc is None:
            reject(r["id"], why)
        else:
            valid.append(doc)

    # -- doublons : même URL (après nettoyage), ou même titre + même domaine à moins de 3 jours ----------
    # valeur : (id de l'original, document s'il est dans ce lot, None s'il est déjà traité)
    by_url: dict[str, tuple[str, dict | None]] = {}
    by_title: dict[tuple[str, str], list[tuple[datetime, dict | None, str]]] = {}
    for r in _recent_done(con, valid):
        try:
            d = parse_date(r["published_at"])
            by_url.setdefault(canon_url(r["url"]), (r["id"], None))
            k = title_key(html.unescape(r["title"] or ""))
        except Exception:  # ligne ancienne abîmée : elle ne sert simplement pas de référence
            continue
        if k and d:
            by_title.setdefault((_domain(r["url"], r["domain"]), k), []).append((d, None, r["id"]))
    kept: list[dict] = []
    late: dict[str, list[str]] = {}         # original déjà traité → requêtes par mine de ses doublons
    dups_of: dict[str, list[dict]] = {}     # original du lot → ses doublons du lot (relève en cas d'erreur)
    for doc in valid:
        try:
            cu = canon_url(doc["url"])
            tk = title_key(doc["title"])
        except Exception as e:
            failed(doc["id"], e)
            continue
        orig, orig_doc = by_url.get(cu, (None, None))
        if orig is None and tk:
            for d, od, oid in by_title.get((doc["domain"], tk), ()):
                if abs(d - doc["published"]) <= DEDUP_WINDOW:
                    orig, orig_doc = oid, od
                    break
        if orig is not None and orig != doc["id"]:
            updates.append(("rejected", f"doublon de {orig}", doc["id"]))
            counters["doublons"] += 1
            if orig_doc is not None:
                dups_of.setdefault(orig, []).append(doc)
            # la requête par mine apporte une info que le doublon ne doit pas perdre, que l'original soit
            # dans ce lot (on la lui ajoute) ou déjà traité (on le relit après le lot)
            if doc["hint_site"]:
                if orig_doc is not None:
                    orig_doc.setdefault("extra_hints", []).append(doc["hint_site"])
                else:
                    late.setdefault(orig, []).append(doc["hint_site"])
            continue
        by_url[cu] = (doc["id"], doc)
        if tk:
            by_title.setdefault((doc["domain"], tk), []).append((doc["published"], doc, doc["id"]))
        kept.append(doc)
    # -- rattachement + classement → events ---------------------------------------------------------
    created = store.now()
    events: list[tuple] = []
    sites_hit: set[str] = set()
    def process(doc: dict) -> bool:
        try:
            matches, st = match(doc, idx)
            rows_ev = _event_rows(doc, matches, idx, created) if matches else []
        except Exception as e:
            failed(doc["id"], e)
            return False
        counters["ambigus"] += st["ambigus"]
        counters["ecartes"] += st["ecartes"]
        updates.append(("done", None, doc["id"]))
        if not matches:
            counters["sans_mine"] += 1
            return True
        counters["rattaches"] += 1
        if all(m["method"] == "requête" for m in matches):
            counters["requete_seule"] += 1
        events.extend(rows_ev)
        sites_hit.update(m["site_id"] for m in matches)
        return True

    for doc in kept:
        if process(doc):
            continue
        # original en erreur : un doublon du lot prend sa place, avec toutes les requêtes par mine du
        # groupe (sinon ni l'original ni ses doublons ne donneraient d'événement)
        hints = [h for h in [doc.get("hint_site"), *(doc.get("extra_hints") or ())] if h]
        for dup in dups_of.get(doc["id"], ()):
            counters["doublons"] -= 1         # plus un doublon : traité, ou en erreur à son tour
            dup["extra_hints"] = [h for h in hints if h != dup["hint_site"]]
            if process(dup):
                break
    if late:
        extra = _late_hints(con, late, idx, now, created)
        counters["requetes_heritees"] = len(extra)
        events += extra
        sites_hit.update(e[2] for e in extra)
    con.executemany(_INSERT_EVENT, events)
    con.executemany("UPDATE raw_docs SET status=?, reject_reason=? WHERE id=?", updates)
    counters["evenements"] = len(events)
    counters["mines"] = len(sites_hit)
    try:
        counters["historiques_reconnus"], counters["historiques_non_reconnus"] = flag_doc_names(con, idx)
    except Exception:  # un catalogue illisible ne doit pas faire perdre le lot : drapeaux au prochain cycle
        log.exception("noms de requête non vérifiés")
    msg = (f"{counters['lus']} documents : {len(events)} événements sur {len(sites_hit)} mines, "
           f"{counters['rejetes']} rejetés, {counters['doublons']} doublons, "
           f"{counters['sans_mine']} sans mine")
    if counters["erreurs"]:
        msg += f", {counters['erreurs']} en erreur"
    store.set_status(con, STAGE, True, msg, **counters)
    con.commit()
    return counters
