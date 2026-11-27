"""Onglet « Veille » de l'interface : contrôles statiques (sans navigateur ni réseau).

La page ne doit lire que le contrat d'API de docs/veille.md : les routes et les champs utilisés sont comparés
au tableau du document, pour qu'un renommage côté serveur casse ce test plutôt que l'affichage.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from strate.veille.store import EVENT_TYPES

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src" / "strate" / "web" / "index.html"
DOC = ROOT / "docs" / "veille.md"


def page() -> str:
    return HTML.read_text("utf-8")


def script() -> str:
    return re.search(r"<script>\n(.*?)</script>", page(), re.S).group(1)


def veille_js() -> str:
    """Région du script consacrée à la veille (bornée par des commentaires repères)."""
    return re.search(r"// == veille =+\n(.*?)// == fin veille", script(), re.S).group(1)


def doc() -> str:
    return DOC.read_text("utf-8")


def doc_fields(pattern: str) -> set[str]:
    m = re.search(pattern, doc())
    assert m, pattern
    return {f.strip() for f in m.group(1).split(",")}


def used_fields(var: str) -> set[str]:
    return set(re.findall(rf"\b{var}\??\.(\w+)", veille_js()))


# -- structure ---------------------------------------------------------------------------------------------
def test_third_tab_and_view():
    p = page()
    assert re.search(r'<nav class="tabs">.*data-v="map".*data-v="screener".*data-v="veille".*</nav>', p)
    assert '<div id="view-veille">' in p
    assert "$('#view-veille').style.display=v==='veille'?'block':'none'" in script()


def test_filters_present():
    for fid in ("vf_type", "vf_lang", "vf_country", "vf_conf", "vf_days"):
        assert f'id="{fid}"' in page(), fid


def test_empty_message():
    # base vide : le message exact demandé, dans le fil, les graphiques, les corrélations et la liste suivie
    js = veille_js()
    assert "const EMPTY='La veille n’a encore rien collecté'" in js
    assert js.count("VEI.empty?EMPTY") >= 3


# -- contrat d'API -----------------------------------------------------------------------------------------
def test_only_contract_routes():
    contract = set(re.findall(r"`GET (/api/veille/\w+)", doc()))
    used = set(re.findall(r"/api/veille/\w+", script()))
    assert contract and used, (contract, used)
    assert used <= contract, used - contract
    # toutes les routes du contrat servent à l'interface
    assert contract <= used, contract - used


def test_event_fields_from_contract():
    allowed = doc_fields(r'"events":\[\{([^}]*)\}')
    for var in ("it", "e2"):
        assert used_fields(var) <= allowed, (var, used_fields(var) - allowed)


def test_correlation_and_tracked_fields_from_contract():
    rows = doc_fields(r'"rows":\[\{([^}]*)\}')
    rows |= {"verdict_rho"}  # contrat partagé : verdict calculé sur l'IC de φ (rho des lignes de type)
    assert used_fields("cr") <= rows, used_fields("cr") - rows
    tracked = doc_fields(r"/api/veille/tracked` \| `\[\{([^}]*)\}")
    assert used_fields("tk") <= tracked, used_fields("tk") - tracked


def test_status_and_stats_keys_from_contract():
    js = veille_js()
    for key in ("stages", "totaux", "service", "actif", "dernier_cycle", "counters", "last_run"):
        assert key in doc() and key in js, key
    for key in ("par_jour", "par_langue", "par_pays", "top_mines"):
        assert key in doc() and f"st.{key}" in js, key
    for key, _ in re.findall(r"\['(\w+)','([^']+)'\]", re.search(r"const TOT=\[(.*?)\];", js, re.S).group(1)):
        assert key in doc(), key


# -- sécurité ----------------------------------------------------------------------------------------------
def test_links_open_in_new_tab_with_noopener():
    blanks = re.findall(r'target="_blank"[^>]*', page())
    assert blanks and all('rel="noopener' in b for b in blanks), blanks


def test_no_raw_interpolation_of_server_fields():
    # tout champ serveur interpolé passe par esc() (ou fmt/num pour un nombre) : jamais ${it.title} brut
    raw = re.findall(r"\$\{(?:it|e2|tk|cr|x|svc|c)\.\w+\}", veille_js())
    assert not raw, raw


def test_urls_filtered_before_href():
    js = veille_js()
    assert "const safeUrl=u=>/^https?:\\/\\//i" in js
    assert 'href="${esc(it.url)}"' not in js and 'href="${esc(e2.url)}"' not in js


def test_type_colour_never_from_arbitrary_key():
    assert "Object.hasOwn(VT_COL,String(t))" in veille_js()


# -- comportement attendu ----------------------------------------------------------------------------------
def test_every_event_type_has_a_colour():
    body = re.search(r"const VT_COL=\{(.*?)\};", veille_js(), re.S).group(1)
    cols = dict(re.findall(r"(\w+):'(#[0-9a-f]{6})'", body))
    assert set(EVENT_TYPES) <= set(cols), set(EVENT_TYPES) - set(cols)
    assert len(set(cols.values())) == len(cols)  # une couleur par type


def test_refresh_every_60s_only_while_visible():
    js = veille_js()
    assert "setTimeout(vRefresh,60000)" in js
    assert "setInterval" not in js  # aucun minuteur permanent (les tests Node comptent les intervalles)
    assert "if(vTab()||VEI.layerOn)" in js


def test_correlation_verdict_requires_q_below_010():
    js = veille_js()
    demo = "cr.verdict==='lien démontré'&&num(cr.q_value)!=null&&cr.q_value<0.10&&vLag(cr)===0&&vOut(cr)"
    assert demo in js
    note = re.search(r'<p class="meta vnote">(.*?)</p>', page(), re.S).group(1)
    for w in ("exploration", "Benjamini-Hochberg", "q &lt; 0,10", "90 %", "pas une causalité"):
        assert w in note, w


def test_method_note_describes_cmh_and_lag0_family():
    # décision : Cochran–Mantel–Haenszel par mine, rapport de risques MH, famille principale = décalage 0,
    # décalages ±1..±3 exploratoires, couverture limitée aux mois collectés ; plus aucune permutation
    note = re.sub(r"\s+", " ", re.search(r'<p class="meta vnote">(.*?)</p>', page(), re.S).group(1))
    for w in ("Cochran–Mantel–Haenszel", "Mantel–Haenszel", "décalage 0", "exploratoire", "bootstrap",
              "continuité 0,5", "réellement collectée", "n/d", "φ"):
        assert w in note, w
    # la correction de continuité : même règle que correlate.METHODE (association parfaite), pas « case vide »
    assert "quand une case est vide" not in note
    assert "aucune anomalie ne tombe hors des mois d’événement" in note
    from strate.veille.correlate import METHODE
    assert "continuité" in METHODE and "hors des mois d'événement" in METHODE
    assert "permutation" not in note.lower() and "permutation" not in veille_js().lower()
    assert "c.methode" in veille_js()  # le texte de correlate.METHODE, servi par l'API, est affiché


# -- comportements rejoués sous Node (région veille seule, DOM et bibliothèques factices) -----------------
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node absent : seuls les contrôles statiques tournent")

STUBS = r"""
const els={}, made=[];
function mk(sel){ return {sel,id:'',style:{},dataset:{},innerHTML:'',textContent:'',value:'',checked:false,
  hidden:false,isConnected:true,className:'',classList:{toggle(){}},addEventListener(){},
  querySelectorAll:()=>[]}; }
const $=s=>els[s]??=mk(s);
class Chart{constructor(el,cfg){this.el=el;this.cfg=cfg;made.push(this);} destroy(){} update(){}}
const L={layerGroup:()=>({addTo(){},clearLayers(){}}),divIcon:()=>({}),
  marker:()=>({bindPopup(){return this;},addTo(){}})};
const map={hasLayer:()=>false,removeLayer(){},on(){}}, charts=[];
let apiImpl=async()=>{throw new Error('pas de réseau');};  // remplaçable par un test (réponses factices)
const api=(...a)=>apiImpl(...a);
const errTxt=e=>String(e), open=()=>{}, show=()=>{};
const out=v=>console.log(JSON.stringify(v));
"""


def run_js(body: str, tmp_path: Path) -> list:
    """Exécute la région veille (avec esc/fmt/num/LBL de la page) puis `body` ; rend les out(...) émis."""
    js = script()
    names = ("LBL", "esc", "fmt", "num")
    helpers = "\n".join(re.search(rf"^const {n}=.*$", js, re.M).group(0) for n in names)
    f = tmp_path / "veille.js"
    f.write_text(STUBS + helpers + "\n" + veille_js() + "\n;(async()=>{\n" + body + "\n})();", "utf-8")
    r = subprocess.run([NODE, str(f)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return [json.loads(line) for line in r.stdout.splitlines() if line.strip()]


@needs_node
def test_all_period_chart_shows_recent_days(tmp_path):
    # « tout » : /api/veille/stats?days=3650 rend 3651 jours, presque tous à 0 ; les événements récents
    # doivent rester sur l'axe (avant : l'axe s'arrêtait au 400e jour compté depuis 2016, graphique vide)
    (res,) = run_js(r"""
      const d0=Date.now()-3650*864e5, J=i=>new Date(d0+i*864e5).toISOString().slice(0,10);
      const par_jour=[...Array(3651).keys()].map(i=>({jour:J(i),n:i>=3648?2:0}));
      const today=J(3650), evs=[{published_at:today+'T08:00:00Z',event_type:'accident'},
        {published_at:today+'T09:00:00Z',event_type:'social'}];
      vStats({par_jour,par_langue:{},par_pays:{},top_mines:[]},{events:evs},'');
      const c=made.at(-1), ds=c.cfg.data.datasets, lab=c.cfg.data.labels, k=lab.indexOf(today);
      out({unit:$('#v_day_h').textContent,first:lab[0],last:lab.at(-1),today,n:lab.length,
        k,total:ds.reduce((a,d)=>a+d.data[k],0),acc:ds.find(d=>d.label==='accident').data[k]});
    """, tmp_path)
    assert res["unit"] == "Événements par jour, par type"
    assert res["k"] >= 0 and res["last"] == res["today"] and res["n"] == 3
    assert res["total"] == 2 and res["acc"] == 1  # 2 détaillés par type, total exact (par_jour)


@needs_node
def test_long_history_switches_to_monthly_bars(tmp_path):
    (res,) = run_js(r"""
      const tot={'2019-03-04':3,'2019-03-20':1,'2024-11-02':5}, cnt={'2024-11-02':{finance:2}};
      const S=vSeries(tot,cnt);
      out({unit:S.unit,first:S.labels[0],last:S.labels.at(-1),n:S.labels.length,mar:S.tot['2019-03'],
        nov:S.tot['2024-11'],fin:S.cnt['2024-11'].finance,empty:vSeries({'2020-01-01':0},{}).labels.length});
    """, tmp_path)
    assert res["unit"] == "mois" and res["first"] == "2019-03" and res["last"] == "2024-11"
    assert res["n"] == 69  # mois contigus, y compris les mois vides
    assert (res["mar"], res["nov"], res["fin"]) == (4, 5, 2)
    assert res["empty"] == 0  # aucun jour non vide : pas d'axe, message « Rien sur la période. »


@needs_node
def test_unknown_language_or_country_never_offered(tmp_path):
    (res,) = run_js(r"""
      vOpts('#vf_country',{Chile:4,inconnu:9,Peru:1},'tous');
      out({keys:vOptKeys({en:3,inconnu:7,'':1,fr:5}),html:$('#vf_country').innerHTML});
    """, tmp_path)
    assert res["keys"] == ["fr", "en"]
    assert "inconnu" not in res["html"] and "Chile" in res["html"]


@needs_node
def test_heatmap_cells_follow_verdict_rules(tmp_path):
    (res,) = run_js(r"""
      const base={event_type:'accident',signal:'bare_km2',n_mines:8,n_events:30,p_value:0.001,q_value:0.01};
      const cell=cr=>vCell({...base,...cr},0);
      out({
        demo:cell({lag_months:0,lift:3,lift_lo:1.5,lift_hi:6,verdict:'lien démontré'}),
        ci1:cell({lag_months:0,lift:3,lift_lo:0.8,lift_hi:6,verdict:'lien démontré'}),
        xp:cell({lag_months:2,lift:3,lift_lo:1.5,lift_hi:6,verdict:'lien démontré'}),
        nd:cell({lag_months:0,lift:null,verdict:'non démontré'}),
        few:cell({lag_months:0,lift:2,verdict:'trop peu de données'}),
        fewxp:cell({lag_months:3,lift:2,lift_lo:1.2,lift_hi:3,n_mines:3,verdict:'exploratoire'}),
        vxp:vVerdict({...base,lag_months:-1,lift:2,lift_lo:1.2,lift_hi:3,verdict:'exploratoire'}),
      });
    """, tmp_path)
    assert "★" in res["demo"] and 'class="demo"' in res["demo"]
    assert "★" not in res["ci1"]  # IC contenant 1 : jamais démontré
    assert "★" not in res["xp"] and 'class="xp"' in res["xp"]  # décalage ≠ 0 : exploratoire
    assert 'class="nd"' in res["nd"] and "n/d" in res["nd"] and "few" not in res["nd"]
    assert 'class="few"' in res["few"]
    assert 'class="few"' in res["fewxp"]  # sous 5 mines ou 10 mois : hachuré, quel que soit le verdict
    assert res["vxp"] == "exploratoire"


@needs_node
def test_event_rows_show_phi_badge(tmp_path):
    # contrat : une ligne de type porte aussi rho = φ (mois d'événement × mois d'anomalie) et son IC ;
    # badge φ à part, verdict_rho tel quel, jamais pris pour une ligne « volume » même quand le lift est nul
    (res,) = run_js(r"""
      const base={event_type:'accident',signal:'bare_km2',n_mines:8,n_events:30,p_value:0.2,q_value:0.4};
      const a={...base,lag_months:0,lift:1.8,lift_lo:1.1,lift_hi:2.9,rho:0.12,rho_lo:0.03,rho_hi:0.2,
        verdict:'non démontré',verdict_rho:'non démontré (q ≥ 0,10)'};
      const b={...base,lag_months:2,lift:1.8,lift_lo:1.1,lift_hi:2.9,rho:0.12,rho_lo:0.03,rho_hi:0.2,
        verdict:'exploratoire'};
      const c={...base,lag_months:0,lift:null,rho:0.05,rho_lo:-0.02,rho_hi:0.1,verdict:'non démontré'};
      const old={...base,lag_months:0,lift:2,lift_lo:1.2,lift_hi:3,verdict:'non démontré'};
      out({a:vBadge(a)+'|'+vPhi(a),b:vPhi(b),c:vBadge(c)+'|'+vPhi(c),cvol:vVol(c),old:vPhi(old),
        vol:vPhi({...base,event_type:'volume',rho:0.3})});
    """, tmp_path)
    assert "lift ×" in res["a"] and "φ" in res["a"] and "non démontré (q ≥ 0,10)" in res["a"]
    assert "IC 90 % +0,03 à +0,20" in res["a"] and "n = 8 mines (bootstrap)" in res["a"]
    assert "p-acc" not in res["a"] and "lien positif" not in res["a"]
    assert ">exploratoire<" in res["b"] and "lien positif" not in res["b"]
    assert res["cvol"] is False and "lift" in res["c"] and "φ" in res["c"]
    assert res["old"] == "" and res["vol"] == ""  # serveur ancien sans φ ; volume : ρ déjà dans vBadge


@needs_node
def test_stage_late_uses_service_margin(tmp_path):
    (res,) = run_js(r"""
      const now=Date.now(), m=60e3;
      out([vLate(now-50*m,900,now),vLate(now-61*m,900,now),vLate(now-80*m,1800,now),vLate(now-91*m,1800,now),
        vLate(now-61*m,undefined,now),vLate(null,900,now)]);
    """, tmp_path)
    assert res == [False, True, False, True, True, False]


@needs_node
def test_site_summary_key_from_api(tmp_path):
    (res,) = run_js(r"""
      const none=veilleOf({veille:{events_30j:0,dernier:null,events_count:0}});
      const some=veilleOf({veille:{events_30j:2,dernier:'2026-11-20T10:00:00Z',events_count:5}});
      out({none,some,absent:veilleOf({}),nullv:veilleOf({veille:null}),
        noneHtml:vSiteBlock(none),someHtml:vSiteBlock(some)});
    """, tmp_path)
    assert res["none"] == {"events": []}  # aucun article : pas de requête /api/veille/site
    assert res["some"]["fetch"] is True and res["some"]["n30"] == 2
    assert res["absent"] is None and res["nullv"] is None
    assert "Aucun article rattaché" in res["noneHtml"]
    assert "Chargement de l’actualité" in res["someHtml"] and "2 en 30 jours" in res["someHtml"]


def test_map_layer_pulses_recent_events():
    p = page()
    assert ".vev.new::after" in p and "@keyframes ping" in p
    assert "prefers-reduced-motion" in p
    assert 'id="v_layer"' in p


def test_mine_sheet_has_news_and_markers():
    js = script()
    assert "vSiteBlock(vs)" in js and "<h2>Actualité (veille)</h2>" in js
    assert "plugins:[vMarkPlugin]" in js and "ch.$vmarks=marks||[]" in js
    # le contrôle d'événements connus de /api/events reste distinct de la veille
    assert "api('/api/events?site='" in js


def test_external_scripts_pinned_on_cdnjs():
    srcs = re.findall(r'<script src="([^"]+)"', page())
    pinned = r"https://cdnjs\.cloudflare\.com/ajax/libs/[\w.-]+/\d+\.\d+\.\d+/"
    assert srcs and all(re.match(pinned, s) for s in srcs)
    assert sum("Chart.js" in s for s in srcs) == 1


@needs_node
def test_load_requests_follow_period_and_confidence(tmp_path):
    # « tout » : stats sur 3650 jours ; stats et découpe par type suivent la confiance minimale du fil
    (urls,) = run_js(r"""
      const seen=[]; apiImpl=async u=>{ seen.push(u); return {}; };
      $('#vf_days').value='0'; $('#vf_conf').value='0.3';
      await vLoad(); out(seen);
    """, tmp_path)
    q = {u.split("?")[0] + "|" + u: dict(x.split("=", 1) for x in u.split("?", 1)[1].split("&"))
         for u in urls if "?" in u}
    stats = [v for k, v in q.items() if k.startswith("/api/veille/stats|")]
    events = [v for k, v in q.items() if k.startswith("/api/veille/events|")]
    assert stats == [{"days": "3650", "min_conf": "0.3"}]
    assert len(events) == 2 and all(e["min_conf"] == "0.3" and "since" not in e for e in events)
    assert {e["limit"] for e in events} == {"200", "1000"}
    assert "/api/veille/status" in urls and "/api/veille/types" in urls


@needs_node
def test_correlation_header_never_says_lags_untested(tmp_path):
    (html,) = run_js(r"""
      const r=l=>({event_type:'accident',signal:'bare_km2',lag_months:l,n_mines:8,n_events:30,lift:1.4,
        lift_lo:0.9,lift_hi:2,p_value:0.2,q_value:0.4,verdict:l?'exploratoire':'non démontré'});
      vCorr({run_at:'2026-11-27T10:00:00Z',methode:'CMH <stratifié>',rows:[r(-1),r(0),r(1)]},'');
      out($('#v_corrmeta').innerHTML+$('#v_corr').innerHTML);
    """, tmp_path)
    assert "seule testée" not in html and "seul décalage testé" not in html
    assert "aucun lien démontré" in html and "CMH &lt;stratifié&gt;" in html
    assert "★ 0 m" in html and "exploratoire" in html
