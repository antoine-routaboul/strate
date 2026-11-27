"""Interface v2 (src/strate/web/index.html) : contrôles statiques et comportements rejoués sous Node.

Barre du haut et menus, routage par hash, thèmes clair et sombre (jetons, contraste), aucune décoration
« générée » (dégradé, lueur, emoji), pages Sociétés / Prévisions / Méthode limitées au contrat de
docs/previsions-societes.md § 3, badge de corrélation (coefficient, IC 90 %, n, verdict) partout où une
relation est montrée, échappement des chaînes du serveur. Sans Node, seuls les contrôles statiques tournent.
"""

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src" / "strate" / "web" / "index.html"
DOC = ROOT / "docs" / "previsions-societes.md"
API = ROOT / "src" / "strate" / "api.py"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node absent : seuls les contrôles statiques tournent")


def page() -> str:
    return HTML.read_text("utf-8")


def script() -> str:
    return re.search(r"<script>\n(.*?)</script>", page(), re.S).group(1)


def css() -> str:
    return re.search(r"<style>(.*?)</style>", page(), re.S).group(1)


def pages_js() -> str:
    """Région des pages Sociétés, Prévisions et Méthode."""
    return re.search(r"// == sociétés, prévisions, méthode =+\n(.*?)// == thème", script(), re.S).group(1)


def section3() -> str:
    return re.search(r"## 3\. API.*?(?=\n## |\Z)", DOC.read_text("utf-8"), re.S).group(0)


# -- barre du haut, menus, pages -------------------------------------------------------------------------
MENUS = {
    "Explorer": [("Carte", "#/carte"), ("Screener", "#/screener")],
    "Veille": [("Flux d’actualité", "#/veille/flux"),
               ("Corrélations événements ↔ satellite", "#/veille/correlations"),
               ("Mines suivies", "#/veille/suivies"), ("État du service", "#/veille/etat")],
    "Sociétés": [("Annuaire", "#/societes"), ("CAPEX & investissements", "#/societes/capex"),
                 ("Effectifs", "#/societes/effectifs")],
    "Prévisions": [("Prévisions par mine", "#/previsions"),
                   ("Rétro-test : l’outil s’évalue", "#/previsions/retro-test")],
    "Méthode": [("Calibration", "#/methode/calibration"), ("Sources & licences", "#/methode/sources"),
                ("Corrélations — toutes", "#/methode/correlations")],
}


def test_top_bar_names_product_and_has_global_search():
    p = page()
    assert "<title>Strate</title>" in p
    top = re.search(r'<header class="top">(.*?)</header>', p, re.S).group(1)
    assert re.search(r'class="brand"[^>]*>.*<span>Strate</span>', top, re.S)
    search = re.search(r'<input id="cmd"[^>]*>', top).group(0)
    assert 'aria-label="Recherche globale"' in search
    js = script()
    assert "api('/api/search?q='" in js and "api('/api/companies?'" in js  # mines et sociétés


def test_menus_group_pages_and_are_keyboard_accessible():
    nav = re.search(r'<nav class="tabs">(.*?)</nav>', page(), re.S).group(1)
    pat = r'<button class="menu-btn"[^>]*aria-expanded="false" aria-controls="([\w-]+)">([^<]+)'
    btns = re.findall(pat, nav)
    assert [html.unescape(b[1]).strip() for b in btns] == list(MENUS)
    for (ctrl, label), want in zip(btns, MENUS.values(), strict=True):
        lst = re.search(rf'<div class="menu-list" id="{ctrl}" hidden>(.*?)</div>', nav, re.S)
        assert lst, ctrl
        links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)', lst.group(1))
        items = [(html.unescape(t).strip(), h) for h, t in links]
        assert items == want, label
    js = script()
    for key in ("ArrowDown", "ArrowUp", "Escape", "Home", "End", "'Tab'"):
        assert key in js, key
    assert "setAttribute('aria-expanded'" in js and "aria-current" in js


def test_one_view_per_page_and_hash_routes():
    p, js = page(), script()
    for v in ("map", "screener", "veille", "societes", "previsions", "methode"):
        assert f'id="view-{v}"' in p, v
        assert f"$('#view-{v}').style.display=" in js, v
    for route in ("'site'", "'screener'", "'veille'", "'societes'", "'societe'", "'previsions'", "'methode'",
                  "'retro-test'", "window.addEventListener('hashchange',route)"):
        assert route in js, route
    assert "location.hash='/site/'+enc(id)" in js  # fiche d'une mine : #/site/<id>


# -- thèmes, couleurs, décoration ----------------------------------------------------------------------------
def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"(--[\w-]+):([^;]+);", block))


def _themes() -> tuple[dict, dict, dict]:
    c = css()
    light = _tokens(re.search(r":root\{(.*?)\n\}", c, re.S).group(1))
    pat = r'@media \(prefers-color-scheme:dark\)\{:root:not\(\[data-theme="light"\]\)\{(.*?)\}\}'
    auto = _tokens(re.search(pat, c, re.S).group(1))
    forced = _tokens(re.search(r':root\[data-theme="dark"\]\{(.*?)\n\}', c, re.S).group(1))
    return light, auto, forced


def _lum(hexa: str) -> float:
    h = hexa.strip().lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_light_and_dark_themes_define_the_same_tokens():
    light, auto, forced = _themes()
    assert auto == forced  # sombre automatique (système) = sombre choisi
    colours = {k for k, v in light.items() if v.strip().startswith("#")}
    assert colours and colours <= set(auto), colours - set(auto)
    assert "<script>try{var t=localStorage.getItem('strate-theme')" in page()  # thème choisi avant le rendu


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_contrast_at_least_4_5(theme):
    light, dark, _ = _themes()
    t = {**light, **dark} if theme == "dark" else light
    for ink in ("--text", "--muted", "--faint", "--accent", "--positive", "--negative", "--warning"):
        for ground in ("--ground", "--surface"):
            assert contrast(t[ink], t[ground]) >= 4.5, (theme, ink, ground, contrast(t[ink], t[ground]))
    for bg, ink in (("--pos-bg", "--pos-ink"), ("--neg-bg", "--neg-ink"), ("--warn-bg", "--warn-ink"),
                    ("--neu-bg", "--neu-ink"), ("--accent-soft", "--accent-ink")):
        assert contrast(t[ink], t[bg]) >= 4.5, (theme, bg, ink)


def test_one_accent_and_no_hard_coded_colours_in_rules():
    c = re.sub(r':root(?::not\(\[data-theme="light"\]\))?(?:\[data-theme="dark"\])?\{[^}]*\}', "", css())
    c = re.sub(r'url\("[^"]*"\)', "", c)
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", c)  # toutes les couleurs passent par les jetons
    light, _, _ = _themes()
    assert sum(k.startswith("--accent") and not k.startswith("--accent-") for k in light) == 1


def test_no_generated_look():
    c, p = css(), page()
    for bad in ("linear-gradient", "radial-gradient", "conic-gradient", "backdrop-filter", "text-shadow",
                "text-transform:uppercase", "box-shadow:0 0"):  # box-shadow:0 0 … = une lueur
        assert bad not in c and bad not in script(), bad
    # emoji : aucun (★ U+2605, marqueur historique de « lien démontré », est un glyphe typographique)
    emoji = re.findall(r"[\U0001F000-\U0001FAFF☀-⛿✀-➿️]", p)
    assert set(emoji) <= {"★"}, set(emoji)
    # ombres : seulement ce qui flotte (menus, listes de résultats, infobulles et bulles de la carte)
    for sel in re.findall(r"([^{}]+)\{[^{}]*box-shadow:(?!none)", c):
        sel = sel.strip()
        assert re.search(r"menu-list|#results|#fc_res|leaflet-(tooltip|popup)", sel), sel


# -- contrat d'API ----------------------------------------------------------------------------------------
def _api_routes() -> set[str]:
    return set(re.findall(r'@app\.(?:get|post)\("(/api/[^"]+)"', API.read_text("utf-8")))


def test_pages_only_use_contract_routes():
    contract = set(re.findall(r"`GET (/api/[\w/]+)", section3()))
    assert len(contract) >= 5  # les cinq routes d'origine, plus celles que la doc ajoute (synthèse)
    used = set(re.findall(r"'(/api/[\w/.]+)", script()))
    assert contract <= used, contract - used  # chaque route du § 3 sert à l'interface
    assert used <= _api_routes() | contract, used - _api_routes() - contract
    new = set(re.findall(r"'(/api/[\w/.]+)", pages_js()))
    allowed = contract | {"/api/search", "/api/screener", "/api/site", "/api/calibration",
                          "/api/correlations/summary"}
    assert new <= allowed, new - allowed


def _doc_fields(route_pat: str, key: str) -> set[str]:
    row = re.search(route_pat + r".*?\"" + key + r"\":\[\{([^}]*)\}", section3())
    assert row, (route_pat, key)
    return {f.strip() for f in row.group(1).split(",")}


def _used(var: str, js: str) -> set[str]:
    return set(re.findall(rf"\b{var}\??\.(\w+)", js))


def test_pages_read_only_contract_fields():
    js = pages_js()
    companies = _doc_fields(r"`GET /api/companies\?", "rows")
    backtest = _doc_fields(r"`GET /api/forecast/backtest`", "rows")
    # n : champ des corrélations de /api/company (même forme, n au lieu de n_points)
    corr = _doc_fields(r"`GET /api/companies/correlations`", "rows") | {"n"}
    prev = _doc_fields(r"`GET /api/forecast/site", "prevision")
    mines = _doc_fields(r"`GET /api/company\?", "mines")
    inv = _doc_fields(r"`GET /api/company\?", "investissements")
    summary = {"domaine", "x", "y", "coefficient", "type", "lo", "hi", "n", "verdict", "source"}
    # contrat partagé de cette révision (producteurs : companies, forecast, api), lu défensivement par la page
    corr |= {"verdict_r", "verdict_rho", "q_value", "q_r", "q_rho", "lag", "rho_lo", "rho_hi"}
    summary |= {"unite_n", "n_bootstrap_unit"}
    backtest |= {"skill_lo", "skill_hi", "note", "r_niveau", "rho_niveau"}
    prev |= {"horizon", "mesure"}
    for var, allowed in (("co", companies), ("bt", backtest), ("bt2", backtest), ("cc", corr), ("fp", prev),
                         ("mi", mines), ("iv", inv), ("sr", summary)):
        got = _used(var, js)
        assert got, var
        assert got <= allowed, (var, got - allowed)


def test_server_strings_never_interpolated_raw():
    raw = re.findall(r"\$\{(?:co|bt|bt2|cc|fp|mi|iv|sr|pf|d|q|g|f|x)\.\w+\}", pages_js())
    assert not raw, raw
    js = pages_js()
    assert "safeUrl(f.source)" in js and "safeUrl(iv.url)" in js  # liens externes filtrés (ni javascript:)


def test_every_relationship_carries_the_badge():
    js = script()
    for fn in ("function ccRow", "function fcBadges", "const calRow", "function srRow", "const vBadge"):
        body = js[js.index(fn):js.index(fn) + 2500]
        assert "corrBadge(" in body, fn
    assert "shown.map(srRow)" in js  # page « toutes les corrélations »
    for fn in ("function fcQuality", "async function btPage"):  # qualité des prévisions : badges de fcBadges
        assert "fcBadges(" in js[js.index(fn):js.index(fn) + 2500], fn
    badge = js[js.index("function corrBadge"):js.index("// couleurs des types")]
    for part in ("IC 90 %", "n = ", "verdict", "aria-label"):
        assert part in badge, part


# -- comportements (Node) ---------------------------------------------------------------------------------
HARNESS = r"""
const fs=require('fs'), vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8'), CAL=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const src=html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const els={}, fetches=[];
function mk(sel){ const attrs={}; return {sel,style:{},dataset:{},innerHTML:'',textContent:'',value:'',
  checked:false,hidden:false,disabled:false,isConnected:true,className:'',listeners:{},
  classList:{toggle(){},add(){},remove(){},contains:()=>false},
  addEventListener(t,f){this.listeners[t]=f;},setAttribute(k,v){attrs[k]=String(v);},
  getAttribute:k=>attrs[k]??null,removeAttribute(k){delete attrs[k];},querySelectorAll:()=>[],
  querySelector:s=>$el(sel+' '+s),insertAdjacentHTML(){},get parentElement(){return $el(sel+'::parent');}}; }
function $el(s){ return els[s]??=mk(s); }
function fetchStub(url,opts){ return new Promise((res,rej)=>fetches.push({url,opts,done:false,
  ok:b=>res(new Response(JSON.stringify(b),{status:200})),status:s=>res(new Response('{}',{status:s}))})); }
function take(part){ const f=fetches.find(f=>!f.done&&f.url.includes(part));
  if(!f) throw new Error('pas de requête '+part+' parmi '+fetches.filter(f=>!f.done).map(f=>f.url));
  f.done=true; return f; }
const flush=async()=>{ for(let i=0;i<8;i++) await new Promise(r=>setImmediate(r)); };
const mapStub={setView(){return mapStub;},getZoom:()=>2,removeLayer(){},on(){return mapStub;},
  invalidateSize(){},hasLayer:()=>false,getContainer:()=>({offsetWidth:800}),
  getBounds:()=>({getWest:()=>0,getSouth:()=>0,getEast:()=>1,getNorth:()=>1})};
const marker=()=>{const o={on:()=>o,bindTooltip:()=>o,addTo:()=>o};return o;};
const L={map:()=>mapStub,tileLayer:()=>({addTo(){}}),canvas:()=>({}),circleMarker:marker,
  rectangle:()=>({addTo:()=>({})}),latLng:()=>({toBounds:()=>({})})};
const charts=[]; class Chart{constructor(el,cfg){this.el=el;this.cfg=cfg;charts.push(this);} destroy(){}}
const loc={hash:process.argv[4]||''};
const ctx=vm.createContext({document:{querySelector:$el,querySelectorAll:()=>[]},L,Chart,fetch:fetchStub,
  location:loc,window:{addEventListener(){}},setTimeout:()=>0,clearTimeout(){},setInterval:()=>0,clearInterval(){},
  console:{error(){},log(){}},Response,TypeError,URLSearchParams,Proxy});
vm.runInContext(src+`
globalThis.P={corrBadge,parseRoute,show,route,setCAL:c=>{CAL=c;},calReady:v=>{CAL_READY=v;},PG,view:()=>VIEW,sub:()=>SUB,
  cur:()=>CURRENT};`,ctx);
const P=ctx.P, out={};
async function check(name,fn){ try{ const r=await fn(); out[name]=r===true?true:String(r); }
  catch(e){ out[name]='exception: '+(e&&e.stack||e); } }
const clean=s=>!/undefined|NaN|\[object/.test(s);
const done=()=>fetches.forEach(f=>f.done=true);
(async()=>{
  await check('badge',()=>{
    const a=P.corrBadge({type:'rho',v:0.42,lo:0.18,hi:0.61,n:34});
    const b=P.corrBadge({type:'r',v:0.1,lo:-0.2,hi:0.3,n:12,verdict:'<img src=x> non démontré'});
    const c=P.corrBadge({type:'lift',v:2.5,lo:1.2,hi:4,n:40,nUnit:'mois'});
    const d=P.corrBadge({type:'r',v:null}), e=P.corrBadge({type:'rho',v:0.3,n:20});
    const f=P.corrBadge({type:'r',v:0.14,lo:0.01,hi:0.25,n:50,verdict:'non démontré : contient 0'});
    const g=P.corrBadge({type:'rho',v:-0.0001,lo:-0.004,hi:0.3,n:9});
    const h=P.corrBadge({type:'phi',v:0.12,lo:0.03,hi:0.2,n:14,nUnit:'mines',verdict:'non démontré'});
    const i=P.corrBadge({type:'rho',v:0.3,n:20,ciNote:'intervalle non publié par le rétro-test'});
    // sans verdict du serveur : verdict calculé, dit comme tel, jamais avec l'accent « démontré »
    const ok=a.includes('ρ')&&a.includes('+0,42')&&a.includes('IC 90 % +0,18 à +0,61')&&a.includes('n = 34')
      &&a.includes('lien positif (calculé d’après l’IC 90 % seul)')&&a.includes('p-pos')&&!a.includes('p-acc')
      &&!b.includes('<img')&&b.includes('&lt;img')&&b.includes('p-neu')
      &&c.includes('lift ×')&&c.includes('2,50')&&c.includes('risque accru (calculé')
      &&c.includes('n = 40 mois')
      &&d.includes('non calculable')&&e.includes('IC 90 % indisponible')&&e.includes('p-warn')
      // verdict du serveur affiché tel quel, même si l'IC affiché exclut 0 : le badge ne l'améliore pas
      &&f.includes('non démontré : contient 0')&&!f.includes('lien positif')&&f.includes('p-neu')
      &&g.includes('0,00')&&g.includes('−0,004')&&!g.includes('−0,00 ')
      &&h.includes('φ')&&h.includes('n = 14 mines')&&h.includes('>non démontré<')&&h.includes('p-neu')
      &&i.includes('intervalle non publié par le rétro-test')&&i.includes('cb-note')
      &&[a,b,c,d,e,f,g,h,i].every(clean);
    return ok||[a,b,c,d,e,f,g,h,i].join('\n');
  });
  await check('badge_never_upgrades_server_verdict',()=>{
    // cas mesurés par la revue : chaque verdict du serveur reste tel quel, style neutre ou avertissement
    const D='lien positif démontré (IC 90 % de ρ au-dessus de 0)';
    const I='intervalle incalculable : chiffres indicatifs';
    const cases=[[{type:'lift',v:2.1,lo:1.3,hi:3.4,n:20,verdict:'exploratoire'},'p-warn'],
      [{type:'lift',v:1.8,lo:1.1,hi:2.9,n:30,verdict:'non démontré'},'p-neu'],
      [{type:'lift',v:3,lo:1.2,hi:6,n:4,verdict:'trop peu de données'},'p-warn'],
      [{type:'rho',v:0.7,lo:0.2,hi:0.9,n:6,verdict:'trop peu de données (n = 6, sociétés = 2)'},'p-warn'],
      [{type:'rho',v:0.7,lo:0.2,hi:0.9,n:60,verdict:I},'p-warn'],
      // verdict « démontré » mais IC affiché contenant 0 : texte gardé, style neutre (jamais l'accent)
      [{type:'rho',v:0.1,lo:-0.1,hi:0.3,n:60,verdict:D},'p-neu'],
      // l'appelant impose « demo » sur un verdict non démontré : refusé
      [{type:'lift',v:2,lo:1.2,hi:3,n:40,verdict:'non démontré',state:'demo'},'p-neu']];
    const bad=cases.filter(([o,cls])=>{ const h=P.corrBadge(o);
      return !h.includes('>'+o.verdict.replace(/&/g,'&amp;')+'<')||!h.includes(cls)||h.includes('p-acc')
        ||/risque accru|lien positif \(/.test(h.replace(o.verdict,'')); });
    const demo=P.corrBadge({type:'rho',v:0.5,lo:0.2,hi:0.7,n:80,verdict:D});
    const star=P.corrBadge({type:'lift',v:2,lo:1.3,hi:3,n:40,verdict:'★ lien démontré',state:'demo'});
    return (!bad.length&&demo.includes('p-acc')&&star.includes('p-acc'))
      ||JSON.stringify(bad.map(([o])=>P.corrBadge(o)))+demo+star;
  });
  await check('routes',()=>{
    const cases=[['',{v:'map'}],['#/carte',{v:'map'}],['#/screener',{v:'screener'}],
      ['#/veille/correlations',{v:'veille',sub:'correlations'}],['#/veille/x',{v:'veille',sub:'flux'}],
      ['#/site/icmm%2F12',{v:'site',arg:'icmm/12'}],['#greenstone',{v:'site',arg:'greenstone'}],
      ['#/societes',{v:'societes',sub:'annuaire'}],['#/societes/capex',{v:'societes',sub:'capex'}],
      ['#/societe/rio_tinto',{v:'societes',sub:'fiche',arg:'rio_tinto'}],
      ['#/previsions',{v:'previsions',sub:'mine',arg:null}],
      ['#/previsions/mine/a%20b',{v:'previsions',sub:'mine',arg:'a b'}],
      ['#/previsions/retro-test',{v:'previsions',sub:'retro'}],['#/methode/sources',{v:'methode',sub:'sources'}],
      ['#/methode',{v:'methode',sub:'calibration'}],['#/nimporte',{v:'map'}]];
    const bad=cases.filter(([h,w])=>JSON.stringify(P.parseRoute(h))!==JSON.stringify(w));
    return !bad.length||JSON.stringify(bad.map(([h])=>[h,P.parseRoute(h)]));
  });
  await check('one_view_at_a_time',()=>{
    P.show('methode','sources');
    const vs=['map','screener','veille','societes','previsions','methode'];
    const vis=vs.filter(v=>$el('#view-'+v).style.display!=='none');
    return (JSON.stringify(vis)==='["methode"]'&&P.view()==='methode'&&P.sub()==='sources')
      ||JSON.stringify(vis);
  });
  await check('directory_empty_message',async()=>{
    done(); P.PG.co=null; P.show('societes','annuaire');
    take('/api/companies?limit=5000').ok({total:0,rows:[],
      message:'Aucune donnée société : uv run strate companies refresh'});
    await flush(); const h=$el('#co_list').innerHTML;
    return h.includes('Aucune donnée société')||h;
  });
  await check('directory_escapes_names',async()=>{
    done(); P.PG.co=null; P.show('societes','annuaire');
    take('/api/companies?limit=5000').ok({total:1,rows:[{id:'x"><b',nom:'<script>alert(1)</script>',pays:'Chile',
      tickers:['NYSE:<X>'],cik:'1',n_mines:2,capex_dernier:1.5e9,capex_annee:2025,effectifs:null,devise:'USD'}]});
    await flush(); const h=$el('#co_list').innerHTML;
    return (!h.includes('<script>')&&h.includes('&lt;script&gt;')&&h.includes('NYSE:&lt;X&gt;')
      &&h.includes('1,50 Md')&&!h.includes('"><b')&&clean(h))||h;
  });
  await check('directory_sorts_amounts_within_currency',async()=>{
    done(); P.PG.co=null; P.PG.coSort={k:'n_mines',desc:true}; P.show('societes','annuaire');
    const co=(id,dev,cap,nm)=>({id,nom:id,devise:dev,capex_dernier:cap,n_mines:nm,tickers:[]});
    take('/api/companies?limit=5000').ok({total:5,rows:[co('A','USD',1e9,3),co('B','CAD',5e9,1),
      co('C','USD',2e9,2),co('D','AUD',3e9,9),co('E',null,null,4)]});
    await flush(); const order=h=>[...h.matchAll(/data-href="#\/societe\/(\w)"/g)].map(m=>m[1]).join('');
    const byMines=order($el('#co_list').innerHTML);
    P.PG.coSort={k:'capex_dernier',desc:true}; P.show('societes','annuaire'); await flush();
    const h=$el('#co_list').innerHTML, byCapex=order(h);
    P.PG.coSort={k:'capex_dernier',desc:false}; P.show('societes','annuaire'); await flush();
    const up=order($el('#co_list').innerHTML);
    // B (5 Md CAD) et D (3 Md AUD) ne passent jamais devant les USD ; montant inconnu en dernier
    return (byMines==='DEACB'&&byCapex==='CADBE'&&up==='ACDBE'&&h.includes('à l’intérieur de chaque devise')
      &&h.includes('USD, AUD, CAD'))||JSON.stringify({byMines,byCapex,up});
  });
  await check('company_sheet_with_badges',async()=>{
    done(); P.show('societes','fiche','rio');
    take('/api/company?id=rio').ok({profil:{name:'Rio',tickers:['NYSE:RIO'],cik:'0000863064'},
      mines:[{site_id:'m1',nom:'Mine <1>',part:0.6}],
      faits:[{indicateur:'capex',annee:2025,valeur:1e10,devise:'USD',source:'javascript:alert(1)',formulaire:'20-F'}],
      investissements:[{date:'2026-10-01',montant:2e8,devise:'USD',categorie:'expansion',titre:'<b>Plan</b>',url:'https://e.org/a',site_id:'m1'}],
      correlations:[{x:'capex_glog',y:'prod_growth',decalage:1,r:0.3,r_lo:0.1,r_hi:0.5,rho:0.2,n:40}]});
    await flush(); const h=$el('#co_sheet').innerHTML;
    return (h.includes('IC 90 % +0,10 à +0,50')&&h.includes('n = 40')&&h.includes('croissance du CAPEX')
      &&!h.includes('javascript:')&&h.includes('Mine &lt;1&gt;')&&h.includes('60 %')
      &&h.includes('&lt;b&gt;Plan')
      &&charts.some(c=>c.el===$el('#co_cap'))&&clean(h))||h;
  });
  await check('company_sheet_uses_retained_fact',async()=>{
    const F=(annee,valeur,devise,origine,retenu)=>({indicateur:'capex',annee,valeur,devise,origine,retenu});
    const run=async(id,faits)=>{ done(); P.show('societes','fiche',id);
      take('/api/company?id='+id).ok({profil:{name:'R',alias:['Rio'],sec_url:'https://x',suivie:true},mines:[],faits,
        investissements:[],correlations:[]});
      await flush(); return [charts.filter(c=>c.el===$el('#co_cap')).at(-1),$el('#co_sheet').innerHTML]; };
    // le serveur marque le fait retenu : jamais le premier venu (ici Wikidata en EUR, rangé avant)
    const [c1,h1]=await run('r1',[F(2024,9e9,'EUR','wikidata',false),F(2024,1e10,'USD','sec',true),
      F(2023,8e9,'USD','sec',true),F(2022,5e9,'CAD','sec',true)]);
    // serveur ancien sans drapeau : SEC d'abord
    const [c2]=await run('r2',[F(2024,9e9,'EUR','wikidata'),F(2024,1e10,'USD','sec')]);
    const d1=c1.cfg.data, d2=c2.cfg.data.datasets[0].data;
    return (JSON.stringify(d1.labels)==='["2023","2024"]'&&JSON.stringify(d1.datasets[0].data)==='[8,10]'
      &&h1.includes('1 exercice dans une autre devise non tracé')&&h1.includes('Md USD')
      &&h1.includes('Autres noms')&&h1.includes('Mine de la liste suivie')&&!h1.includes('sec url')
      &&JSON.stringify(d2)==='[10]')||JSON.stringify([d1,d2])+h1;
  });
  await check('company_head_never_gets_raw_tickers',async()=>{
    done(); const el=$el('#co_sub'), seen=[]; let v='';
    Object.defineProperty(el,'innerHTML',{configurable:true,get:()=>v,
      set:x=>{ seen.push(String(x)); v=String(x); }});
    P.show('societes','fiche','t1');
    take('/api/company?id=t1').ok({profil:{name:'T',tickers:['<img src=x onerror=alert(1)>'],cik:'1'},
      mines:[],faits:[],investissements:[],correlations:[]});
    await flush(); delete el.innerHTML; el.innerHTML=v;
    return (seen.length>0&&seen.every(x=>!x.includes('<img'))&&v.includes('&lt;img'))||JSON.stringify(seen);
  });
  await check('company_corr_verdicts_per_coefficient',async()=>{
    done(); P.PG.co=null; P.show('societes','capex');
    const base={x:'capex_glog',y:'prod_growth',r:0.5,r_lo:0.2,r_hi:0.7,rho:0.4,rho_lo:0.1,rho_hi:0.6,
      n_points:30};
    take('/api/companies/correlations').ok({methode:'m',rows:[
      {...base,decalage:0,n_societes:4,q_value:0.04,verdict:'non démontré : l’IC 90 % de ρ contient 0',
        verdict_r:'trop peu de données (n = 30, sociétés = 4)',verdict_rho:'non démontré (q ≥ 0,10)'},
      {...base,decalage:1,n_societes:2,verdict:'trop peu de données (n = 30, sociétés = 2)'},
      {...base,decalage:2,n_societes:40,verdict:'lien positif démontré (IC 90 % de ρ au-dessus de 0)'},
      // q propre à chaque coefficient : q_value (= q_rho) ne doit jamais passer pour la q de r
      {...base,decalage:3,n_societes:40,q_r:0.3,q_rho:0.04,q_value:0.04,
        verdict_r:'non démontré (q = 0.30 ≥ 0.10)',verdict_rho:'lien positif démontré'}]});
    take('/api/companies?limit=5000').ok({total:0,rows:[]});
    await flush(); const h=$el('#co_capex').innerHTML;
    const rows=h.split('class="cbrow"').slice(1);
    const pills=r=>[...r.matchAll(/class="pill ([\w-]+)">([^<]*)</g)].map(m=>[m[1],m[2]]);
    const [a,b,c,d]=rows.map(pills);
    // ligne 1 : chaque coefficient porte son propre verdict ; ligne 2 (serveur ancien) : « trop peu »
    // vaut pour r ; ligne 3 : le verdict de ρ n'est jamais recopié sur r (verdict calculé, dit comme tel)
    const ok=rows.length===4&&a[0][1].startsWith('trop peu')&&a[0][0]==='p-warn'
      &&a[1][1]==='non démontré (q ≥ 0,10)'&&a[1][0]==='p-neu'
      &&b[0][1].startsWith('trop peu')&&b[1][1].startsWith('trop peu')
      &&c[0][1].includes('calculé')&&c[0][0]==='p-pos'&&c[1][0]==='p-acc'
      &&rows[0].includes('q (Benjamini-Hochberg) : ρ 0,04')&&!rows[0].includes('q (Benjamini-Hochberg) : r')
      &&rows[0].includes('rééchantillonné sur 4 sociétés')
      &&rows[3].includes('q (Benjamini-Hochberg) : r 0,3, ρ 0,04')&&d[0][1].startsWith('non démontré')
      &&clean(h);
    return ok||JSON.stringify([a,b,c]);
  });
  await check('capex_page_empty',async()=>{
    done(); P.PG.co=null; P.show('societes','capex');
    take('/api/companies/correlations').ok({rows:[],methode:'m',message:'Aucune donnée société'});
    take('/api/companies?limit=5000').ok({total:0,rows:[],message:'Aucune donnée société'});
    await flush(); const h=$el('#co_capex').innerHTML;
    return (h.includes('Aucune donnée société')&&clean(h))||h;
  });
  await check('backtest_page',async()=>{
    done(); P.show('previsions','retro');
    take('/api/forecast/backtest').ok({run_at:'2026-11-27T00:00:00',periode:{reglage:'≤2014',test:'2015–2021'},
      rows:[{model:'naif',horizon:1,n_mines:600,n_points:2600,r:null,rho:null,skill:0,facteur_median:1.1,bon_sens:0.5,
      couverture80:0.8,couverture90:0.89},{model:'tendance',horizon:1,n_mines:600,n_points:2600,r:0.11,r_lo:0.03,
      r_hi:0.21,rho:0.03,rho_lo:-0.001,rho_hi:0.07,skill:0.01,facteur_median:1.1,bon_sens:0.5,couverture80:0.79,
      couverture90:0.89}],retenu:{'1':'tendance'},methode:'walk-forward'});
    await flush(); const h=$el('#bt_out').innerHTML;
    return (h.includes('retenu')&&h.includes('IC 90 % +0,03 à +0,21')&&h.includes('non calculable')
      &&h.includes('n = '+(2600).toLocaleString('fr-FR')+' points')&&h.includes('bat le naïf')
      &&h.includes('cible 90 %')&&clean(h))||h;
  });
  await check('backtest_empty',async()=>{
    done(); P.show('previsions','retro');
    take('/api/forecast/backtest').ok({rows:[],retenu:{},periode:{},message:'Rétro-test pas encore lancé'});
    await flush(); const h=$el('#bt_out').innerHTML; return h.includes('Rétro-test pas encore lancé')||h;
  });
  await check('forecast_without_series',async()=>{
    done(); P.show('previsions','mine','m9');
    take('/api/forecast/site?site=m9').ok(null);  // rétro-test déjà en cache (test précédent)
    await flush(); take('/api/site?site=m9').ok({nom:'Mine 9'}); await flush();
    const h=$el('#fc_out').innerHTML;
    return h.includes('Pas de série de production')||h;
  });
  await check('forecast_with_series',async()=>{
    done(); P.show('previsions','mine','m1');
    take('/api/forecast/site?site=m1').ok({serie:'minerai extrait',historique:[{year:2019,value:2e6},
      {year:2020,value:2.2e6}],prevision:[{year:2021,model:'tendance',value:2.3e6,lo80:2e6,hi80:2.6e6,lo90:1.9e6,
      hi90:2.8e6}],qualite:{r:0.4,rho:0.3,skill:0.1,facteur_median:1.2,couverture90:0.9,n:30}});
    await flush(); take('/api/site?site=m1').ok({nom:'Mine <1>'}); await flush();
    const h=$el('#fc_out').innerHTML, c=charts.at(-1);
    return (h.includes('IC 90 % indisponible')&&h.includes('2,3 Mt')&&c.el===$el('#fc_c')
      &&c.cfg.data.datasets.some(d=>d.borderDash)&&$el('#fc_h1').textContent.includes('Mine <1>')
      &&clean(h))||h;
  });
  await check('forecast_contract_quality_and_horizons',async()=>{
    done(); P.show('previsions','mine','m2');
    const q={model:'tendance',horizon:1,r:0.2,r_lo:0.1,r_hi:0.3,rho:0.15,rho_lo:0.05,rho_hi:0.25,skill:0.04,
      skill_lo:0.01,skill_hi:0.07,couverture80:0.79,couverture90:0.9,facteur_median:1.2,n:2676,
      n_mines:610,portee:'global'};
    const row=(y,h,m)=>({year:y,horizon:h,mesure:m,model:'tendance',value:2e6,lo80:1.8e6,hi80:2.2e6,
      lo90:1.7e6,hi90:2.4e6});
    take('/api/forecast/site?site=m2').ok({serie:'minerai',
      historique:[{year:2020,value:2e6},{year:2021,value:2e6}],
      prevision:[row(2022,1,true),row(2025,4,false)],qualite:q,suivi_site:{n:6,facteur_median:1.3}});
    await flush(); take('/api/site?site=m2').ok({nom:'Mine 2'}); await flush();
    const h=$el('#fc_out').innerHTML;
    const ok=h.includes('IC 90 % +0,10 à +0,30')&&h.includes('IC 90 % +0,05 à +0,25')
      &&h.includes('n = '+(2676).toLocaleString('fr-FR')+' points')
      &&h.includes('score global du rétro-test')&&h.includes('pas un score propre à cette mine')
      &&h.includes('610 mines')&&h.includes('skill +0,04 (IC 90 % +0,01 à +0,07)')&&h.includes('4 ans')
      &&(h.match(/non mesuré/g)||[]).length>=2&&h.includes('2025 : horizon au-delà')
      &&h.includes('Sur cette mine seule')
      &&!h.includes('IC 90 % indisponible')&&clean(h);
    return ok||h;
  });
  await check('forecast_naive_quality_in_levels',async()=>{
    done(); P.show('previsions','mine','m3');
    const q={model:'naif',horizon:1,r:null,rho:null,r_niveau:0.95,r_niveau_lo:0.93,r_niveau_hi:0.97,
      rho_niveau:0.94,rho_niveau_lo:0.9,rho_niveau_hi:0.96,n:2676,n_mines:610,portee:'global',
      note:'le naïf recopie la dernière valeur : aucune variation prévue'};
    take('/api/forecast/site?site=m3').ok({historique:[{year:2021,value:1e6}],
      prevision:[{year:2022,horizon:1,mesure:true,model:'naif',value:1e6}],qualite:q});
    await flush(); take('/api/site?site=m3').ok({nom:'Mine 3'}); await flush();
    const h=$el('#fc_out').innerHTML;
    return (h.includes('r niveau')&&h.includes('ρ niveau')&&h.includes('IC 90 % +0,93 à +0,97')
      &&h.includes('aucune variation prévue')&&h.includes('non calculable')&&clean(h))||h;
  });
  await check('all_correlations_fallback',async()=>{
    done(); P.setCAL(CAL); P.show('methode','correlations');
    take('/api/correlations/summary').status(404); await flush();
    take('/api/companies/correlations').ok({rows:[]}); take('/api/forecast/backtest').ok({rows:[]});
    await flush();
    const h=$el('#me_corr').innerHTML;
    return (h.includes('calibration')&&h.includes('IC 90 %')&&h.includes('n = ')&&clean(h))||h;
  });
  await check('calibration_page',()=>{
    P.setCAL(CAL); P.show('methode','calibration'); const h=$el('#me_cal').innerHTML;
    P.setCAL({}); P.show('methode','calibration'); const e=$el('#me_cal').innerHTML;
    P.calReady(true); P.show('methode','calibration'); const e2=$el('#me_cal').innerHTML; P.calReady(false);
    P.setCAL(CAL);
    return (h.includes('IC 90 %')&&h.includes('Fisher z')&&/pas<\/b> été confrontées/.test(h)&&clean(h)
      &&e.includes('Chargement')&&e2.includes('Calibration absente'))||[h,e,e2].join('\n');
  });
  process.stdout.write(JSON.stringify(out));
})();
"""

# lien direct ou rechargement : la page est routée avant la fin de init() (calibration pas encore là)
BOOT_CHECKS = r"""
  await check('deep_link_calibration',async()=>{
    const before=$el('#me_cal').innerHTML;
    take('/api/catalog/points').ok([]); take('/api/catalog/stats').ok({}); take('/api/radar').ok([]);
    take('/api/calibration').ok(CAL); await flush(); take('/api/sites').ok([]); await flush();
    const after=$el('#me_cal').innerHTML;
    return (before.includes('Chargement')&&!before.includes('absente')&&after.includes('IC 90 %')
      &&clean(after))
      ||[before,after].join('\n');
  });
  process.stdout.write(JSON.stringify(out));
})();
"""
# fiche rechargée avant la calibration : redessinée une fois quand elle arrive (note et badges de calibration)
BOOT_SITE = r"""
  await check('deep_link_calibration',async()=>{
    take('/api/site?site=m1').ok({nom:'M',lat:0,lon:0,rayon_m:1000}); await flush();
    const n0=fetches.filter(f=>f.url.includes('/api/site?site=m1')).length;
    take('/api/catalog/points').ok([]); take('/api/catalog/stats').ok({}); take('/api/radar').ok([]);
    take('/api/calibration').ok(CAL); await flush(); take('/api/sites').ok([]); await flush();
    const n1=fetches.filter(f=>f.url.includes('/api/site?site=m1')).length;
    return (n0===1&&n1===2&&P.cur()==='m1')||JSON.stringify({n0,n1});
  });
  process.stdout.write(JSON.stringify(out));
})();
"""

# chaque API répond vide (null, {} ou []) : aucune exception, aucun « undefined » à l'écran, sur toutes
# les pages
EMPTY_ALL = r"""
  const BODY=JSON.parse(process.argv[5]);
  const answer=async()=>{ for(let k=0;k<6;k++){
    fetches.filter(f=>!f.done).forEach(f=>{ f.done=true; f.ok(BODY); }); await flush(); } };
  await answer();
  const routes=['#/carte','#/screener','#/veille/flux','#/veille/correlations','#/veille/suivies',
    '#/veille/etat',
    '#/societes','#/societes/capex','#/societes/effectifs','#/societe/x','#/previsions','#/previsions/mine/m1',
    '#/previsions/retro-test','#/methode/calibration','#/methode/sources','#/methode/correlations','#/site/m1'];
  const bad=[];
  for(const h of routes){ loc.hash=h; try{ P.route(); }catch(e){ ERRS.push(h+' '+e.stack); } await answer();
    for(const [k,el] of Object.entries(els)){ const t=String(el.innerHTML||'')+String(el.textContent||'');
      if(/undefined|NaN|\[object/.test(t)) bad.push(h+' '+k+' '+t.slice(0,160)); } }
  process.stdout.write(JSON.stringify({errs:ERRS.slice(0,10),bad:[...new Set(bad)].slice(0,10),
    co:$el('#co_list').innerHTML,cal:$el('#me_cal').innerHTML}));
})();
"""

CHECKS = ["badge", "badge_never_upgrades_server_verdict", "routes", "one_view_at_a_time",
          "directory_empty_message", "directory_escapes_names", "directory_sorts_amounts_within_currency",
          "company_sheet_with_badges", "company_sheet_uses_retained_fact",
          "company_head_never_gets_raw_tickers", "company_corr_verdicts_per_coefficient",
          "capex_page_empty", "backtest_page", "backtest_empty", "forecast_without_series",
          "forecast_with_series", "forecast_contract_quality_and_horizons",
          "forecast_naive_quality_in_levels", "all_correlations_fallback", "calibration_page"]


def _run(tmp_path_factory, body: str, hash_: str = "") -> dict:
    h = tmp_path_factory.mktemp("ui_v2") / "harness.js"
    h.write_text(body, "utf-8")
    cal = ROOT / "src" / "strate" / "data" / "calibration.json"
    p = subprocess.run([NODE, str(h), str(HTML), str(cal), hash_], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


@pytest.fixture(scope="module")
def node_results(tmp_path_factory):
    if NODE is None:
        pytest.skip("Node absent : comportements de la page non rejoués")
    return _run(tmp_path_factory, HARNESS)


@needs_node
@pytest.mark.parametrize("name", CHECKS)
def test_page_behaviour(node_results, name):
    assert node_results.get(name) is True, node_results.get(name)


@needs_node
@pytest.mark.parametrize("hash_", ["#/methode/calibration", "#/site/m1"])
def test_deep_link_redrawn_after_boot(tmp_path_factory, hash_):
    prefix = HARNESS[:HARNESS.index("(async()=>{\n") + len("(async()=>{\n")]
    res = _run(tmp_path_factory, prefix + (BOOT_SITE if "site" in hash_ else BOOT_CHECKS), hash_)
    assert res.get("deep_link_calibration") is True, res


@needs_node
@pytest.mark.parametrize("body", ["null", "{}", "[]"])
def test_every_page_survives_empty_api(tmp_path_factory, body):
    prefix = HARNESS[:HARNESS.index("(async()=>{\n") + len("(async()=>{\n")]
    spy = "console:{error(...a){ERRS.push(a.join(' '));},log(){}}"
    prefix = prefix.replace("console:{error(){},log(){}}", spy)
    trap = "process.on('unhandledRejection',e=>ERRS.push('rejet : '+(e&&e.stack||e)));"
    prefix = "const ERRS=[]; " + trap + "\n" + prefix
    h = tmp_path_factory.mktemp("ui_empty") / "harness.js"
    h.write_text(prefix + EMPTY_ALL, "utf-8")
    cal = ROOT / "src" / "strate" / "data" / "calibration.json"
    cmd = [NODE, str(h), str(HTML), str(cal), "", body]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)
    assert res["errs"] == [] and res["bad"] == [], res
    assert "Aucune donnée société" in res["co"]  # message clair plutôt qu'une table vide
    assert "Calibration absente" in res["cal"]
