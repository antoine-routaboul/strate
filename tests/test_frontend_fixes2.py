"""Interface web, seconde revue : analyse récente / données anciennes, suivi jusqu'à l'état final, note de
calibration par population, mois « image récente » calculés depuis aujourd'hui, légende de la carte.

Même principe que test_frontend_fixes : le script de la page tourne sous Node avec un DOM, Leaflet, Chart.js,
fetch et des minuteurs factices. Sans Node, seuls les tests statiques tournent.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src" / "strate" / "web" / "index.html"
CAL = ROOT / "src" / "strate" / "data" / "calibration.json"


def page() -> str:
    return HTML.read_text("utf-8")


def script() -> str:
    return re.search(r"<script>\n(.*?)</script>", page(), re.S).group(1)


# -- invariants statiques --------------------------------------------------------------------------------
def test_legend_red_means_tracked_list():
    # le cercle rouge n'est dessiné que pour /api/sites (liste suivie) : la légende doit dire la même chose
    legend = re.search(r'<p class="meta legend">(.*?)</p>', page(), re.S).group(1)
    red = re.search(r'#ff5d5d[^>]*></i>([^<]*)<', legend).group(1)
    assert "suivie" in red and "analys" not in red, red


def test_no_hardcoded_recent_months():
    assert not re.search(r"'20\d\d-\d\d'", script())


def test_watch_does_not_end_on_analyse_flag():
    assert "s?.analyse){ stop()" not in script() and "||s?.analyse" not in script()


# -- comportements (Node) --------------------------------------------------------------------------------
HARNESS = r"""
const fs=require('fs'), vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const CALJSON=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const src=html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const els={}, fetches=[];
function mk(sel){ return {sel,style:{},dataset:{},innerHTML:'',textContent:'',value:'',checked:false,src:'',
  disabled:false,isConnected:true,className:'',listeners:{},classList:{toggle(){}},addEventListener(t,f){this.listeners[t]=f;},
  querySelectorAll:()=>[],querySelector:s=>$el(sel+' '+s),insertAdjacentHTML(){},
  get parentElement(){return $el(sel+'::parent');}}; }
function $el(s){ return els[s]??=mk(s); }
function fetchStub(url,opts){ return new Promise((res,rej)=>fetches.push({url,opts,done:false,
  ok:b=>res(new Response(JSON.stringify(b),{status:200})),status:s=>res(new Response('{}',{status:s}))})); }
function take(part){ const f=fetches.find(f=>!f.done&&f.url.includes(part));
  if(!f) throw new Error('pas de requête '+part); f.done=true; return f; }
const pending=part=>fetches.some(f=>!f.done&&f.url.includes(part));
let tid=0; const timers=new Map();
const setT=(fn)=>{const id=++tid;timers.set(id,{fn,rep:false});return id;};
const setI=(fn)=>{const id=++tid;timers.set(id,{fn,rep:true});return id;};
const clr=id=>timers.delete(id);
function runIntervals(){ for(const [,t] of [...timers]) if(t.rep) t.fn(); }
const intervals=()=>[...timers.values()].filter(t=>t.rep).length;
const flush=async()=>{ for(let i=0;i<6;i++) await new Promise(r=>setImmediate(r)); };
const mapStub={setView(){return mapStub;},getZoom:()=>2,removeLayer(){},on(){return mapStub;},
  invalidateSize(){},
  getContainer:()=>({offsetWidth:800}),getBounds:()=>({getWest:()=>0,getSouth:()=>0,getEast:()=>1,getNorth:()=>1})};
const marker=()=>{const o={on:()=>o,bindTooltip:()=>o,addTo:()=>o};return o;};
const L={map:()=>mapStub,tileLayer:()=>({addTo(){}}),canvas:()=>({}),circleMarker:marker,
  rectangle:()=>({addTo:()=>({})}),latLng:()=>({toBounds:()=>({})})};
class Chart{constructor(el,cfg){this.el=el;this.cfg=cfg;} destroy(){}}
const loc={hash:''};
const ctx=vm.createContext({document:{querySelector:$el,querySelectorAll:()=>[]},L,Chart,fetch:fetchStub,
  location:loc,window:{addEventListener(){}},setTimeout:setT,clearTimeout:clr,setInterval:setI,
  clearInterval:clr,console:{error(){},log(){}},Response,TypeError,URLSearchParams,Proxy});
vm.runInContext(src+`
globalThis.P={open,calNote,recentMonths,cur:()=>CURRENT,setCAL:c=>{CAL=c;}};`,ctx);
const P=ctx.P, out={};
const site=(nom,o={})=>Object.assign({nom,lat:45,lon:2,rayon_m:1000,analyse:false,donnees_anciennes:false,
  derniere_date:null,source_coordonnees:'x',production_emprise:null,production_reelle:null,statut_analyse:null},o);
const dossier=[{date:'2019-08-01',bare_km2:1,level:0},{date:'2019-10-01',bare_km2:1.1,level:0}];
async function check(name,fn){ try{ const r=await fn(); out[name]=r===true?true:String(r); }
  catch(e){ out[name]='exception: '+(e&&e.stack||e); } }
const clean=s=>!/undefined|NaN|null|\[object/.test(s);

(async()=>{
  take('/api/catalog/points').ok([]); take('/api/catalog/stats').ok({}); take('/api/radar').ok([]);
  take('/api/calibration').ok(CALJSON); await flush(); take('/api/sites').ok([]); await flush();

  // #13 : chaque chiffre attribué à sa population
  await check('calnote_populations',()=>{
    P.setCAL(CALJSON);
    const t=P.calNote(), me=CALJSON.modele_emprise;
    const f2=v=>Number(v).toLocaleString('fr-FR',{maximumFractionDigits:2});
    const first=t.split('.')[0];  // phrase « Calibration satellite sur N mines (fenêtre) »
    const want=[`${CALJSON.mines} mines`,`${CALJSON.fenetre[0]}–${CALJSON.fenetre[1]}`,'2017–2020',
      `${f2(me.minerai.rho)} sur ${me.minerai.n} mines de minerai`,
        `${f2(me.charbon.rho)} sur ${me.charbon.n} mines de charbon`,
      f2(CALJSON.taille.fenetre_seule.rho),'faiblement',"taille de l'emprise"];
    const miss=want.filter(w=>!t.includes(w));
    if(miss.length) return 'manque '+JSON.stringify(miss)+' dans '+t;
    if(first.includes(f2(me.minerai.rho))||first.includes('2017'))
      return 'chiffres du modèle dans la phrase calibration : '+first;
    return clean(t)||t;
  });
  await check('calnote_defensive',()=>{
    const cases=[{mines:12},{mines:12,modele_emprise:{minerai:{rho:0.5}}},
      {mines:12,taille:{fenetre_seule:{}}},
      {mines:12,fenetre:'x',modele_emprise:null,taille:null},{}];
    const r=cases.map(c=>{P.setCAL(c); return P.calNote();}); P.setCAL(CALJSON);
    return (r.slice(0,4).every(t=>t.includes('12 mines')&&clean(t))&&r[1].includes('0,5 (minerai)')
      &&r[4]==='')
      ||JSON.stringify(r);
  });

  // #17 : mois calculés depuis la date du jour, selon l'hémisphère
  await check('recent_months',()=>{
    const n=P.recentMonths(45,new Date(2026,10,24)), s=P.recentMonths(-30,new Date(2026,10,24));
    const j=P.recentMonths(60,new Date(2027,0,5));
    const ok=JSON.stringify(n)===JSON.stringify(['2026-09','2026-08','2026-07','2026-10'])
      &&JSON.stringify(s)===JSON.stringify(['2026-10','2026-09','2026-03','2025-10'])
      &&JSON.stringify(j)===JSON.stringify(['2026-09','2026-08','2026-07','2026-12']);
    return ok||JSON.stringify({n,s,j});
  });
  await check('fresh_page_uses_today_months',async()=>{
    P.open('F1'); take('site=F1').ok(site('Mine F1')); await flush();
    const d=$el('#detail').innerHTML, want=P.recentMonths(45)[0], src=$el('#i1').src;
    if(!d.includes('Analyser ce site')||!/pas encore été analysée/.test(d)) return d;
    if(!src.includes('month='+want)) return 'src='+src+' attendu '+want;
    $el('#i1').onload();
    return /Sentinel-2, \d{4}-\d\d/.test($el('#i1m').textContent)||$el('#i1m').textContent;
  });

  // #14 : données anciennes → fiche de signaux + bouton de mise à jour daté
  await check('stale_page_offers_update',async()=>{
    $el('#go').onclick=null;
    P.open('S1');
    take('site=S1').ok(site('Spring Creek',{donnees_anciennes:true,derniere_date:'2019-10'})); await flush();
    take('/api/dossier?site=S1').ok(dossier); take('/api/events?site=S1').ok([]); await flush();
    const d=$el('#detail').innerHTML;
    return (d.includes('Mettre à jour l’analyse')&&/Données anciennes/.test(d)&&d.includes('2019-10')
      &&!d.includes('Analyser ce site')&&typeof $el('#go').onclick==='function'&&clean(d))||d;
  });
  await check('recent_page_has_no_block',async()=>{
    P.open('R1');
    take('site=R1').ok(site('Mine R1',{analyse:true,derniere_date:'2026-10',statut_analyse:'terminé'}));
    await flush(); take('/api/dossier?site=R1').ok(dossier); take('/api/events?site=R1').ok([]);
    await flush();
    const d=$el('#detail').innerHTML;
    return (!d.includes('id="go"')&&d.includes('2026-10')&&intervals()===0)||d;
  });

  // #15 : le suivi ne s'arrête pas quand l'optique est enregistrée, seulement sur terminé / erreur
  await check('watch_until_final_status',async()=>{
    P.open('W1'); take('site=W1').ok(site('Mine W1',{statut_analyse:'en cours (1/3) : optique'}));
    await flush();
    if(intervals()!==1) return 'pas de suivi lancé';
    runIntervals(); await flush();
    take('site=W1').ok(site('Mine W1',{analyse:true,statut_analyse:'en cours (2/3) : attention Wikipedia'}));
    await flush();
    if(pending('/api/dossier?site=W1')||intervals()!==1) return 'suivi arrêté sur analyse=vrai';
    if($el('#go').textContent!=='en cours (2/3) : attention Wikipedia')
      return 'statut non affiché : '+$el('#go').textContent;
    runIntervals(); await flush();
    take('site=W1').ok(site('Mine W1',{analyse:true,statut_analyse:'terminé'})); await flush();
    take('/api/radar').ok([]); await flush();
    return (pending('site=W1')&&intervals()===0)||'fiche non rouverte à la fin';
  });
  await check('analysed_page_busy_keeps_watching',async()=>{
    for(const f of fetches) f.done=true;
    P.open('A1');
    take('site=A1').ok(site('Mine A1',{analyse:true,statut_analyse:'en cours (3/3) : radar'})); await flush();
    take('/api/dossier?site=A1').ok(dossier); take('/api/events?site=A1').ok([]); await flush();
    const d=$el('#detail').innerHTML;
    if(!d.includes('en cours (3/3) : radar')||intervals()!==1) return 'statut ou suivi absent : '+d;
    runIntervals(); await flush();
    take('site=A1').ok(site('Mine A1',{analyse:true,statut_analyse:'erreur : radar indisponible'}));
    await flush();
    if(intervals()!==0||!pending('site=A1')) return 'erreur : fiche non rouverte';
    take('site=A1').ok(site('Mine A1',{analyse:true,statut_analyse:'erreur : radar indisponible'}));
    await flush();
    take('/api/dossier?site=A1').ok(dossier); take('/api/events?site=A1').ok([]); await flush();
    const e=$el('#detail').innerHTML;
    return (e.includes('erreur : radar indisponible')&&/a échoué/.test(e)&&intervals()===0)||e;
  });
  await check('error_before_optical_shown_in_place',async()=>{
    for(const f of fetches) f.done=true;
    P.open('E1'); take('site=E1').ok(site('Mine E1',{statut_analyse:'en file'})); await flush();
    runIntervals(); await flush(); take('site=E1').ok(site('Mine E1',{statut_analyse:'erreur : STAC'}));
    await flush();
    return ($el('#jst').textContent==='erreur : STAC'&&!$el('#go').disabled&&intervals()===0
      &&!pending('site=E1'))
      ||`jst=${$el('#jst').textContent} int=${intervals()}`;
  });
  await check('launch_error_restores_label',async()=>{
    for(const f of fetches) f.done=true;
    P.open('S2'); take('site=S2').ok(site('Mine S2',{donnees_anciennes:true,derniere_date:'2019-10'}));
    await flush();
    take('/api/dossier?site=S2').ok(dossier); take('/api/events?site=S2').ok([]); await flush();
    const go=$el('#go'); go.dataset.label='Mettre à jour l’analyse'; go.onclick();
    take('/api/analyze?site=S2').status(500); await flush();
    return go.textContent==='Mettre à jour l’analyse'||go.textContent;
  });

  // jumeau d'une mine suivie : la suite de la fiche part de l'id suivi
  await check('twin_redirect',async()=>{
    for(const f of fetches) f.done=true;
    P.open('icmmT'); take('site=icmmT').ok(site('Mine suivie',{id:'watched',analyse:true})); await flush();
    const ok=pending('/api/dossier?site=watched')&&!pending('/api/dossier?site=icmmT')
      &&P.cur()==='watched'&&loc.hash==='watched';
    return ok||`cur=${P.cur()} hash=${loc.hash} ${fetches.filter(f=>!f.done).map(f=>f.url)}`;
  });
  process.stdout.write(JSON.stringify(out));
})();
"""

CHECKS = ["calnote_populations", "calnote_defensive", "recent_months", "fresh_page_uses_today_months",
          "stale_page_offers_update", "recent_page_has_no_block", "watch_until_final_status",
          "analysed_page_busy_keeps_watching", "error_before_optical_shown_in_place",
          "launch_error_restores_label", "twin_redirect"]


@pytest.fixture(scope="module")
def node_results(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node absent : comportements de la page non rejoués")
    h = tmp_path_factory.mktemp("front2") / "harness.js"
    h.write_text(HARNESS, "utf-8")
    p = subprocess.run([node, str(h), str(HTML), str(CAL)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


@pytest.mark.parametrize("name", CHECKS)
def test_page_behaviour(node_results, name):
    assert node_results.get(name) is True, node_results.get(name)
