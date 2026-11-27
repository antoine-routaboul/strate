"""Interface web (src/strate/web/index.html) : invariants statiques et comportements rejoués sous Node.

Le script de la page tourne dans un contexte Node avec un DOM, Leaflet, Chart.js, fetch et des minuteurs
factices : on peut ainsi rejouer les courses (réponses dans le désordre), les erreurs serveur et les textes
de calibration sans navigateur. Sans Node, seuls les tests statiques tournent.
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
def test_tooltips_escape_names():
    # Leaflet bindTooltip(chaîne) écrit en innerHTML : un libellé Wikidata ne doit jamais devenir du HTML
    calls = re.findall(r"\.bindTooltip\(([^)]*\)?)", script())
    assert calls and all(c.startswith("esc(") for c in calls), calls


def test_withdrawn_bare_area_estimate_not_read():
    assert "production_estimee" not in script()


def test_image_frame_positioned_in_every_state():
    assert ".imgs>div{position:relative}" in page()


def test_no_moveend_reload_gated_on_hidden_screener():
    assert "view-screener').style.display==='block'" not in script()


# -- comportements (Node) --------------------------------------------------------------------------------
HARNESS = r"""
const fs=require('fs'), vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const CALJSON=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const src=html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const els={}, fetches=[], tooltips=[], charts=[], mapCalls=[];
function mk(sel){ return {sel,style:{},dataset:{},innerHTML:'',textContent:'',value:'',checked:false,
  disabled:false,isConnected:true,className:'',listeners:{},classList:{toggle(){}},addEventListener(t,f){this.listeners[t]=f;},
  querySelectorAll:()=>[],querySelector:s=>$el(sel+' '+s),insertAdjacentHTML(){},
  get parentElement(){return $el(sel+'::parent');}}; }
function $el(s){ return els[s]??=mk(s); }
function fetchStub(url,opts){ return new Promise((res,rej)=>fetches.push({url,opts,done:false,
  ok:b=>res(new Response(JSON.stringify(b),{status:200})),status:s=>res(new Response('{}',{status:s})),
  fail:()=>rej(new TypeError('Failed to fetch'))})); }
function take(part){ const f=fetches.find(f=>!f.done&&f.url.includes(part));
  if(!f) throw new Error('pas de requête '+part); f.done=true; return f; }
let tid=0; const timers=new Map();
const setT=(fn)=>{const id=++tid;timers.set(id,{fn,rep:false});return id;};
const setI=(fn)=>{const id=++tid;timers.set(id,{fn,rep:true});return id;};
const clr=id=>timers.delete(id);
function runTimers(){ for(const [id,t] of [...timers]){ if(!t.rep) timers.delete(id); t.fn(); } }
const flush=async()=>{ for(let i=0;i<5;i++) await new Promise(r=>setImmediate(r)); };
const box={w:-10,s:-5,e:10,n:5}, container={offsetWidth:800};
const mapStub={setView(c){mapCalls.push(c);return mapStub;},getZoom:()=>2,removeLayer(){},
  on(){return mapStub;},invalidateSize(){},getContainer:()=>container,
  getBounds:()=>({getWest:()=>box.w,getSouth:()=>box.s,getEast:()=>box.e,getNorth:()=>box.n})};
const marker=()=>{const o={on:()=>o,bindTooltip:t=>{tooltips.push(t);return o;},addTo:()=>o};return o;};
const L={map:()=>mapStub,tileLayer:()=>({addTo(){}}),canvas:()=>({}),circleMarker:marker,
  rectangle:()=>({addTo:()=>({})}),latLng:()=>({toBounds:()=>({})})};
class Chart{constructor(el,cfg){this.el=el;this.cfg=cfg;charts.push(this);} destroy(){}}
const ctx=vm.createContext({document:{querySelector:$el,querySelectorAll:()=>[]},L,Chart,fetch:fetchStub,
  location:{hash:''},window:{addEventListener(){}},setTimeout:setT,clearTimeout:clr,setInterval:setI,
  clearInterval:clr,console:{error(){},log(){}},Response,TypeError,URLSearchParams,Proxy});
vm.runInContext(src+`
globalThis.P={open,prodChart,bboxParam,calNote,prodEstimate,scrNote,line,mapBox,show,scrAnalyze,launch,SCR,
  cur:()=>CURRENT,setCAL:c=>{CAL=c;}};`,ctx);
const P=ctx.P, out={};
const site=(nom,o={})=>Object.assign({nom,lat:1,lon:2,rayon_m:1000,analyse:false,source_coordonnees:'x',
  production_emprise:null,statut_analyse:null},o);
async function check(name,fn){ try{ const r=await fn(); out[name]=r===true?true:String(r); }
  catch(e){ out[name]='exception: '+(e&&e.stack||e); } }

(async()=>{
  // init : noms hostiles dans les infobulles
  take('/api/catalog/points').ok([['a',1,2,0,'<img src=x onerror=alert(1)>']]);
  take('/api/catalog/stats').ok({}); take('/api/radar').ok([]); take('/api/calibration').ok(CALJSON);
  await flush(); take('/api/sites').ok([{id:'w',lat:0,lon:0,nom:'Mine <Nord> & "Sud"'}]); await flush();
  await check('tooltips_escaped',
    ()=>tooltips.length===2&&!tooltips.some(t=>t.includes('<'))||JSON.stringify(tooltips));

  await check('open_stale_response_ignored',async()=>{
    mapCalls.length=0; P.open('A'); P.open('B');
    take('site=B').ok(site('Mine B',{lat:1,lon:2})); await flush();
    take('site=A').ok(site('Mine A',{lat:50,lon:60})); await flush();
    const d=$el('#detail').innerHTML;
    return (d.includes('Mine B')&&!d.includes('Mine A')&&P.cur()==='B'&&!mapCalls.some(c=>c[0]===50))
      ||`detail=${d.slice(0,80)} cur=${P.cur()} map=${JSON.stringify(mapCalls)}`;
  });

  await check('open_error_rendered',async()=>{
    P.open('fp_unknown'); take('site=fp_unknown').status(404); await flush();
    const d=$el('#detail').innerHTML; return d.includes('introuvable (404)')||d;
  });

  await check('open_switches_to_map_view',async()=>{
    $el('#view-map').style.display='none'; P.open('C'); take('site=C').ok(site('Mine C')); await flush();
    return $el('#view-map').style.display!=='none'||'carte toujours cachée';
  });

  await check('prodchart_skips_replaced_canvas',async()=>{
    const n=charts.length, cv={isConnected:false,insertAdjacentHTML(){throw new Error('h3 inséré');}};
    P.prodChart('A','',cv); take('/api/production?site=A').ok([{serie:'charbon',year:2019,value_tonnes:2e6}]);
    await flush(); if(charts.length!==n) return 'graphique dessiné dans un canevas remplacé';
    const live={isConnected:true,insertAdjacentHTML(){}};
    P.prodChart('B','',live);
    take('/api/production?site=B').ok([{serie:'charbon',year:2019,value_tonnes:2e6}]); await flush();
    return (charts.length===n+1&&charts.at(-1).el===live)||'pas de graphique pour la fiche courante';
  });

  await check('search_out_of_order_and_stale_enter',async()=>{
    const cmd=$el('#cmd'), res=$el('#results'), before=P.cur();
    cmd.value='esc'; cmd.listeners.input({target:cmd}); runTimers();
    cmd.value='escondida'; cmd.listeners.input({target:cmd}); runTimers();
    take('q=escondida').ok([{id:'e1',nom:'Escondida'}]); await flush();
    take('q=esc').ok([{id:'x',nom:'Escape Mine'}]); await flush();
    if(!res.innerHTML.includes('Escondida')||res.innerHTML.includes('Escape'))
      return 'réponse tardive affichée : '+res.innerHTML;
    cmd.value='e'; cmd.listeners.input({target:cmd});
    if(res.innerHTML!==''||res.style.display!=='none') return 'liste non vidée sous 2 caractères';
    cmd.listeners.keydown({key:'Enter'});
    return P.cur()===before||'Entrée a ouvert un ancien résultat';
  });

  await check('watch_stops_when_job_lost',async()=>{
    P.open('D'); take('site=D').ok(site('Mine D',{statut_analyse:'en cours (optique)'})); await flush();
    const go=$el('#go'); go.disabled=true; runTimers(); take('site=D').ok(site('Mine D')); await flush();
    const left=[...timers.values()].filter(t=>t.rep).length;
    return (!go.disabled&&go.textContent==='Réessayer'&&/interrompue/.test($el('#jst').textContent)&&left===0)
      ||`disabled=${go.disabled} text=${go.textContent} jst=${$el('#jst').textContent} intervals=${left}`;
  });

  await check('launch_starts_jobs_poll_and_recovers_on_error',async()=>{
    const go=$el('#go'); go.isConnected=true;
    P.launch('E'); take('/api/analyze?site=E').ok({site:'E',statut:'en file'}); await flush();
    if(!fetches.some(f=>!f.done&&f.url==='/api/jobs')) return 'jobsPoll non relancé';
    P.launch('F'); take('/api/analyze?site=F').status(500); await flush();
    return (!go.disabled&&/Lancement impossible/.test($el('#jst').textContent))||'bouton bloqué';
  });

  await check('bbox_wrapped',()=>{
    const r=[P.bboxParam(165,55,200,75),P.bboxParam(-190,0,-170,10),P.bboxParam(-200,0,170,10),
      P.bboxParam(10,0,20,10),P.bboxParam(170,0,180,10),P.bboxParam(180,0,200,10)];
    const want=['165.000,55.000,-160.000,75.000','170.000,0.000,-170.000,10.000',
      '-180.000,0.000,180.000,10.000','10.000,0.000,20.000,10.000','170.000,0.000,180.000,10.000','-180.000,0.000,-160.000,10.000'];
    return JSON.stringify(r)===JSON.stringify(want)||JSON.stringify(r);
  });

  await check('bbox_kept_while_map_hidden',()=>{
    container.offsetWidth=800; Object.assign(box,{w:-75,s:-30,e:-65,n:-20}); const seen=P.mapBox();
    container.offsetWidth=0; Object.assign(box,{w:0,s:0,e:0,n:0});
    return P.mapBox()===seen||'zone dégénérée lue sur une carte cachée';
  });

  await check('screener_reloads_bbox_on_return',async()=>{
    P.SCR.loaded=true; P.SCR.shown='ancien'; $el('#f_bbox').checked=true;
    P.show('screener'); const f=fetches.find(f=>!f.done&&f.url.startsWith('/api/screener?'));
    if(!f) return 'table non rechargée'; f.done=true; f.ok({total:3,rows:[]}); await flush();
    $el('#f_bbox').checked=false; return P.SCR.shown.includes('bbox=')||P.SCR.shown;
  });

  await check('batch_message_uses_deja_en_cours',async()=>{
    P.SCR.shown='pays=Chile'; P.SCR.total=40; const msg=$el('#scr_msg');
    P.scrAnalyze(); const f=take('/api/screener/analyze?pays=Chile');
    f.ok({mises_en_file:[],deja_en_cours:['a','b'],en_file:2}); await flush();
    const m=msg.textContent; if(!/déjà en file ou en cours/.test(m)||/mises en file/.test(m)) return m;
    P.scrAnalyze(); take('/api/screener/analyze?').ok({mises_en_file:[{site:'c',statut:'en file'},
      {site:'d',statut:'en file'},{site:'e',statut:'en file'}],deja_en_cours:['a','b'],en_file:5});
    await flush();
    return /^3 mines mises en file.*2 déjà en file ou en cours/.test(msg.textContent)||msg.textContent;
  });

  await check('tooltip_shows_raw_z',()=>{
    const c=P.line({},['a','b'],[1,2],{a:[0,3.2],z:[29.6,3.2]},false);
    const cb=c.cfg.options.plugins.tooltip.callbacks, r=c.cfg.data.datasets[0].pointRadius;
    const t0=cb.afterLabel({dataIndex:0}), t1=cb.afterLabel({dataIndex:1});
    return (t0.includes('29.6')&&/isolé/.test(t0)&&t1==='z = 3.2'&&r[0]===0&&r[1]===4)
      ||`${t0} | ${t1} | ${r}`;
  });

  const clean=s=>!/undefined|NaN|null|\[object/.test(s);
  await check('calnote_wording',()=>{
    P.setCAL(CALJSON); const t=P.calNote();
    return (t.includes('annuelle de plus de 10 % de la surface nue')
      &&/pas<\/b> été confrontées à la production/.test(t)
      &&/taille de l'emprise/.test(t)&&/Maus/.test(t)&&/faiblement/.test(t)&&clean(t))||t;
  });
  await check('calnote_tolerates_renamed_keys',()=>{
    P.setCAL({mines:12}); const t=P.calNote(); P.setCAL({}); const e=P.calNote(); P.setCAL(CALJSON);
    return (t.includes('12 mines')&&clean(t)&&e==='')||t;
  });
  await check('scr_note_per_model',()=>{
    P.setCAL(CALJSON); const t=P.scrNote({total:100,avec_prod_estimee:10}), m=CALJSON.modele_emprise||{};
    const f=v=>Number(v).toLocaleString('fr-FR',{maximumFractionDigits:1});
    const want=[m.minerai?.facteur_90,m.charbon?.facteur_90].filter(v=>v!=null).map(v=>'× '+f(v));
    P.setCAL({}); const e=P.scrNote({total:1,avec_prod_estimee:0}); P.setCAL(CALJSON);
    return (want.every(w=>t.includes(w))&&/usines/.test(t)&&/sans commodité/.test(t)&&clean(t)&&clean(e))||t;
  });
  await check('prod_estimate_from_footprint_only',()=>{
    const f={type:'minerai extrait',tonnes_an:2e7,bas_90:2e6,haut_90:2e8,facteur_typique:2.5,facteur_90:8.4,
      emprise_km2:12,mines_calibration:218};
    const t=P.prodEstimate({production_emprise:f}), old=P.prodEstimate({production_estimee:f});
    return (t.includes('× 2,5')&&t.includes('× 8,4')&&t.includes('2 Mt')&&t.includes('200 Mt')
      &&old===''&&clean(t))||t;
  });
  process.stdout.write(JSON.stringify(out));
})();
"""

CHECKS = ["tooltips_escaped", "open_stale_response_ignored", "open_error_rendered",
          "open_switches_to_map_view", "prodchart_skips_replaced_canvas",
          "search_out_of_order_and_stale_enter", "watch_stops_when_job_lost",
          "launch_starts_jobs_poll_and_recovers_on_error", "bbox_wrapped", "bbox_kept_while_map_hidden",
          "screener_reloads_bbox_on_return", "batch_message_uses_deja_en_cours", "tooltip_shows_raw_z",
          "calnote_wording", "calnote_tolerates_renamed_keys", "scr_note_per_model",
          "prod_estimate_from_footprint_only"]


@pytest.fixture(scope="module")
def node_results(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node absent : comportements de la page non rejoués")
    h = tmp_path_factory.mktemp("front") / "harness.js"
    h.write_text(HARNESS, "utf-8")
    p = subprocess.run([node, str(h), str(HTML), str(CAL)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


@pytest.mark.parametrize("name", CHECKS)
def test_page_behaviour(node_results, name):
    assert node_results.get(name) is True, node_results.get(name)
