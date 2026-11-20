// Run with: node --test tests/test_training_ui_groups_web.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.dataset={}; this.value=''; this.classList={add(){},remove(){},toggle(){}}; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=children; }
  addEventListener() {}
  dispatchEvent() {}
  querySelector(selector) { return this.children.find(el=>selector===`[value=${el.value}]`); }
  querySelectorAll(tag) { return this.children.flatMap(el=>[...((el.tag===tag||(tag==='input:checked'&&el.tag==='input'&&el.checked))?[el]:[]),...el.querySelectorAll(tag)]); }
}
function setup() {
  const elements=new Map(['groups','groups-result','sampling-mode','dataset-candidates','apply-datasets','wandb-mode','auto-norm','allow-busy'].map(id=>[id,new Element('div')]));
  elements.get('sampling-mode').value='weighted_replacement';
  const context=vm.createContext({URLSearchParams,Event:class {},location:{hash:''},document:{
    getElementById:id=>elements.get(id), createElement:tag=>new Element(tag), querySelectorAll:()=>[],
  }});
  const source=fs.readFileSync(path.join(__dirname,'../scripts/training_ui_web/app.js'),'utf8');
  // Load the actual form logic; startup/polling bindings are exercised by the browser.
  vm.runInContext(source.slice(0,source.indexOf("$('preview').addEventListener")),context);
  vm.runInContext(`baseline='example'; fields.set('dataset_root',{input:{value:'/data'}});
    updatePath=()=>{}; notice=()=>{};
    addGroup({name:'A',weight:7,episodes:[99]}); addGroup({name:'B',weight:3,episodes:[]});`,context);
  return {context,elements,run:code=>vm.runInContext(code,context)};
}
const detected={episodes:4,frames:40,groups:[
  {name:'A',episodes:[0,1],frames:20,source_root:'/old/A-subtask'},
  {name:'B',episodes:[4],frames:10,source_root:'/old/B-subtask'},
  {name:'C',episodes:[8],frames:10,source_root:'/old/C'},
]};

test('auto-fill keeps weights, defaults new groups, and compacts episode ranges',async()=>{
  const {context,run,elements}=setup();
  context.result=detected;
  run('api=async()=>result');
  await run('detectGroups()');
  assert.deepEqual(JSON.parse(run('JSON.stringify(groupInputs())')), [
    {name:'A',weight:7,episodes:'0-1'}, {name:'B',weight:3,episodes:'4'}, {name:'C',weight:1,episodes:'8'},
  ]);
  assert.match(elements.get('groups-result').textContent,/4 条 episode/);
  assert.equal(run("JSON.stringify(parseIds(formatIds([5,2,1,0,5])))"),'[0,1,2,5]');
});

test('failed detection preserves existing groups',async()=>{
  const {run}=setup(); const before=run('JSON.stringify(groupInputs())');
  run("api=async()=>{throw Error('missing provenance')}");
  await assert.rejects(run('detectGroups()'),/missing provenance/);
  assert.equal(run('JSON.stringify(groupInputs())'),before);
});

test('loading an existing group preserves its episode order',()=>{
  const {run}=setup();
  run("$('groups').replaceChildren(); addGroup({name:'existing',weight:1,episodes:[5,0,2]})");
  assert.equal(run('groupInputs()[0].episodes'),'5,0,2');
});

for(const mutation of ["fields.get('dataset_root').input.value='/new'",'loadingVersion++',"$('sampling-mode').value='epoch_shuffle'","paperTrial='t3'","addGroup({name:'manual',weight:2,episodes:[9]})"]){
  test('in-flight result cannot overwrite changed input: '+mutation,async()=>{
    const {context,run}=setup(); context.result=detected;
    run('let finish; api=()=>new Promise(resolve=>{finish=resolve})');
    const pending=run('detectGroups()'); run(mutation);
    const changed=run('JSON.stringify(groupInputs())'); run('finish(result)');
    await assert.rejects(pending,/请重新读取/);
    assert.equal(run('JSON.stringify(groupInputs())'),changed);
  });
}

test('target percentages normalize relative weights and reject zero weights',()=>{
  const {run}=setup();
  assert.equal(run("JSON.stringify(groupShares([{weight:7},{weight:3}]))"),'[0.7,0.3]');
  assert.equal(run("JSON.stringify(groupShares([{weight:7},{weight:0}]))"),'[]');
});

test('stop from a task row targets that job, including download/preparation stages',async()=>{
  const {context,run}=setup(); context.calls=[];context.confirm=()=>true;
  run(`currentJob='another-job';
    api=async(path,data)=>{calls.push([path,data]);return path.startsWith('/api/job?')?{status:'running',stage:'下载权重',exp_name:'selected'}:{message:'已请求停止'};};
    refreshJobs=async()=>{};showJob=async id=>{calls.push(['shown',id]);};`);
  await run("stopTask('selected-job')");
  assert.equal(context.calls[1][0],'/api/stop');
  assert.equal(context.calls[1][1].id,'selected-job');
  assert.equal(context.calls[2][1],'selected-job');
});

test('stop cancellation and completed jobs do not send stop requests',async()=>{
  const {context,run}=setup();context.calls=[];context.confirm=()=>false;
  run("api=async(path,data)=>{calls.push(path);return {status:'running',exp_name:'test'};}");
  await run("stopTask('id')");
  assert.equal(context.calls.length,1);
  run("api=async()=>({status:'completed'})");
  await assert.rejects(run("stopTask('id')"),/已结束/);
});

test('parent discovery keeps full selected and subtasks unchecked, then applies only checked roots',async()=>{
  const {context,run,elements}=setup();
  context.discovery={datasets:[
    {root:'/0924/A',episodes:106,frames:10,fps:30,selected:true},
    {root:'/1010/full',episodes:191,frames:20,fps:30,selected:true},
    {root:'/1010/subtasks',episodes:760,frames:20,fps:30,selected:false,note:'alternative'},
  ],warnings:[]};
  run("api=async()=>discovery;fields.get('dataset_root').input.dispatchEvent=()=>{};fields.set('sources',{input:{value:'',dispatchEvent(){}}});independentStats=()=>{};");
  await run('discoverDatasets()');
  assert.equal(elements.get('dataset-candidates').querySelectorAll('input:checked').length,2);
  run('applyDatasets()');
  assert.equal(run("fields.get('dataset_root').input.value"),'/0924/A\n/1010/full');
  assert.equal(run("fields.get('sources').input.value"),'/0924/A\n/1010/full');
  assert.equal(elements.get('groups').children.length,0);
});

test('changed directory cannot apply stale discovery, and browser can append multiple paths',async()=>{
  const {context,run}=setup();context.discovery={datasets:[],warnings:[]};
  run("api=async()=>discovery;initial={dataset_root:'/default'};");
  assert.equal(run("appendDatasetPath('/default','/0924')"),'/0924');
  assert.equal(run("appendDatasetPath('/0924','/1010')"),'/0924\n/1010');
  await run('discoverDatasets()');
  run("fields.get('dataset_root').input.value='/changed'");
  assert.throws(()=>run('applyDatasets()'),/目录已修改/);
});

test('launch request carries every selected root in order while one root keeps legacy shape',()=>{
  const {run}=setup();
  run("$('sampling-mode').value='epoch_shuffle';fields.get('dataset_root').input.value='/a\\n/b';fields.set('owner',{input:{value:'tester'},type:'text'});");
  assert.equal(run('JSON.stringify(request().values.dataset_roots)'),'["/a","/b"]');
  assert.equal(run('request().values.dataset_root'),'/a');
  run("fields.get('dataset_root').input.value='/a'");
  assert.equal(run('request().values.dataset_roots'),undefined);
});

test('T3 on the ordinary form preserves selected roots, RTC and group weights',()=>{
  const {elements,run}=setup();
  for(const id of ['paper-fields','paper-timestamp','paper-correlated','paper-summary','discover-datasets','new-stats','detect-groups','group-source-help']){
    elements.set(id,new Element('div'));
  }
  run(`initial={dataset_root:'/original',action_semantics:'absolute',action_horizon:50,paper_trial:'none'};
    protocol={paper_available:true};renderProtocol=()=>{};
    fields.get('dataset_root').input.value='/different location\\n/other/source';
    independentStats=()=>{};
    for(const [key,value] of Object.entries({exp_name:'experiment_absolute',action_horizon:'50',rtc_max_delay:'10',action_semantics:'absolute'})){
      fields.set(key,{input:Object.assign(document.createElement('input'),{value})});
    }
    fields.get('action_semantics').input.append(Object.assign(document.createElement('option'),{value:'absolute'}));`);
  const before=run('JSON.stringify(groupInputs())');
  elements.get('paper-timestamp').checked=true;
  elements.get('paper-correlated').checked=true;
  run('changePaperFeatures()');
  assert.equal(run('paperTrial'),'t3');
  assert.equal(run("fields.get('action_semantics').input.value"),'delta');
  assert.equal(run("fields.get('rtc_max_delay').input.value"),'10');
  assert.equal(run("fields.get('dataset_root').input.value"),'/different location\n/other/source');
  assert.equal(run('JSON.stringify(groupInputs())'),before);
  assert.equal(elements.get('sampling-mode').value,'weighted_replacement');
});
