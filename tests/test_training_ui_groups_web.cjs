// Run with: node --test tests/test_training_ui_groups_web.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.dataset={}; this.value=''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=children; }
  addEventListener() {}
  querySelectorAll(tag) { return this.children.flatMap(el=>[...(el.tag===tag?[el]:[]),...el.querySelectorAll(tag)]); }
}
function setup() {
  const elements=new Map(['groups','groups-result','sampling-mode'].map(id=>[id,new Element('div')]));
  elements.get('sampling-mode').value='weighted_replacement';
  const context=vm.createContext({URLSearchParams,location:{hash:''},document:{
    getElementById:id=>elements.get(id), createElement:tag=>new Element(tag),
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

for(const mutation of ["fields.get('dataset_root').input.value='/new'",'loadingVersion++',"$('sampling-mode').value='epoch_shuffle'","addGroup({name:'manual',weight:2,episodes:[9]})"]){
  test('in-flight result cannot overwrite changed input: '+mutation,async()=>{
    const {context,run}=setup(); context.result=detected;
    run('let finish; api=()=>new Promise(resolve=>{finish=resolve})');
    const pending=run('detectGroups()'); run(mutation);
    const changed=run('JSON.stringify(groupInputs())'); run('finish(result)');
    await assert.rejects(pending,/请重新读取/);
    assert.equal(run('JSON.stringify(groupInputs())'),changed);
  });
}
