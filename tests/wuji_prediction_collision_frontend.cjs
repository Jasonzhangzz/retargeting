const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('moxian_web/static/index.html','utf8'),ids=new Map();
for(const tag of html.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)){
 assert(!ids.has(tag[1]),`duplicate id ${tag[1]}`);
 ids.set(tag[1],{checked:/\schecked(?:\s|>)/.test(tag[0]),hidden:/\shidden(?:\s|>)/.test(tag[0]),listeners:{},
  addEventListener(k,fn){this.listeners[k]=fn;}});
}
assert(!ids.get('wujiRVQOverlay').checked);assert(!ids.get('wujiShowGraspType').checked);
assert(ids.get('wujiRVQ').hidden);assert(ids.get('wujiGraspPreview').hidden);
const context=vm.createContext({document:{getElementById:id=>{assert(ids.has(id),id);return ids.get(id);}},window:{addEventListener(){}}});
vm.runInContext(fs.readFileSync('moxian_web/static/wuji-view.js','utf8').replace(/^import .*;\n/gm,''),context);
vm.runInContext(`
 function link(owner){return {userData:{owner},material:{color:{value:'#aeb8b7',set(color){this.value=color;}}}};}
 model={link_names:['right_palm_link','right_finger1_link4','right_finger2_link4','right_finger5_link4']};
 robot=[link(0),link(1),link(2),link(3)];rvqRobot=[{visible:true}];
 syncPredictionVisibility();
`,context);
assert.equal(vm.runInContext('rvqRobot[0].visible',context),false);
// Independent toggles: disabling grasp type cannot be undone by frame updates.
ids.get('wujiShowGraspType').checked=true;vm.runInContext('syncPredictionVisibility()',context);
assert(!ids.get('wujiRVQ').hidden);assert(!ids.get('wujiGraspPreview').hidden);
ids.get('wujiShowGraspType').checked=false;vm.runInContext('syncPredictionVisibility()',context);
assert(ids.get('wujiRVQ').hidden);assert(ids.get('wujiGraspPreview').hidden);
context.report={query_valid:true,pair_depths_mm:{thumb_link4__index_link4:1.2,palm__pinky_link4:.5}};
vm.runInContext('updateSelfCollisionColors(report,true)',context);
assert.equal(vm.runInContext('robot[1].material.color.value',context),'#ed4545');
assert.equal(vm.runInContext('robot[2].material.color.value',context),'#ed4545');
assert.equal(vm.runInContext('robot[0].material.color.value',context),'#aeb8b7');
assert.equal(vm.runInContext('robot[3].material.color.value',context),'#aeb8b7');
vm.runInContext('updateSelfCollisionColors(report,false)',context);
assert.equal(vm.runInContext('robot.every(m=>m.material.color.value==="#aeb8b7")',context),true);
context.report={query_valid:false,pair_depths_mm:{thumb_link4__index_link4:5}};
vm.runInContext('updateSelfCollisionColors(report,true)',context);
assert.equal(vm.runInContext('robot.every(m=>m.material.color.value==="#aeb8b7")',context),true);
context.report={query_valid:true,pair_depths_mm:{}};
vm.runInContext('updateSelfCollisionColors(report,true)',context);
assert.equal(vm.runInContext('robot.every(m=>m.material.color.value==="#aeb8b7")',context),true);
assert.equal(vm.runInContext('collisionLinkName("left_finger5_link4")',context),'pinky_link4');
// Direct neighbours, self pairs, palm/base attachments, and unknown pairs
// cannot color a link even if an old/malformed report contains deep values.
for(const pair of ['thumb_link3__thumb_link4','index_link1__index_link2','palm__pinky_link1','thumb_link4__thumb_link4','bad__thumb_link4']){
 context.report={query_valid:true,pair_depths_mm:{[pair]:9}};
 assert.equal(vm.runInContext('selfCollisionLinks(report,true).size',context),0,pair);
}
context.report={query_valid:true,pair_depths_mm:{thumb_link2__thumb_link4:2}};
assert.equal(vm.runInContext('selfCollisionLinks(report,true).size',context),2);
console.log('Prediction and collision UI: defaults, independent toggles, pair name mapping, thresholds, release and invalid/stale clearing passed');
