const assert=require('node:assert/strict'),fs=require('node:fs');
(async()=>{
 const source=fs.readFileSync('moxian_web/static/wuji-collision-colors.js','utf8');
 const {visualState,COLORS,basis}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
 assert.deepEqual(basis([1,2,3]),[2,1,-3]);
 const d={threshold_mm:.5,neural:{query_valid:true},pairs:[
  {a:1,b:8,dense_mm:2,matched_mm:2,neural_mm:2},
  {a:2,b:9,dense_mm:2,matched_mm:2,neural_mm:.1},
  {a:3,b:10,dense_mm:0,matched_mm:0,neural_mm:2}],
  visualization:{witnesses:{positions_m:[[0,0,0],[1,1,1],[2,2,2]],source_links:[1,2,3],target_links:[8,9,10],mesh_sdf_mm:[-2,-2,1],neural_sdf_mm:[-2,1,-2]}}};
 const p=visualState(d);
 assert.equal(p.colors[0][1],COLORS.both);assert.equal(p.colors[1][8],COLORS.both);
 assert.equal(p.colors[0][2],COLORS.meshOnly);assert.equal(p.colors[1][9],COLORS.clear);
 assert.equal(p.colors[0][3],COLORS.clear);assert.equal(p.colors[1][10],COLORS.neuralOnly);
 assert.deepEqual(p.points,[[0,1],[0,2]]);
 const filtered=visualState(d,'matched',[9,2]);
 assert.deepEqual(filtered.points,[[1],[]]);assert.equal(filtered.colors[0][1],COLORS.clear);
 const dense=visualState({...d,pairs:[{a:1,b:8,dense_mm:2,matched_mm:0,neural_mm:0}]},'dense');
 assert.equal(dense.colors[0][1],COLORS.meshOnly);
 console.log('Collision 3D colors: matched hits, missed hits, false positives, pair filters and coordinate basis passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
