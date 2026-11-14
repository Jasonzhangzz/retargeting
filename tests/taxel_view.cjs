const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function fixture(){
 class Attribute{constructor(size,items){this.count=size/items;this.values=[];}setXYZ(i,...v){this.values[i]=v;}}
 class Geometry{constructor(){this.attributes={};}setAttribute(k,v){this.attributes[k]=v;}computeBoundingSphere(){}}
 class Color{constructor(value){this.r=value==='#ffe15b'?1:.5;this.g=.8;this.b=.2;}}
 class Matrix{makeTranslation(...position){this.position=position;return this;}}
 class Points{constructor(g,m){this.geometry=g;this.material=m;this.userData={};}add(child){}}
 class Balls{constructor(g,m,count){this.material=m;this.radius=g.radius;this.count=count;this.positions=[];this.instanceMatrix={setUsage(){}};}setMatrixAt(i,m){this.positions[i]=m.position.slice();}}
 class Sphere{constructor(radius){this.radius=radius;}}
 const context=vm.createContext({document:{createElement:()=>({getContext:()=>({beginPath(){},arc(){},fill(){}})})},THREE:{Points,BufferGeometry:Geometry,Float32BufferAttribute:Attribute,PointsMaterial:class{constructor(options){Object.assign(this,options);}},Color,CanvasTexture:class{},SphereGeometry:Sphere,MeshBasicMaterial:class{constructor(options){Object.assign(this,options);}},InstancedMesh:Balls,DynamicDrawUsage:1,Matrix4:Matrix},ArrayBuffer,Float32Array});
 vm.runInContext(fs.readFileSync('moxian_web/static/taxel-view.js','utf8').replace(/^import .*;\n/gm,'').replaceAll('export ',''),context);
 return context;
}

test('only actual active SIDs create bright balls on the same deforming material surface',()=>{
 const c=fixture();vm.runInContext(`
  const points=createTaxels();const top={faces:[[0,1,2]],sid_face_ids:[0,0],sid_barycentric:[[.5,.25,.25],[0,.5,.5]]};
  let vertices=new Float32Array([0,0,0,1,0,0,0,1,0]);
  const update=contact=>updateTaxels(points,vertices,top,contact,p=>[p[1],p[0],-p[2]]);
  update({ready:true,cell_contact:[false,true]});
  globalThis.initial={count:points.contactSpheres.count,position:points.contactSpheres.positions[0],materialPoint:points.geometry.attributes.position.values[1],radius:points.contactSpheres.radius,depthWrite:points.contactSpheres.material.depthWrite};
  vertices=new Float32Array([0,0,.2,1,0,.2,0,1,.2]);update({ready:true,cell_contact:[false,true]});globalThis.moved=points.contactSpheres.positions[0];
  update({ready:false,cell_contact:[true,true]});globalThis.unknown=points.contactSpheres.count;
 `,c);
 assert.equal(c.initial.count,1);assert.deepEqual(Array.from(c.initial.materialPoint),[.5,.5,-0]);
 assert.deepEqual(Array.from(c.initial.position),[.5,.5,-.001]);assert.ok(Math.abs(c.moved[2]+.201)<1e-7);
 assert.ok(c.initial.radius>=.0015);assert.equal(c.initial.depthWrite,false);assert.equal(c.unknown,0);
});

test('calibrated normal offsets move material points with the displayed surface',()=>{
 const c=fixture();vm.runInContext(`
 const points=createTaxels(),top={faces:[[0,1,2]],sid_face_ids:[0],sid_barycentric:[[.5,.25,.25]],sid_normal_offset_m:[.002]};
 updateTaxels(points,new Float32Array([0,0,0,1,0,0,0,1,0]),top,{ready:true,cell_contact:[true]});
 globalThis.offsetPoint=points.geometry.attributes.position.values[0];globalThis.offsetBall=points.contactSpheres.positions[0];
 `,c);
 assert.deepEqual(Array.from(c.offsetPoint),[.25,.25,.002]);assert.deepEqual(Array.from(c.offsetBall),[.25,.25,.003]);
});
