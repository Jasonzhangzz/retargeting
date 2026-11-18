// Cubic point-normal triangles (Vlachos et al., I3D 2001).
// https://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
// The 778 controls and their FK are untouched. Fixed material coordinates
// also remap taxels and mesh picking to this actual tessellated surface.
import * as THREE from 'three';

export class CurvedSurface {
 constructor(topology,segments=4){
  this.source=topology;this.segments=segments;this.recipes=[];this.faces=[];this.material=[];this.parent=[];
  if(topology.surface_policy==='official_MANO_triangles'||topology.surface_policy==='qnbot_original_triangles'){
   this.linear=true;this.faces=topology.faces;this.topology=topology;this.positions=new Float32Array(topology.rest_vertices_m.length*3);this.controls=new THREE.BufferGeometry();return;
  }
  const lookup=new Map(),children=[];
  const vertex=(face,bc)=>{
   const ids=topology.faces[face],nz=bc.map((v,i)=>v>0?i:-1).filter(i=>i>=0);let key;
   if(nz.length===1)key='v'+ids[nz[0]];
   else if(nz.length===2){const a=nz[0],b=nz[1],lo=Math.min(ids[a],ids[b]),hi=Math.max(ids[a],ids[b]);key=`e${lo}_${hi}_${bc[ids[a]===hi?a:b]}`;}
   else key=`f${face}_${bc.join('_')}`;
   if(!lookup.has(key)){
    const b=bc.map(x=>x/segments),[u,v,w]=b;
    lookup.set(key,this.recipes.length);this.recipes.push({face,bary:b,coeff:[u**3,v**3,w**3,3*u*u*v,3*u*v*v,3*u*u*w,3*u*w*w,3*v*v*w,3*v*w*w,6*u*v*w]});
   }
   return lookup.get(key);
  };
  topology.faces.forEach((_,f)=>{
   const grid=new Map();for(let i=0;i<=segments;i++)for(let j=0;j<=segments-i;j++)grid.set(`${i},${j}`,{id:vertex(f,[segments-i-j,i,j]),bc:[1-(i+j)/segments,i/segments,j/segments]});
   children[f]=[];
   const triangle=keys=>{const nodes=keys.map(([i,j])=>grid.get(`${i},${j}`));children[f].push(this.faces.length);this.faces.push(nodes.map(n=>n.id));this.material.push(nodes.map(n=>n.bc));this.parent.push(f);};
   for(let i=0;i<segments;i++)for(let j=0;j<segments-i;j++){
    triangle([[i,j],[i+1,j],[i,j+1]]);
    if(i+j<segments-1)triangle([[i+1,j],[i+1,j+1],[i,j+1]]);
   }
  });
  const sidFaces=[],sidBary=[];
  topology.sid_face_ids.forEach((face,sid)=>{
   const p=topology.sid_barycentric[sid];let found=false;
   for(const child of children[face]){
    const [a,b,c]=this.material[child],x=p[1]-a[1],y=p[2]-a[2],ab=[b[1]-a[1],b[2]-a[2]],ac=[c[1]-a[1],c[2]-a[2]],det=ab[0]*ac[1]-ab[1]*ac[0];
    const v=(x*ac[1]-y*ac[0])/det,w=(ab[0]*y-ab[1]*x)/det,bc=[1-v-w,v,w];
    if(Math.min(...bc)>=-1e-6){const clipped=bc.map(x=>Math.max(x,0)),sum=clipped.reduce((a,b)=>a+b,0);sidFaces.push(child);sidBary.push(clipped.map(x=>x/sum));found=true;break;}
   }
   if(!found)throw Error('SID 无法绑定细分曲面 '+sid);
  });
  this.topology={...topology,faces:this.faces,sid_face_ids:sidFaces,sid_barycentric:sidBary};
  this.controls=new THREE.BufferGeometry();this.controls.setIndex(topology.faces.flat());this.controls.setAttribute('position',new THREE.Float32BufferAttribute(778*3,3));
  this.positions=new Float32Array(this.recipes.length*3);this.controlPoints=new Float64Array(topology.faces.length*30);
 }
 dispose(){this.controls.dispose();}
 update(vertices){
  if(this.linear){this.positions.set(vertices.flat());this.stats={vertices:vertices.length,triangles:this.faces.length,maxCurvatureM:0};return this.positions;}
  const began=performance.now(),control=this.controls.attributes.position;
  vertices.forEach((p,i)=>control.setXYZ(i,...p));control.needsUpdate=true;this.controls.computeVertexNormals();
  const normals=this.controls.attributes.normal.array,points=this.controlPoints;
  this.source.faces.forEach((ids,f)=>{
   const p=ids.map(i=>vertices[i]),n=ids.map(i=>Array.from(normals.slice(i*3,i*3+3))),base=f*30;
   for(let k=0;k<3;k++)for(let d=0;d<3;d++)points[base+k*3+d]=p[k][d];
   [[0,1],[1,0],[0,2],[2,0],[1,2],[2,1]].forEach(([a,b],k)=>{
    const projection=p[b].reduce((s,v,d)=>s+(v-p[a][d])*n[a][d],0);
    for(let d=0;d<3;d++)points[base+(3+k)*3+d]=(2*p[a][d]+p[b][d]-projection*n[a][d])/3;
   });
   for(let d=0;d<3;d++){
    let edge=0;for(let k=3;k<9;k++)edge+=points[base+k*3+d]/6;
    points[base+27+d]=edge*1.5-(p[0][d]+p[1][d]+p[2][d])/6;
   }
  });
  let max=0;
  this.recipes.forEach((recipe,i)=>{
   const ids=this.source.faces[recipe.face],base=recipe.face*30,out=[0,0,0],linear=[0,0,0];
   for(let d=0;d<3;d++){
    for(let k=0;k<10;k++)out[d]+=recipe.coeff[k]*points[base+k*3+d];
    for(let k=0;k<3;k++)linear[d]+=recipe.bary[k]*vertices[ids[k]][d];
   }
   const distance=Math.hypot(...out.map((v,d)=>v-linear[d]));max=Math.max(max,distance);
   // Bound interpolation overshoot at severely bent knuckles. This cap is
   // shared by edge vertices, so neighboring patches stay watertight.
   const amount=distance>.0015?.0015/distance:1;
   for(let d=0;d<3;d++)this.positions[i*3+d]=linear[d]+amount*(out[d]-linear[d]);
  });
  this.stats={vertices:this.recipes.length,triangles:this.faces.length,maxRawCurvatureM:max,maxCurvatureM:Math.min(max,.0015),processingMs:performance.now()-began};
  return this.positions;
 }
 mapHit(face,bary){
  if(this.linear)return {face,barycentric:bary};
  return {face:this.parent[face],barycentric:[0,1,2].map(d=>bary.reduce((s,v,k)=>s+v*this.material[face][k][d],0))};
 }
}
