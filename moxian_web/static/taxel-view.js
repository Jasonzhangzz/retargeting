import * as THREE from 'three';

export const TAXEL_PURPLE='#897493',TAXEL_YELLOW='#ffe15b';
export function createTaxels(){
 const c=document.createElement('canvas');c.width=c.height=32;const ctx=c.getContext('2d');ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(16,16,14,0,Math.PI*2);ctx.fill();
 const points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:.0014,vertexColors:true,map:new THREE.CanvasTexture(c),transparent:true,opacity:.38,alphaTest:.4,depthTest:true,depthWrite:false}));
 points.frustumCulled=false;points.renderOrder=2;
 // Small unlit spheres remain bright at every camera angle. Only measured
 // active SIDs instantiate a sphere; inactive grid points remain unobtrusive.
 const balls=new THREE.InstancedMesh(new THREE.SphereGeometry(.0018,10,7),new THREE.MeshBasicMaterial({color:TAXEL_YELLOW,depthTest:true,depthWrite:false,toneMapped:false}),460);
 balls.count=0;balls.frustumCulled=false;balls.renderOrder=6;balls.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
 points.contactSpheres=balls;points.add(balls);return points;
}

// Evaluate fixed material coordinates on the displayed triangles. No nearest
// point search, contact-dependent projection, or rigid wrist approximation.
export function updateTaxels(points,vertices,topology,contact,basis=p=>p){
 const faces=topology.sid_face_ids,bary=topology.sid_barycentric;
 if(!faces||!bary){points.visible=false;return;}
 const n=faces.length,g=points.geometry;
 if(g.attributes.position?.count!==n){g.setAttribute('position',new THREE.Float32BufferAttribute(n*3,3));g.setAttribute('color',new THREE.Float32BufferAttribute(n*3,3));}
 const purple=new THREE.Color(TAXEL_PURPLE),yellow=new THREE.Color(TAXEL_YELLOW),p=g.attributes.position,c=g.attributes.color;
 let active=0;const balls=points.contactSpheres,matrix=balls?new THREE.Matrix4():null;
 for(let sid=0;sid<n;sid++){
  const ids=topology.faces[faces[sid]],bc=bary[sid],v=[0,0,0];
  for(let k=0;k<3;k++)for(let d=0;d<3;d++)v[d]+=bc[k]*(ArrayBuffer.isView(vertices)?vertices[ids[k]*3+d]:vertices[ids[k]][d]);
  const offset=topology.sid_normal_offset_m?.[sid]||0;
  if(offset){
   const a=ids.map(id=>ArrayBuffer.isView(vertices)?Array.from(vertices.slice(id*3,id*3+3)):vertices[id]);
   const u=a[1].map((x,d)=>x-a[0][d]),w=a[2].map((x,d)=>x-a[0][d]);
   const n=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]],length=Math.hypot(...n)||1;
   for(let d=0;d<3;d++)v[d]+=offset*n[d]/length;
  }
  p.setXYZ(sid,...basis(v));const on=contact?.ready===true&&contact.cell_contact?.[sid]===true;
  if(on){
   if(balls){
    const a=ids.map(id=>ArrayBuffer.isView(vertices)?Array.from(vertices.slice(id*3,id*3+3)):vertices[id]);
    const u=a[1].map((x,d)=>x-a[0][d]),w=a[2].map((x,d)=>x-a[0][d]);
    const normal=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]],length=Math.hypot(...normal)||1;
    const center=basis(v.map((x,d)=>x+.001*normal[d]/length));
    matrix.makeTranslation(...center);balls.setMatrixAt(active,matrix);
   }
   active++;
  }
  const color=on?yellow:purple;c.setXYZ(sid,color.r,color.g,color.b);
 }
 if(balls){balls.count=active;balls.instanceMatrix.needsUpdate=true;}
 p.needsUpdate=c.needsUpdate=true;g.computeBoundingSphere();points.userData={count:n,active,source:contact?.source||'sensor',ready:contact?.ready===true};
}
