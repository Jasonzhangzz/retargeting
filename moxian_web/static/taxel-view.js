import * as THREE from 'three';

export const TAXEL_PURPLE='#897493',TAXEL_YELLOW='#ffe15b';
export function createTaxels(){
 const c=document.createElement('canvas');c.width=c.height=32;const ctx=c.getContext('2d');ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(16,16,14,0,Math.PI*2);ctx.fill();
 const points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:.002,vertexColors:true,map:new THREE.CanvasTexture(c),transparent:true,opacity:.85,alphaTest:.4,depthTest:true,depthWrite:false}));
 points.frustumCulled=false;points.renderOrder=2;return points;
}

// Evaluate fixed material coordinates on the displayed triangles. No nearest
// point search, contact-dependent projection, or rigid wrist approximation.
export function updateTaxels(points,vertices,topology,contact,basis=p=>p){
 const faces=topology.sid_face_ids,bary=topology.sid_barycentric;
 if(!faces||!bary){points.visible=false;return;}
 const n=faces.length,g=points.geometry;
 if(g.attributes.position?.count!==n){g.setAttribute('position',new THREE.Float32BufferAttribute(n*3,3));g.setAttribute('color',new THREE.Float32BufferAttribute(n*3,3));}
 const purple=new THREE.Color(TAXEL_PURPLE),yellow=new THREE.Color(TAXEL_YELLOW),p=g.attributes.position,c=g.attributes.color;
 let active=0;
 for(let sid=0;sid<n;sid++){
  const ids=topology.faces[faces[sid]],bc=bary[sid],v=[0,0,0];
  for(let k=0;k<3;k++)for(let d=0;d<3;d++)v[d]+=bc[k]*(ArrayBuffer.isView(vertices)?vertices[ids[k]*3+d]:vertices[ids[k]][d]);
  p.setXYZ(sid,...basis(v));const on=contact?.ready===true&&contact.cell_contact?.[sid]===true;
  if(on)active++;const color=on?yellow:purple;c.setXYZ(sid,color.r,color.g,color.b);
 }
 p.needsUpdate=c.needsUpdate=true;g.computeBoundingSphere();points.userData={count:n,active,source:contact?.source||'sensor',ready:contact?.ready===true};
}
