import * as THREE from 'three';

const overlays=new WeakMap();
function positions(geometry,values){
 if(geometry.attributes.position?.count!==values.length/3){
  geometry.dispose();geometry.deleteAttribute('normal');geometry.setAttribute('position',new THREE.Float32BufferAttribute(values,3));
 }else{
  geometry.attributes.position.array.set(values);geometry.attributes.position.needsUpdate=true;
 }
 geometry.computeBoundingSphere();
}

window.updateCartHullOverlay=(scene,hull,map)=>{
 let view=overlays.get(scene);
 const visible=window.cartHullVisible!==false&&!!(hull?.ready||hull?.contact_points_m?.length);
 if(!view&&!visible)return;
 if(!view){
  const group=new THREE.Group();
  const fill=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial({color:0x26c6a5,transparent:true,opacity:.22,depthWrite:false,side:THREE.DoubleSide,roughness:.65}));
  const edges=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x087f70,transparent:true,opacity:.9,depthTest:false}));
  const tips=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:0x0a8f79,size:.003,depthTest:false}));
  const contacts=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x3cdbb0,depthTest:false,transparent:true,opacity:.8}));
  edges.renderOrder=8;tips.renderOrder=9;contacts.renderOrder=8;group.add(fill,edges,tips,contacts);scene.add(group);
  view={group,fill,edges,tips,contacts};overlays.set(scene,view);
 }
 view.group.visible=visible;if(!visible)return;
 const key=[hull.timestamp_ms,hull.geometry_version,hull.dimension,hull.preview_only,window.cartHullVisible].join(':');
 if(view.key===key)return;view.key=key;
 const vertices=hull.ready?hull.vertices_m||[]:[],faces=hull.ready?hull.faces||[]:[];
 positions(view.fill.geometry,faces.flatMap(face=>face.flatMap(i=>map(vertices[i]))));
 view.fill.geometry.computeVertexNormals();view.fill.visible=faces.length>0;
 positions(view.edges.geometry,(hull.ready?hull.edges||[]:[]).flatMap(edge=>edge.flatMap(i=>map(vertices[i]))));
 positions(view.tips.geometry,(hull.contact_points_m||hull.finger_points_m||vertices).flatMap(map));
 positions(view.contacts.geometry,hull.contact_points_m?.length?hull.contact_points_m.flatMap(map):(hull.pairs||[]).flatMap(pair=>(pair.points_m||[]).flatMap(map)));
 view.fill.material.color?.setHex(hull.preview_only?0xe5a13a:0x26c6a5);
 view.edges.material.color?.setHex(hull.preview_only?0xe5a13a:0x087f70);
};
