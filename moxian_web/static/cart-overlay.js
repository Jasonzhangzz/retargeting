import * as THREE from 'three';

const overlays=new WeakMap();
function positions(geometry,values){
 if(geometry.attributes.position?.count!==values.length/3){
  geometry.dispose();geometry.setAttribute('position',new THREE.Float32BufferAttribute(values,3));
 }else{geometry.attributes.position.array.set(values);geometry.attributes.position.needsUpdate=true;}
 geometry.computeBoundingSphere();
}
window.updateCartHullOverlay=(scene,contact,map)=>{
 let view=overlays.get(scene);
 const visible=window.cartHullVisible!==false&&!!contact?.contact_points_m?.length;
 if(!view&&!visible)return;
 if(!view){
  const group=new THREE.Group();
  const yellow=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:0xffdc25,size:.0035,depthTest:false}));
  const blue=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:0x268cff,size:.0009,depthTest:false,transparent:true,opacity:.85}));
  yellow.renderOrder=10;blue.renderOrder=9;group.add(blue,yellow);scene.add(group);view={group,yellow,blue};overlays.set(scene,view);
 }
 view.group.visible=visible;if(!visible)return;
 const key=[contact.timestamp_ms,contact.geometry_version,contact.contact_points_m.length,contact.patch_points_m?.length].join(':');
 if(view.key===key)return;view.key=key;
 positions(view.yellow.geometry,contact.contact_points_m.flatMap(map));
 positions(view.blue.geometry,(contact.patch_points_m||[]).flatMap(map));
};
