/* Native slide/hinge FK, evaluated once for a selected pose; no render cache. */
const mul=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,x,k)=>s+x*b[k][j],0)));
const apply=(r,p)=>r.map(row=>row.reduce((s,x,i)=>s+x*p[i],0));
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const transpose=r=>r[0].map((_,i)=>r.map(row=>row[i]));
const eye=()=>[[1,0,0],[0,1,0],[0,0,1]];
function quaternion([w,x,y,z]){
 return [[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
  [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
  [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]];
}
export function workspaceFrames(spec,joints){
 if(!spec||joints?.length!==20||!joints.every(Number.isFinite))throw Error('缺少有效参考pose或机器人模型');
 const q=[0,0,0,0,0,0,...joints],p=[[0,0,0]],r=[eye()];
 for(let body=1;body<spec.parents.length;body++){
  let local=spec.positions[body].slice(),rotation=quaternion(spec.quaternions_wxyz[body]);
  for(const j of spec.body_joints[body]){
   const angle=q[spec.addresses[j]]-spec.reference[spec.addresses[j]],axis=spec.axes[j];
   if(spec.kinds[j]===2)local=add(local,apply(rotation,axis).map(x=>x*angle));
   else if(spec.kinds[j]===3){
    const [a,b,c]=axis,cross=[[0,-c,b],[c,0,-a],[-b,a,0]],square=mul(cross,cross);
    const turn=eye().map((row,i)=>row.map((v,k)=>v+Math.sin(angle)*cross[i][k]+(1-Math.cos(angle))*square[i][k]));
    const pivot=spec.pivots[j],turned=apply(turn,pivot);
    local=add(local,apply(rotation,pivot.map((v,i)=>v-turned[i])));rotation=mul(rotation,turn);
   }else throw Error('不支持的参考关节类型');
  }
  const parent=spec.parents[body];p.push(add(p[parent],apply(r[parent],local)));r.push(mul(r[parent],rotation));
 }
 const palm=p[spec.links[0]],basis=transpose(spec.basis),canonical=v=>apply(basis,v.map((x,i)=>x-palm[i]));
 return {link_positions_m:spec.links.map(i=>canonical(p[i])),link_rotations:spec.links.map(i=>mul(basis,r[i])),
  keypoints_m:spec.tip_bodies.map((body,i)=>canonical(add(p[body],apply(r[body],spec.tip_positions[i]))))};
}
