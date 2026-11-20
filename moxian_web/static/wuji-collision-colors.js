/* Pure collision-to-color projection, shared by the renderer and its tests. */
export const COLORS={clear:'#b7c1c4',both:'#ed555b',meshOnly:'#58bffa',neuralOnly:'#ffb743'};
export const basis=p=>[p[1],p[0],-p[2]];
export const linkLabel=name=>name==='palm'?'掌部':name.replace(/^(thumb|index|middle|ring|pinky)_link(\d)$/,(_,f,n)=>`${({thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',pinky:'小指'})[f]} L${n}`);
export function visualState(data,reference='matched',pair=null){
 const threshold=data.threshold_mm??.5,hasNeural=data.neural?.query_valid===true,left=Array(21).fill(COLORS.clear),right=[...left],ld=Array(21).fill(0),rd=[...ld];
 const lp=Array(21).fill(0),rp=[...lp],selected=(a,b)=>!pair||(a===pair[0]&&b===pair[1])||(a===pair[1]&&b===pair[0]);
 for(const p of data.pairs||[]){
  if(!selected(p.a,p.b))continue;
  const real=reference==='dense'?p.dense_mm:p.matched_mm,pred=p.neural_mm,hit=real>threshold,nhit=Number.isFinite(pred)&&pred>threshold;
  for(const i of [p.a,p.b]){
   ld[i]=Math.max(ld[i],real);rd[i]=Math.max(rd[i],pred||0);
   const a=hit?(nhit||!hasNeural?1:2):0,b=nhit?(hit?1:2):0;
   if(a>lp[i]){lp[i]=a;left[i]=nhit||!hasNeural?COLORS.both:COLORS.meshOnly;}
   if(b>rp[i]){rp[i]=b;right[i]=hit?COLORS.both:COLORS.neuralOnly;}
  }
 }
 const w=data.visualization?.witnesses,points=[[],[]],pointColors=[[],[]];
 if(w)w.positions_m.forEach((_,i)=>{
  if(!selected(w.source_links[i],w.target_links[i]))return;
  const hit=w.mesh_sdf_mm[i]<-threshold,nhit=w.neural_sdf_mm?.[i]<-threshold;
  if(hit){points[0].push(i);pointColors[0].push(nhit||!hasNeural?COLORS.both:COLORS.meshOnly);}
  if(nhit){points[1].push(i);pointColors[1].push(hit?COLORS.both:COLORS.neuralOnly);}
 });
 return {colors:[left,right],depths:[ld,rd],points,pointColors};
}
