const {test}=require('node:test');
const assert=require('node:assert/strict');
require('../moxian_web/static/motion-guide.js');
const guide=globalThis.MoxianMotionGuide;
test('static guidance shows one held target and separate spread/together endpoints',()=>{
 for(const endpoint of ['spread','together']){
  const html=guide.staticMarkup('motion_spread_together',endpoint,'right');
  assert.equal((html.match(/<figure/g)||[]).length,1);assert.match(html,/20帧稳定后自动采集/);
 }
 assert.notEqual(guide.staticMarkup('motion_spread_together','spread','right'),guide.staticMarkup('motion_spread_together','together','right'));
});
test('all motion stages have actual hand diagrams, phase highlights and accessible names',()=>{
 const stages=['range_thumb','range_index','range_middle','range_ring','range_pinky','thumb_opposition','pinch_index','pinch_middle','pinch_ring','pinch_pinky','open_repeat','together_repeat'];
 for(const stage of stages)for(const side of ['left','right'])for(const [phase] of guide.sequence(stage)){
  const html=guide.markup(stage,phase,side);
  assert.equal((html.match(/class="motion-step active"/g)||[]).length,1);
  assert.equal((html.match(/<svg /g)||[]).length,guide.sequence(stage).length);
  assert.match(html,/role="img" aria-label=/);
  assert.equal(html.includes('scale(-1,1)'),side==='left');
 }
});
test('pinches show separate approach, contact and release rather than a single static hold',()=>{
 assert.deepEqual(guide.sequence('pinch_index').map(x=>x[0]),['apart','approach','touch','release']);
 assert.notEqual(guide.illustration('pinch_index','approach','right'),guide.illustration('pinch_index','touch','right'));
});
test('every pinch phase highlights the exact two region IDs from the training contract',()=>{
 for(const finger of ['index','middle','ring','pinky'])for(const side of ['left','right'])for(const [phase] of guide.sequence('pinch_'+finger)){
  const html=guide.illustration('pinch_'+finger,phase,side);
  assert.match(html,/data-target-patch="thumb_distal"/);
  assert.ok(html.includes('data-target-patch="'+finger+'_distal"'));
  assert.equal((html.match(/data-target-patch=/g)||[]).length,2);
 }
 assert.match(guide.markup('range_index','extend','right'),/无指定接触目标/);
 assert.match(guide.illustration('together_repeat','hold','right'),/data-target-region="adjacent_finger_side"/);
 assert.equal((guide.overview('right').match(/<figure /g)||[]).length,17);
});

test('extended spread exercise shows straight, spread and together for both hands',()=>{
 assert.deepEqual(guide.sequence('motion_spread_together').map(x=>x[0]),['extend','spread','together','spread']);
 for(const side of ['left','right']){
  const spread=guide.illustration('motion_spread_together','spread',side),together=guide.illustration('motion_spread_together','together',side);
  assert.notEqual(spread,together);assert.equal(spread.includes('scale(-1,1)'),side==='left');
  assert.equal((spread.match(/data-target-patch=/g)||[]).length,0);
 }
});

test('v4 diagrams share every endpoint, ignore thumb for fist/fold, and do not prescribe speed or hold time',()=>{
 const definition=require('../moxian_web/static/calibration-contact-targets.json');
 for(const stage of definition.protocols.contact_motion_v3.stages)for(const side of ['left','right'])for(const [phase] of guide.sequence(stage)){
  const html=guide.illustration(stage,phase,side);
  for(const region of definition.stages[stage].patches)assert.ok(html.includes('data-target-patch="'+region+'"'),stage+region);
  assert.equal(html.includes('scale(-1,1)'),side==='left');
  if(['contact_fist','contact_fold'].includes(stage))assert.match(html,/data-finger="thumb" data-observation-weight="0"/);
 }
 assert.match(guide.markup('contact_pair_index','touch','right'),/不规定速度或保持时长/);
 assert.match(guide.trajectoryMarkup({trajectory:[{t_ms:0,gap_m:.03},{t_ms:20,gap_m:.002},{t_ms:40,gap_m:.03}],contact_events:[{timestamp_ms:20,gap_m:.002,repetition:1}]}),/data-contact-event="1"/);
});
