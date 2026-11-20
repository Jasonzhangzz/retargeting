"""The learned CARD field's precision denominator matches its pad-only queries."""
import torch
from moxian_web.cart_objective import CartHand
from moxian_web.cart_collision import CollisionModel,SDFLearner
from moxian_web.cart_card_neural_surface import NeuralCardCostObjective

def test_nearest_validation_uses_only_policy_pad_mode_and_preserves_rng():
    torch.set_num_threads(1);h=CartHand('cpu');c=CollisionModel(h);NeuralCardCostObjective(h,c)
    learner=SDFLearner(c,capacity=128,batch_size=16);original=c.model.contact;observed=[]
    def contact(x,d,links,palmar):
        observed.append((d,links,palmar));return original(x,d,links,palmar)
    c.model.contact=contact
    points,links=c.oracle.random_queries(32);state=torch.get_rng_state().clone()
    report=learner.validate(points,links)
    assert report['contact_samples']==32
    d,l,p=observed[-1]
    assert not d.any() and p.all() and set(l.tolist())<=set(learner.teacher.pad_owners)
    # Lazy teacher initialization is deterministic and consumes no torch RNG.
    assert torch.equal(state,torch.get_rng_state())
