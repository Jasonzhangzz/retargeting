"""Coverage diagnostics in the original normalized 20-D joint space."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def metrics(q, labels, lower, upper):
    q=np.asarray(q,float);labels=np.asarray(labels)
    if not len(q):return dict(samples=0)
    x=(q-np.asarray(lower))/(np.asarray(upper)-np.asarray(lower))
    hist=np.stack([np.histogram(x[:,j],bins=20,range=(0,1))[0] for j in range(20)])
    probability=hist/len(x)
    entropy=-(probability*np.log(np.maximum(probability,1e-30))).sum(1)/np.log(20)
    if len(x)<2:return dict(samples=len(x),marginal_entropy=entropy.tolist())
    tree=cKDTree(x);distance,index=tree.query(x,k=min(11,len(x)),workers=1)
    # An epsilon-neighbour graph diagnoses sampled connectivity. It does NOT
    # certify feasible interpolation or prove that the full manifold connects.
    rows=np.repeat(np.arange(len(x)),index.shape[1]-1)
    cols=index[:,1:].ravel();d=distance[:,1:].ravel()
    valid=(np.max(np.abs(x[rows]-x[cols]),axis=1)<=.1)&(d<=.3)
    graph=coo_matrix((np.ones(valid.sum()),(rows[valid],cols[valid])),shape=(len(x),len(x))).tocsr()
    components,comp=connected_components(graph,directed=False)
    _,sizes=np.unique(comp,return_counts=True)
    nearest=distance[:,1]
    classes=np.unique(labels);adj={str(int(a)):{str(int(b)):int(((labels[rows[valid]]==a)&(labels[cols[valid]]==b)).sum())
                                                for b in classes} for a in classes}
    return dict(samples=len(x),marginal_bins=hist.tolist(),marginal_entropy=entropy.tolist(),
                mean_marginal_entropy=float(entropy.mean()),nearest_distance_mean=float(nearest.mean()),
                nearest_distance_cv=float(nearest.std()/max(nearest.mean(),1e-12)),
                sampled_graph_components=int(components),largest_component_fraction=float(sizes.max()/len(x)),
                cross_type_neighbour_counts=adj,graph_linf_threshold=.1,graph_l2_threshold=.3,
                continuous_feasibility_proven=False,
                interpretation='finite normalized 20-D sample diagnostics; not a full reachable-space coverage percentage')


class Pool:
    """Deterministic bounded ring pools; density is shared across grasp labels."""
    def __init__(self, lower, upper, capacity=256):
        self.lower=np.asarray(lower);self.width=np.asarray(upper)-self.lower
        self.capacity=capacity;self.data={}

    def add(self,name,poses,rows,gids):
        entries=self.data.setdefault(name,[])
        entries.extend(dict(pose=np.asarray(q).tolist(),row=int(r),geometry=str(g)) for q,r,g in zip(poses,rows,gids))
        del entries[:-self.capacity]

    def weights(self,name):
        own=self.data.get(name,[])
        if not own:return np.empty(0)
        all_rows=[p['pose'][6:] for entries in self.data.values() for p in entries]
        x=(np.asarray(all_rows)-self.lower)/self.width
        targets=(np.asarray([p['pose'][6:] for p in own])-self.lower)/self.width
        k=min(9,len(x));d=cKDTree(x).query(targets,k=k,workers=1)[0]
        sparsity=d[:,-1] if k>1 else np.ones(len(own))
        weights=np.maximum(sparsity,.005)
        weights=np.minimum(weights,10*np.median(weights))
        return weights/weights.sum()
