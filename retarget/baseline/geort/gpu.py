"""CUDA execution of native FK and Open3D legacy voxel centroid semantics.

No changes to the upstream networks or objectives. Voxel iteration order differs
from Open3D's implementation-specific unordered map; centroid sets agree.
"""
import numpy as np


class RobotFK:
    def __init__(self, robot, device='cuda:0'):
        import torch
        from retarget.env.grasp.pose_gpu import ModelFK
        self.robot=robot;self.device=device
        self.fk=ModelFK(robot.model,device)
        self.bodies=torch.as_tensor(robot.model.site_bodyid[robot.tip_ids],device=device,dtype=torch.long)
        self.local=torch.as_tensor(robot.model.site_pos[robot.tip_ids],device=device,dtype=torch.float32)
        self.basis=torch.as_tensor(robot.basis,device=device,dtype=torch.float32)

    def __call__(self,q):
        import torch
        q=torch.as_tensor(q,device=self.device,dtype=torch.float32)
        position,rotation=self.fk.frames(torch.cat((q.new_zeros((*q.shape[:-1],6)),q),-1))
        tips=position[...,self.bodies,:]+(rotation[...,self.bodies,:,:]@self.local[...,None]).squeeze(-1)
        return (tips-position[...,self.robot.body_ids[0],None,:])@self.basis


def voxel_centroids(points, voxel_size=.001, device='cuda:0'):
    import torch
    # Open3D legacy uses double precision and min_bound - half voxel.
    p=torch.as_tensor(points,device=device,dtype=torch.float64)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not torch.isfinite(p).all():
        raise ValueError('无效的体素点云')
    origin=p.amin(0)-voxel_size*.5
    keys=torch.floor((p-origin)/voxel_size).to(torch.int64)
    unique,inverse,counts=torch.unique(keys,dim=0,sorted=True,return_inverse=True,return_counts=True)
    sums=p.new_zeros((len(unique),3));sums.index_add_(0,inverse,p)
    return sums/counts[:,None]


def human_dataset(points,resample_to=50000,device='cuda:0'):
    import torch
    from retarget.baseline.geort.training import reference
    clouds=[]
    for finger in np.asarray(points).transpose(1,0,2):
        cloud=voxel_centroids(finger,device=device)
        if len(cloud)<2:raise ValueError('每个手指至少需要两个不同的1 mm体素，请录制完整活动范围')
        # Keep upstream independent finger sampling, including its exclusive n-1.
        indices=np.random.randint(0,len(cloud)-1,resample_to)
        clouds.append(cloud[torch.as_tensor(indices,device=device)].float().cpu().numpy())
    return reference('dataset').MultiPointDataset(np.stack(clouds))


def nearest_chamfer(input_points,target_points,target_counts=None):
    """Same squared bidirectional nearest-point objective, small backward graph.

    Closest indices are discrete. Find them without recording the quadratic
    distance matrix, then differentiate the exact gathered point differences.
    No sqrt, matrix-product distance approximation or change of loss weights.
    Official mode continues to call the pinned upstream implementation.
    """
    import torch
    with torch.no_grad():
        distances=(input_points[:,:,None]-target_points[:,None]).square().sum(-1)
        forward=distances.argmin(-1);reverse=distances.argmin(-2)
    a=target_points.gather(1,forward[...,None].expand(-1,-1,3))
    b=input_points.gather(1,reverse[...,None].expand(-1,-1,3))
    forward_loss=(input_points-a).square().sum(-1).mean()
    reverse=(target_points-b).square().sum(-1)
    if target_counts is None:return forward_loss+reverse.mean()
    weights=target_counts.to(device=reverse.device,dtype=reverse.dtype)
    return forward_loss+(reverse*weights).sum(-1).div(weights.sum()).mean()
