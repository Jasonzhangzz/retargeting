"""Device-resident Lambda contact-set sampling and analytic batched force LM.

Like trigrasp's Lambda optimizer, each GPU thread solves one complete contact
combination. Sampling, surface projection, force solves and selection stay on
Warp. The NumPy solver is retained as an independent numerical reference.
"""
from functools import lru_cache
import numpy as np
import warp as wp
from retarget.contact_projection import closest_triangle
from retarget.spider_pose import quat_log, quat_exp, right_jacobian, inverse_right_jacobian

Vec6=wp.spatial_vector
Mat6=wp.types.matrix(shape=(6,6),dtype=wp.float32)


@lru_cache(None)
def kernels(contacts):
    NC=wp.constant(int(contacts));NF=wp.constant(3*int(contacts))
    Force=wp.types.vector(length=3*contacts,dtype=wp.float32)
    Jac=wp.types.matrix(shape=(6,3*contacts),dtype=wp.float32)

    @wp.struct
    class Physics:
        free: Vec6
        goal: Vec6
        support_row: Vec6
        inverse_row: Vec6
        mobility: float
        gap: float
        external_norm: float

    @wp.struct
    class Job:
        original: wp.array(dtype=wp.vec3)
        current: wp.array(dtype=wp.vec3)
        hand: wp.array(dtype=wp.vec3)
        dual: wp.array(dtype=wp.vec3)
        weights: wp.array(dtype=float)
        friction: wp.array(dtype=wp.vec2)
        rotation: wp.array(dtype=wp.mat33)
        position: wp.array(dtype=wp.vec3)
        velocity: wp.array(dtype=Vec6)
        goal: wp.array(dtype=Vec6)
        templates: wp.array(dtype=wp.vec3)
        starts: wp.array(dtype=int)
        counts: wp.array(dtype=int)
        anchors: wp.array(dtype=wp.vec3)
        anchor_normals: wp.array(dtype=wp.vec3)
        triangles: wp.array2d(dtype=wp.vec3)
        face_normals: wp.array(dtype=wp.vec3)
        support_vertices: wp.array(dtype=wp.vec3)
        points: wp.array(dtype=wp.vec3)
        normals: wp.array(dtype=wp.vec3)
        forces: wp.array2d(dtype=float)
        costs: wp.array(dtype=float)
        motion_errors: wp.array(dtype=float)
        search: wp.array(dtype=int)
        selected: wp.array(dtype=wp.vec3)
        statistics: wp.array2d(dtype=float)
        physics: wp.array(dtype=Physics)
        inverse: Mat6
        gravity: wp.vec3
        com: wp.vec3
        floor_point: wp.vec3
        floor_normal: wp.vec3
        mass: float
        dt: float
        motion_tolerance: float
        min_improvement: float
        radius: float
        object_radius: float
        rho: float
        wrench_weight: float
        samples: int
        worlds: int
        iterations: int
        seed: int
        floor_enabled: int
        sweep: int
        spider_pose: int
        pose_weights: Vec6

    @wp.func
    def basis(outward: wp.vec3):
        n=-wp.normalize(outward)
        axis=wp.vec3(1.,0.,0.)
        least=wp.abs(n[0])
        if wp.abs(n[1]) < least:
            axis=wp.vec3(0.,1.,0.);least=wp.abs(n[1])
        if wp.abs(n[2]) < least:
            axis=wp.vec3(0.,0.,1.)
        t=wp.normalize(wp.cross(n,axis))
        return n,t,wp.cross(n,t)

    @wp.kernel(enable_backward=False)
    def prepare(job: Job):
        group=wp.tid()
        p=Physics()
        r=job.rotation[group]
        f=job.mass*(wp.transpose(r)@job.gravity)
        torque=wp.cross(job.com,f)
        external=Vec6(f[0],f[1],f[2],torque[0],torque[1],torque[2])
        p.free=job.velocity[group]+job.dt*(job.inverse*external)
        p.goal=job.goal[group]
        p.external_norm=wp.length(f)
        p.support_row=Vec6(0.);p.inverse_row=Vec6(0.)
        p.gap=0.;p.mobility=1.
        if job.floor_enabled == 1:
            normal=wp.transpose(r)@job.floor_normal
            minimum=float(1.e30);point=wp.vec3(0.)
            for i in range(job.support_vertices.shape[0]):
                vertex=job.support_vertices[i]
                height=wp.dot(vertex,normal)
                if height < minimum:
                    minimum=height;point=wp.vec3(vertex)
            angular=wp.cross(point,normal)
            p.support_row=Vec6(normal[0],normal[1],normal[2],angular[0],angular[1],angular[2])
            p.inverse_row=job.inverse*p.support_row
            p.mobility=wp.max(wp.dot(p.inverse_row,p.support_row),1.e-15)
            p.gap=minimum+wp.dot(job.position[group]-job.floor_point,job.floor_normal)
        job.physics[group]=p

    @wp.kernel(enable_backward=False)
    def sample_project(job: Job, incumbent_only: int):
        row=wp.tid()
        slot=row%NC
        candidate=row//NC
        group=candidate//job.samples
        sample=candidate%job.samples
        if (incumbent_only == 1 and sample != 0) or (incumbent_only == 0 and (sample == 0 or job.search[group] == 0)):
            return
        index=group*NC+slot
        weight=job.weights[(group//job.worlds)*NC+slot]
        point=wp.vec3(job.current[index])
        if sample == 1:
            point=(job.rho*(job.hand[index]+job.dual[index])+.1*job.templates[index])/(job.rho+.1)
        elif sample > 1:
            state=wp.rand_init(job.seed,sample*NC+slot)
            direction=wp.vec3(wp.randn(state),wp.randn(state),wp.randn(state))
            point+=.5*job.radius*wp.randf(state)*wp.normalize(direction)
        best=wp.vec3(job.anchors[index]);normal=wp.vec3(job.anchor_normals[index])
        distance=float(1.e30)
        if weight > 0.:
            for offset in range(job.counts[index]):
                face=job.starts[index]+offset
                value=closest_triangle(point,job.triangles[face,0],job.triangles[face,1],job.triangles[face,2])
                if wp.length_sq(value-job.templates[index]) <= job.radius*job.radius+1.e-12:
                    d=wp.length_sq(value-point)
                    if d < distance:
                        distance=d;best=value;normal=wp.vec3(job.face_normals[face])
        job.points[row]=best
        job.normals[row]=normal

    @wp.func
    def project_force(job: Job, force: Force, group: int):
        out=Force(0.)
        for c in range(NC):
            index=group*NC+c
            if job.weights[(group//job.worlds)*NC+c] > 0. and job.counts[index] > 0:
                mu=job.friction[index]
                limit=10./wp.sqrt(1.+wp.max(mu[0],mu[1])*wp.max(mu[0],mu[1]))
                original_normal=force[3*c]
                t1=wp.where(mu[0]>0.,force[3*c+1],0.)
                t2=wp.where(mu[1]>0.,force[3*c+2],0.)
                norm=wp.sqrt(wp.pow(t1/wp.max(mu[0],1.e-15),2.)+wp.pow(t2/wp.max(mu[1],1.e-15),2.))
                normal=wp.max(original_normal,0.)
                if norm > original_normal:
                    normal=wp.max(.5*(norm+original_normal),0.)
                fn=wp.min(normal,limit)
                ratio=wp.min(1.,normal/wp.max(norm,1.e-15))*wp.min(1.,fn/wp.max(normal,1.e-15))
                out[3*c]=fn;out[3*c+1]=t1*ratio;out[3*c+2]=t2*ratio
        return out

    @wp.func
    def evaluate(job: Job, p: Physics, matrix: Jac, force: Force):
        v=p.free+job.dt*(job.inverse*(matrix*force))
        correction=-(.2*(job.dt*wp.dot(p.support_row,v)+p.gap)+.001*wp.dot(p.support_row,v))/p.mobility
        mobility=Mat6(0.)
        for i in range(6):
            mobility[i,i]=1.
        if p.gap < 0. and correction > 0.:
            v+=p.inverse_row*correction
            for i in range(6):
                for j in range(6):
                    mobility[i,j]-=(.2*job.dt+.001)*p.inverse_row[i]*p.support_row[j]/p.mobility
        omega=wp.vec3(v[3],v[4],v[5])
        length=wp.length(omega)
        safe=wp.max(length,1.e-10)
        a=.5*job.dt
        factor=2.*wp.atan(a*safe)/safe
        df=(2.*a/(1.+a*a*safe*safe)-factor)/(safe*safe)
        if length < 1.e-5:
            df=-job.dt*job.dt*job.dt/6.
        residual=Vec6(0.);pose_jac=Mat6(0.);physical=float(0.)
        scale=wp.sqrt(job.wrench_weight)*job.object_radius/job.radius
        for i in range(3):
            physical+=wp.pow(job.dt*v[i]-p.goal[i],2.)+.3*job.object_radius*job.object_radius*wp.pow(factor*omega[i]-p.goal[i+3],2.)
            residual[i]=scale*(job.dt*v[i]-p.goal[i])/job.object_radius
            residual[i+3]=scale*wp.sqrt(.3)*(factor*omega[i]-p.goal[i+3])
            pose_jac[i,i]=scale*job.dt/job.object_radius
            for j in range(3):
                element=df*omega[i]*omega[j]
                if i == j:
                    element+=factor
                pose_jac[i+3,j+3]=scale*wp.sqrt(.3)*element
        if job.spider_pose!=0:
            # Only the task residual changes. The force LM, friction
            # projection, candidate sampling and selection remain shared.
            phi=factor*omega
            goal=wp.vec3(p.goal[3],p.goal[4],p.goal[5])
            error=quat_log(wp.quat_inverse(quat_exp(goal))*quat_exp(phi))
            log_jac=inverse_right_jacobian(error)*right_jacobian(phi)
            increment=wp.mat33(0.)
            for i in range(3):
                for j in range(3):
                    element=df*omega[i]*omega[j]
                    if i==j:element+=factor
                    increment[i,j]=element
            rotation_jac=log_jac*increment
            physical=0.
            # Keep run.py's contact-radius normalization for LM conditioning;
            # candidate task ranking below uses the unscaled SPIDER pose norm.
            scale=wp.sqrt(job.wrench_weight)/job.radius
            for i in range(3):
                position=(job.dt*v[i]-p.goal[i])*job.pose_weights[i]
                rotation=error[i]*job.pose_weights[i+3]
                residual[i]=scale*position;residual[i+3]=scale*rotation
                physical+=position*position+rotation*rotation
                pose_jac[i,i]=scale*job.pose_weights[i]*job.dt
                for j in range(3):pose_jac[i+3,j+3]=scale*job.pose_weights[i+3]*rotation_jac[i,j]
        jacobian=pose_jac*mobility*job.inverse*matrix*job.dt
        return residual,jacobian,wp.sqrt(physical)

    @wp.func
    def solve6(matrix: Mat6, rhs: Vec6):
        lower=Mat6(0.);interim=Vec6(0.);answer=Vec6(0.)
        for i in range(6):
            for j in range(i+1):
                value=matrix[i,j]
                for k in range(j):
                    value-=lower[i,k]*lower[j,k]
                if i == j:
                    lower[i,j]=wp.sqrt(wp.max(value,1.e-20))
                else:
                    lower[i,j]=value/lower[j,j]
        for i in range(6):
            value=rhs[i]
            for j in range(i):
                value-=lower[i,j]*interim[j]
            interim[i]=value/lower[i,i]
        for reverse in range(6):
            i=5-reverse;value=interim[i]
            for j in range(i+1,6):
                value-=lower[j,i]*answer[j]
            answer[i]=value/lower[i,i]
        return answer

    @wp.kernel(enable_backward=False)
    def solve(job: Job, incumbent_only: int):
        candidate=wp.tid();group=candidate//job.samples
        sample=candidate%job.samples
        if (incumbent_only == 1 and sample != 0) or (incumbent_only == 0 and (sample == 0 or job.search[group] == 0)):
            return
        p=job.physics[group]
        matrix=Jac(0.);force=Force(0.)
        active=float(0.);total_weight=float(0.);geometry=float(0.)
        for c in range(NC):
            weight=job.weights[(group//job.worlds)*NC+c]
            total_weight+=weight
            if weight > 0. and job.counts[group*NC+c] > 0:
                active+=1.
                normal,t1,t2=basis(job.normals[candidate*NC+c])
                point=job.points[candidate*NC+c]
                for axis in range(3):
                    direction=normal
                    if axis == 1:
                        direction=t1
                    elif axis == 2:
                        direction=t2
                    moment=wp.cross(point,direction)
                    for k in range(3):
                        matrix[k,3*c+axis]=direction[k]
                        matrix[k+3,3*c+axis]=moment[k]
            geometric_error=job.points[candidate*NC+c]-job.hand[group*NC+c]-job.dual[group*NC+c]
            move=job.points[candidate*NC+c]-job.templates[group*NC+c]
            geometry+=weight*(.5*job.rho*wp.length_sq(geometric_error)+.05*wp.length_sq(move))
        geometry/=wp.max(total_weight,1.e-12)*job.radius*job.radius
        for c in range(NC):
            force[3*c]=p.external_norm/wp.max(active,1.)
        force=project_force(job,force,group)
        r,jac,error=evaluate(job,p,matrix,force)
        cost=wp.dot(r,r)+1.e-6*wp.dot(force,force)
        damping=float(.001)
        for iteration in range(job.iterations):
            g=wp.transpose(jac)*r+1.e-6*force
            mu=damping+1.e-6
            system=jac*wp.transpose(jac)
            for k in range(6):
                system[k,k]+=mu
            correction=solve6(system,jac*g)
            delta=-(g-wp.transpose(jac)*correction)/mu
            delta*=wp.min(1.,2./wp.max(wp.length(delta),1.e-15))
            proposal=project_force(job,force+delta,group)
            next_r,next_jac,next_error=evaluate(job,p,matrix,proposal)
            next_cost=wp.dot(next_r,next_r)+1.e-6*wp.dot(proposal,proposal)
            if wp.isfinite(next_cost) and next_cost < cost:
                force=proposal;r=next_r;jac=next_jac;cost=next_cost;error=next_error
                damping=wp.max(damping*.5,1.e-8)
            else:
                damping=wp.min(damping*4.,1.e6)
        job.motion_errors[candidate]=error
        job.costs[candidate]=cost+geometry
        for k in range(NF):
            job.forces[candidate,k]=force[k]

    @wp.kernel(enable_backward=False)
    def check(job: Job):
        group=wp.tid()
        error=job.motion_errors[group*job.samples]
        job.search[group]=wp.where(error > job.motion_tolerance,1,0)
        if job.sweep == 0:
            job.statistics[group,0]=job.costs[group*job.samples]
            job.statistics[group,7]=wp.where(error <= job.motion_tolerance,1.,0.)
            job.statistics[group,8]=0.
        if job.search[group] == 1:
            job.statistics[group,8]=1.

    @wp.kernel(enable_backward=False)
    def select(job: Job):
        group=wp.tid();chosen=int(0)
        baseline=group*job.samples
        best=job.costs[baseline];error=job.motion_errors[baseline]
        found_feasible=error <= job.motion_tolerance
        threshold=wp.max(1.e-9,job.min_improvement*error)
        if job.search[group] == 1:
            for sample in range(1,job.samples):
                value=job.costs[baseline+sample]
                candidate_error=job.motion_errors[baseline+sample]
                feasible=candidate_error <= job.motion_tolerance
                eligible=feasible or candidate_error < job.motion_errors[baseline]-threshold
                if wp.isfinite(value) and eligible:
                    better=(feasible and not found_feasible) or (feasible and found_feasible and value < best) or (not feasible and not found_feasible and candidate_error < error)
                    if better:
                        best=value;chosen=sample;error=candidate_error;found_feasible=feasible
        shift=float(0.);projection=float(0.);active=float(0.);frozen=float(0.)
        for c in range(NC):
            point=job.points[(group*job.samples+chosen)*NC+c]
            job.selected[group*NC+c]=point
            if job.weights[(group//job.worlds)*NC+c]>0.:
                active+=1.
                if job.counts[group*NC+c] == 0:
                    frozen+=1.
                shift=wp.max(shift,wp.length(point-job.templates[group*NC+c]))
                projection=wp.max(projection,wp.length(job.points[group*job.samples*NC+c]-job.current[group*NC+c]))
        job.statistics[group,9]=error*1000.
        job.statistics[group,10]=wp.where(error <= job.motion_tolerance,1.,0.)
        job.statistics[group,1]=best
        job.statistics[group,2]=float(chosen)
        job.statistics[group,3]=shift*1000.
        job.statistics[group,4]=active
        job.statistics[group,5]=frozen
        job.statistics[group,6]=projection*1000.

    return Job,Physics,prepare,sample_project,solve,check,select


class DeviceLambdaLM:
    def __init__(self, explorer, *, groups, contacts, worlds, physics, device):
        self.explorer,self.device,self.groups=explorer,device,groups
        Job,Physics,self.prepare_kernel,self.sample_kernel,self.solve_kernel,self.check_kernel,self.select_kernel=kernels(contacts)
        job=self.job=Job();self.contacts=contacts
        job.samples,job.worlds=explorer.samples,worlds
        job.iterations=explorer.lm_iterations
        job.mass,job.dt=explorer.mass,explorer.dt
        job.radius,job.rho=explorer.radius,explorer.rho
        job.motion_tolerance,job.min_improvement=explorer.motion_tolerance,explorer.min_improvement
        job.wrench_weight,job.seed=explorer.wrench_weight,explorer.seed
        job.spider_pose=int('spider_pose_weights' in physics)
        job.pose_weights=Vec6(*physics.get('spider_pose_weights',np.ones(6)))
        job.inverse=Mat6(*np.asarray(physics['inverse'],np.float32).ravel())
        job.com=wp.vec3(*physics['com']);job.gravity=wp.vec3(*physics['gravity'])
        job.floor_enabled=int(physics['floor'] is not None)
        if physics['floor'] is not None:
            job.floor_point=wp.vec3(*physics['floor'][0]);job.floor_normal=wp.vec3(*physics['floor'][1])
        job.support_vertices=wp.array(physics['support_vertices'],dtype=wp.vec3,device=device)
        size=groups*job.samples*contacts
        job.points=wp.empty(size,dtype=wp.vec3,device=device)
        job.normals=wp.empty(size,dtype=wp.vec3,device=device)
        job.costs=wp.empty(groups*job.samples,dtype=float,device=device)
        job.motion_errors=wp.empty(groups*job.samples,dtype=float,device=device)
        job.search=wp.empty(groups,dtype=int,device=device)
        job.forces=wp.empty((groups*job.samples,3*contacts),dtype=float,device=device)
        job.physics=wp.empty(groups,dtype=Physics,device=device)
        job.current=wp.empty(groups*contacts,dtype=wp.vec3,device=device)
        job.selected=wp.empty(groups*contacts,dtype=wp.vec3,device=device)
        job.statistics=wp.empty((groups,11),dtype=float,device=device)
        self.graph=None

    def bind(self,batch,template,weights,object_radius):
        job=self.job
        self.explorer.projector._prepare(template,weights)
        names=('templates','starts','counts','anchors','anchor_normals','triangles','face_normals')
        geometry=self.explorer.projector.geometry
        changed=any(getattr(job,name) is not value for name,value in zip(names,geometry))
        for name,value in zip(names,geometry):setattr(job,name,value)
        for name,source in [('original','c'),('hand','x'),('dual','dual'),('weights','weights'),
                            ('friction','friction'),('rotation','rotation'),('position','position'),
                            ('velocity','velocity'),('goal','goal')]:
            setattr(job,name,getattr(batch,source).flatten())
        job.object_radius=object_radius
        if changed:self.graph=None
        return changed

    def _launch(self):
        job=self.job
        wp.copy(job.current,job.original)
        wp.launch(self.prepare_kernel,self.groups,inputs=[job],device=self.device)
        for sweep in range(self.explorer.sweeps):
            job.sweep=sweep
            wp.launch(self.sample_kernel,job.points.shape[0],inputs=[job,1],device=self.device)
            wp.launch(self.solve_kernel,self.groups*job.samples,inputs=[job,1],device=self.device)
            wp.launch(self.check_kernel,self.groups,inputs=[job],device=self.device)
            wp.launch(self.sample_kernel,job.points.shape[0],inputs=[job,0],device=self.device)
            wp.launch(self.solve_kernel,self.groups*job.samples,inputs=[job,0],device=self.device)
            wp.launch(self.select_kernel,self.groups,inputs=[job],device=self.device)
            # The caller executes one contact block per ADMM round. Multiple
            # sweeps seed subsequent samples with the previously selected set.
            if self.explorer.sweeps > 1:
                wp.copy(job.current,job.selected)

    def solve(self):
        if self.device.is_cuda:
            if self.graph is None:
                self._launch();wp.synchronize_device(self.device)
                with wp.ScopedCapture(device=self.device) as capture:self._launch()
                self.graph=capture.graph
            wp.capture_launch(self.graph)
        else:self._launch()
        return self.job.selected
