// Studio-style per-finger bounded Gauss--Newton / active-set QP.
// Generic Moxian topology; no SDK binary, device API or 21-DoF truncation.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
using Eigen::Matrix3d; using Eigen::Vector3d; using Eigen::VectorXd; using Eigen::MatrixXd;
using Row3=Eigen::Matrix<double,3,3,Eigen::RowMajor>;
struct Model {
 int nb,nq,wrist; std::vector<int> parent,distal,periodic;
 std::vector<std::vector<int>> joints,chains,ids;
 std::vector<Vector3d> pos,axis,pivot,mountp;
 std::vector<Matrix3d> rot,mountr;
 VectorXd lower,upper,rest,posture,temporal;
 std::vector<double> config;
};
static Vector3d vec(const double* p){return Eigen::Map<const Vector3d>(p);}
static Matrix3d mat(const double* p){return Eigen::Map<const Row3>(p);}
static Matrix3d crossmat(const Vector3d& v){Matrix3d a;a<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return a;}
static Matrix3d jointrot(const Vector3d& axis,double q){
 double n=axis.norm();return n<1e-15?Matrix3d::Identity():Eigen::AngleAxisd(n*q,axis/n).toRotationMatrix();
}
static void body(const Model& m,int b,const VectorXd& q,Matrix3d& r,Vector3d& p,
                 std::vector<Vector3d>* axes=nullptr,std::vector<Vector3d>* centres=nullptr){
 p+=r*m.pos[b];r=r*m.rot[b];
 for(int j:m.joints[b]){
  if(axes){axes->push_back(r*m.axis[j]);centres->push_back(p+r*m.pivot[j]);}
  Matrix3d jr=jointrot(m.axis[j],q[j]);p+=r*(m.pivot[j]-jr*m.pivot[j]);r=r*jr;
 }
}
struct Sensor { Matrix3d r; Vector3d p; MatrixXd jp,jw; };
static Sensor sensor(const Model& m,int f,const VectorXd& q){
 Matrix3d r=Matrix3d::Identity();Vector3d p=Vector3d::Zero();
 std::vector<Vector3d> axes,centres;
 for(int b:m.chains[f])body(m,b,q,r,p,&axes,&centres);
 Sensor out;out.p=p+r*m.mountp[f];out.r=r*m.mountr[f];
 int n=axes.size();out.jp.resize(3,n);out.jw.resize(3,n);
 for(int j=0;j<n;j++){out.jw.col(j)=axes[j];out.jp.col(j)=axes[j].cross(out.p-centres[j]);}
 return out;
}
static Vector3d rotlog(const Matrix3d& r){
 Eigen::Quaterniond q(r);q.normalize();if(q.w()<0)q.coeffs()*=-1;
 double n=q.vec().norm();if(n<1e-12)return 2*q.vec();
 return q.vec()*(2*std::atan2(n,q.w())/n);
}
struct Eval {double cost,ep,er; VectorXd residual; MatrixXd jac;};
static Eval evaluate(const Model& m,int f,const VectorXd& q,const Vector3d& targetp,
                     const Matrix3d& targetr,const VectorXd& rest,const VectorXd& reference,bool history){
 auto s=sensor(m,f,q);Vector3d er=rotlog(s.r.transpose()*targetr);double angle=er.norm();
 Matrix3d k=crossmat(er);double c=angle<1e-5?1./12+angle*angle/720:(1-.5*angle/std::tan(.5*angle))/(angle*angle);
 MatrixXd jr=-(Matrix3d::Identity()-.5*k+c*k*k)*s.r.transpose()*s.jw;
 const auto& ids=m.ids[f];int n=ids.size();Eval out;
 out.residual.resize(6);out.residual.head(3)=(s.p-targetp)/m.config[5];out.residual.tail(3)=er/m.config[6];
 out.jac.resize(6,n);out.jac.topRows(3)=s.jp/m.config[5];out.jac.bottomRows(3)=jr/m.config[6];
 out.cost=out.residual.squaredNorm();out.ep=(s.p-targetp).norm();out.er=angle;
 for(int j:ids){out.cost+=m.posture[j]*std::pow(q[j]-rest[j],2);
  if(history)out.cost+=m.temporal[j]*std::pow(q[j]-reference[j],2);}
 return out;
}
static VectorXd boxqp(const MatrixXd& h,const VectorXd& g,const VectorXd& lo,const VectorXd& hi){
 int n=g.size();VectorXd x=VectorXd::Zero(n).cwiseMax(lo).cwiseMin(hi);std::vector<int> active(n,0);
 for(int i=0;i<n;i++){if(lo[i]>hi[i])throw std::runtime_error("inverted QP bounds");if(lo[i]==hi[i])active[i]=2;}
 for(int pass=0;pass<n+2;pass++){
  std::vector<int> free,fixed;for(int i=0;i<n;i++)(active[i]==0?free:fixed).push_back(i);
  if(!free.empty()){
   int nf=free.size();MatrixXd block(nf,nf);VectorXd rhs(nf);
   for(int a=0;a<nf;a++){rhs[a]=-g[free[a]];for(int i:fixed)rhs[a]-=h(free[a],i)*x[i];for(int b=0;b<nf;b++)block(a,b)=h(free[a],free[b]);}
   Eigen::LLT<MatrixXd> chol(block);if(chol.info()!=Eigen::Success)throw std::runtime_error("non-SPD QP");
   VectorXd diagonal=chol.matrixL().toDenseMatrix().diagonal();if(diagonal.array().square().minCoeff()<1e-12)throw std::runtime_error("singular QP");
   VectorXd solved=chol.solve(rhs),target=x;for(int a=0;a<nf;a++)target[free[a]]=solved[a];
   VectorXd delta=target-x;double alpha=1.;int hit=-1,side=0;
   for(int i:free){double a=delta[i]>0?(hi[i]-x[i])/delta[i]:delta[i]<0?(lo[i]-x[i])/delta[i]:INFINITY;
    if(a<alpha){alpha=std::max(0.,a);hit=i;side=delta[i]>0?1:-1;}}
   x=(x+alpha*delta).cwiseMax(lo).cwiseMin(hi);
   if(hit>=0){active[hit]=side;continue;}
  }
  VectorXd gradient=h*x+g;double violation=1e-10;int release=-1;
  for(int i=0;i<n;i++){double value=active[i]==-1?-gradient[i]:active[i]==1?gradient[i]:-INFINITY;
   if(value>violation){violation=value;release=i;}}
  if(release<0)break;active[release]=0;
 }
 return x;
}
extern "C" {
void* studio_qp_create(int nb,int nq,int wrist,const int* parent,const int* jointbody,
 const double* pos,const double* rot,const double* axis,const double* pivot,const int* distal,
 const double* mount,const double* limits,const double* rest,const double* posture,const double* temporal,
 const int* periodic,const double* config){
 try{
  auto m=std::make_unique<Model>();m->nb=nb;m->nq=nq;m->wrist=wrist;
  m->parent.assign(parent,parent+nb);m->distal.assign(distal,distal+5);m->periodic.assign(periodic,periodic+nq);
  m->joints.resize(nb);for(int j=0;j<nq;j++)m->joints.at(jointbody[j]).push_back(j);
  for(int b=0;b<nb;b++){m->pos.push_back(vec(pos+3*b));m->rot.push_back(mat(rot+9*b));}
  for(int j=0;j<nq;j++){m->axis.push_back(vec(axis+3*j));m->pivot.push_back(vec(pivot+3*j));}
  m->lower.resize(nq);m->upper.resize(nq);for(int j=0;j<nq;j++){m->lower[j]=limits[2*j];m->upper[j]=limits[2*j+1];}
  m->rest=Eigen::Map<const VectorXd>(rest,nq);m->posture=Eigen::Map<const VectorXd>(posture,nq);m->temporal=Eigen::Map<const VectorXd>(temporal,nq);
  m->config.assign(config,config+11);
  for(int f=0;f<5;f++){
   std::vector<int> chain,ids;int b=distal[f];for(int i=0;b!=wrist&&i<nb;i++){chain.push_back(b);b=parent[b];}
   if(b!=wrist)throw std::runtime_error("invalid finger chain");std::reverse(chain.begin(),chain.end());
   for(int bb:chain)for(int j:m->joints[bb])ids.push_back(j);
   m->chains.push_back(chain);m->ids.push_back(ids);m->mountp.push_back(vec(mount+16*f+3));
   m->mountp.back()<<mount[16*f+3],mount[16*f+7],mount[16*f+11];
   Matrix3d rr;for(int i=0;i<3;i++)for(int j=0;j<3;j++)rr(i,j)=mount[16*f+4*i+j];m->mountr.push_back(rr);
  }
  return m.release();
 }catch(...){return nullptr;}
}
void studio_qp_destroy(void* p){delete static_cast<Model*>(p);}
int studio_qp_step(void* p,const double* initial,const double* reference,int history,double dt,
 const double* targets,const double* rotations,double* result,double* reports){
 try{
  const auto& m=*static_cast<Model*>(p);VectorXd start=Eigen::Map<const VectorXd>(initial,m.nq),q=start;
  VectorXd ref=Eigen::Map<const VectorXd>(reference,m.nq);
  for(int f=0;f<5;f++){
   const auto& ids=m.ids[f];int n=ids.size();VectorXd rest=m.rest,lo(n),hi(n);
   for(int a=0;a<n;a++){int j=ids[a];if(m.periodic[j])rest[j]+=2*M_PI*std::round((start[j]-rest[j])/(2*M_PI));
    lo[a]=m.periodic[j]?start[j]-M_PI:m.lower[j];hi[a]=m.periodic[j]?start[j]+M_PI:m.upper[j];
    if(history){lo[a]=std::max(lo[a],start[j]-m.config[0]*dt);hi[a]=std::min(hi[a],start[j]+m.config[0]*dt);}}
   VectorXd origin=q,best=q;auto evaluateq=[&](const VectorXd& x){return evaluate(m,f,x,vec(targets+3*f),mat(rotations+9*f),rest,ref,history);};
   double bestcost=evaluateq(q).cost;int iterations=0;
   for(int retry=0;retry<3;retry++){
    q=origin;int limit=int(m.config[retry==0?7:8]);double multiplier=std::pow(2.,retry);
    for(int iter=0;iter<limit;iter++){
     auto v=evaluateq(q);iterations++;double mu=multiplier*m.config[1]*v.residual.squaredNorm()+m.config[2];
     MatrixXd h=v.jac.transpose()*v.jac;VectorXd g=v.jac.transpose()*v.residual;
     for(int a=0;a<n;a++){int j=ids[a];double temporal=history?m.temporal[j]:0;
      h(a,a)+=mu+m.posture[j]+temporal;g[a]+=m.posture[j]*(q[j]-rest[j])+temporal*(q[j]-ref[j]);}
     VectorXd lower(n),upper(n);for(int a=0;a<n;a++){lower[a]=m.config[3]*(lo[a]-q[ids[a]]);upper[a]=m.config[3]*(hi[a]-q[ids[a]]);}
     VectorXd delta=boxqp(h,g,lower,upper);bool accepted=false;
     for(double scale:{1.,.5,.25,.125,.0625}){
      VectorXd proposal=q;for(int a=0;a<n;a++)proposal[ids[a]]=std::clamp(q[ids[a]]+scale*delta[a],lo[a],hi[a]);
      double score=evaluateq(proposal).cost;
      if(std::isfinite(score)&&score<=v.cost+1e-12){q=proposal;accepted=true;if(score<bestcost){best=q;bestcost=score;}break;}
     }
     if(!accepted||delta.cwiseAbs().maxCoeff()<1e-7)break;
    }
    auto v=evaluateq(best);if(v.ep<=m.config[9]&&v.er<=m.config[10])break;
   }
   q=best;auto v=evaluateq(q);reports[5*f]=v.ep;reports[5*f+1]=v.er;reports[5*f+2]=iterations;reports[5*f+3]=bestcost;reports[5*f+4]=1;
  }
  if(!q.allFinite())return -2;Eigen::Map<VectorXd>(result,m.nq)=q;return 0;
 }catch(...){return -1;}
}
int studio_qp_forward(void* p,const double* angles,double* positions,double* rotations){
 try{
  const auto& m=*static_cast<Model*>(p);VectorXd q=Eigen::Map<const VectorXd>(angles,m.nq);
  std::vector<Matrix3d> r(m.nb,Matrix3d::Identity());std::vector<Vector3d> t(m.nb,Vector3d::Zero());
  for(int b=1;b<m.nb;b++){r[b]=r[m.parent[b]];t[b]=t[m.parent[b]];body(m,b,q,r[b],t[b]);}
  Matrix3d wr=r[m.wrist].transpose();Vector3d wp=t[m.wrist];
  for(int b=0;b<m.nb;b++){Eigen::Map<Vector3d>(positions+3*b)=wr*(t[b]-wp);Eigen::Map<Row3>(rotations+9*b)=wr*r[b];}
  return 0;
 }catch(...){return -1;}
}
int studio_qp_sensor(void* p,int finger,const double* angles,double* position,double* rotation,double* linear,double* angular){
 try{const auto& m=*static_cast<Model*>(p);auto s=sensor(m,finger,Eigen::Map<const VectorXd>(angles,m.nq));
  Eigen::Map<Vector3d> op(position);op=s.p;Eigen::Map<Row3> rr(rotation);rr=s.r;
  int n=m.ids[finger].size();for(int i=0;i<3;i++)for(int j=0;j<n;j++){linear[i*n+j]=s.jp(i,j);angular[i*n+j]=s.jw(i,j);}return 0;
 }catch(...){return -1;}
}
}
