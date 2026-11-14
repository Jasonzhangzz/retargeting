// C++ analytic shooting model. Python only constructs data and invokes solve.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/core/solvers/box-fddp.hpp>
#include <Eigen/Geometry>
#include <chrono>
namespace py=pybind11;
using V=Eigen::VectorXd;using M=Eigen::MatrixXd;using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;
M mat(py::handle value){auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(value);auto b=a.request();int rows=b.shape[0],cols=b.ndim==1?1:b.size/rows;return Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>((double*)b.ptr,rows,cols);}
M3 skew(V3 v){M3 r;r<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return r;}
V3 log3(M3 r){double c=std::max(-1.,std::min(1.,(r.trace()-1)*.5));V3 v;v<<r(2,1)-r(1,2),r(0,2)-r(2,0),r(1,0)-r(0,1);v*=.5;double theta=atan2(v.norm(),c);if(theta<1e-5)return v*(1+theta*theta/6);if(theta>3.141592653589793-1e-5){M3 p=r+r.transpose()-2*c*M3::Identity();int k=0;for(int j=1;j<3;j++)if(p.col(j).squaredNorm()>p.col(k).squaredNorm())k=j;V3 a=p.col(k).normalized();if(a.dot(v)<0)a=-a;return a*theta;}return v*(theta/std::max(v.norm(),1e-15));}
M3 log_inv(V3 v){double t=v.norm();M3 k=skew(v);double a=t<1e-4?1./12+t*t/720:(1-.5*t/tan(.5*t))/(t*t);return M3::Identity()-.5*k+a*k*k;}
struct Joint{int parent,child,qi;V3 xyz,axis;M3 r;};
struct FK{
 int d,nlinks,root;std::vector<Joint> joints;std::vector<int> outputs;M masks;M3 root_r;V3 root_p;
 FK(py::dict p){d=py::cast<int>(p["d"]);nlinks=py::cast<int>(p["nlinks"]);root=py::cast<int>(p["root"]);masks=mat(p["masks"]);root_r=mat(p["root_r"]);root_p=mat(p["root_p"]);outputs=py::cast<std::vector<int>>(p["outputs"]);for(auto handle:py::cast<py::list>(p["joints"])){py::dict j=py::cast<py::dict>(handle);joints.push_back({py::cast<int>(j["parent"]),py::cast<int>(j["child"]),py::cast<int>(j["qi"]),mat(j["xyz"]),mat(j["axis"]),mat(j["rotation"])});}}
 void eval(const V&q,M&p,std::vector<M3>&r,std::vector<M>&jp,std::vector<M>&jw)const{
  std::vector<M3> rotations(nlinks);std::vector<V3> positions(nlinks),axes(d),origins(d);rotations[root]=root_r;positions[root]=root_p;
  for(const auto&j:joints){V3 pos=positions[j.parent]+rotations[j.parent]*j.xyz;M3 rot=rotations[j.parent]*j.r;if(j.qi>=0){axes[j.qi]=rot*j.axis;origins[j.qi]=pos;rot=rot*Eigen::AngleAxisd(q[j.qi],j.axis).toRotationMatrix();}positions[j.child]=pos;rotations[j.child]=rot;}
  p.resize(outputs.size(),3);r.resize(outputs.size());jp.resize(outputs.size());jw.resize(outputs.size());
  for(size_t i=0;i<outputs.size();i++){int k=outputs[i];p.row(i)=positions[k].transpose();r[i]=rotations[k];jp[i]=M::Zero(3,d);jw[i]=M::Zero(3,d);for(int j=0;j<d;j++)if(masks(i,j)!=0){jp[i].col(j)=axes[j].cross(positions[k]-origins[j]);jw[i].col(j)=axes[j];}}
 }
};
struct Spatial{
 std::shared_ptr<FK> fk;bool human;int stage;M source,lap,diff,dirs,edges,pos,rot;V normal,posture,ref;double cosine,sine,thumb;
 Spatial(std::shared_ptr<FK>k,py::dict p):fk(k){human=py::cast<bool>(p["human"]);stage=py::cast<int>(p["stage"]);if(human){pos=mat(p["position"]);rot=mat(p["rotation"]);posture=mat(p["posture"]);ref=mat(p["reference"]);}else{source=mat(p["source"]);lap=mat(p["laplacian"]);diff=mat(p["differential"]);dirs=mat(p["directions"]);edges=mat(p["edges"]);normal=mat(p["normal"]);cosine=py::cast<double>(p["cosine"]);sine=py::cast<double>(p["sine"]);thumb=py::cast<double>(p["thumb_weight"]);}}
 int nr()const{return human?30+fk->d:(stage==2?63:0)+3*edges.rows()+12;}
 void eval(const V&q,V&r,M&j)const{
  M p;std::vector<M3> ro;std::vector<M> jp,jw;fk->eval(q,p,ro,jp,jw);r=V::Zero(nr());j=M::Zero(nr(),fk->d);int row=0;
  if(human){for(int f=0;f<5;f++){M3 target=rot.block<3,3>(3*f,0);V3 v=log3(ro[f].transpose()*target);double a=std::sqrt(.5);r.segment<3>(row)=a*300*(p.row(f).transpose()-pos.row(f).transpose());j.block(row,0,3,fk->d)=a*300*jp[f];row+=3;r.segment<3>(row)=a*5.729577951308232*v;j.block(row,0,3,fk->d)=-a*5.729577951308232*log_inv(v)*ro[f].transpose()*jw[f];row+=3;}r.tail(fk->d)=(q-ref).cwiseProduct(posture);j.bottomRows(fk->d)=posture.asDiagonal();return;}
  if(stage==2){M residual=lap*p-diff;for(int i=0;i<21;i++){double w=(i>=1&&i<=3)?0:std::sqrt(150.);r.segment<3>(row)=w*residual.row(i).transpose();M derivative=M::Zero(3,fk->d);for(int k=0;k<21;k++)derivative+=lap(i,k)*jp[k];j.block(row,0,3,fk->d)=w*derivative;row+=3;}}
  for(int k=0;k<edges.rows();k++){int a=(int)edges(k,0),b=(int)edges(k,1);V3 v=(p.row(b)-p.row(a)).transpose();double length=std::max(v.norm(),1e-8);V3 u=v/length;M du=(M3::Identity()-u*u.transpose())*(jp[b]-jp[a])/length;double w=std::sqrt(stage==2?.1:1.);r.segment<3>(row)=w*(u-dirs.row(k).transpose());j.block(row,0,3,fk->d)=w*du;row+=3;}
  V3 u=(p.row(3)-p.row(2)).transpose(),v=(p.row(4)-p.row(3)).transpose();double lu=std::max(u.norm(),1e-8),lv=std::max(v.norm(),1e-8);u/=lu;v/=lv;M ju=(M3::Identity()-u*u.transpose())*(jp[3]-jp[2])/lu,jv=(M3::Identity()-v*v.transpose())*(jp[4]-jp[3])/lv;
  r[row]=thumb*(u.dot(v)-cosine);j.row(row++)=thumb*(v.transpose()*ju+u.transpose()*jv);V3 nm=normal;r[row]=thumb*(nm.dot(u.cross(v))-sine);j.row(row++)=thumb*(v.cross(nm).transpose()*ju+nm.cross(u).transpose()*jv);
  for(int k=0;k<fk->d;k++)if(k%4==2||k%4==3){r[row]=-std::sqrt(20.)*std::min(q[k],0.);if(q[k]<0)j(row,k)=-std::sqrt(20.);row++;}
 }
};
class Action:public crocoddyl::ActionModelAbstract{
 public:std::shared_ptr<Spatial> spatial;double weight;bool terminal;size_t ncalc=0,ndiff=0;
 std::shared_ptr<crocoddyl::ActionModelBase> cloneAsDouble()const override{return std::make_shared<Action>(*this);}
 std::shared_ptr<crocoddyl::ActionModelBase> cloneAsFloat()const override{throw std::runtime_error("Wuji analytic model is FP64 only");}
 Action(std::shared_ptr<Spatial>s,double w,V lo,V hi,bool term=false):crocoddyl::ActionModelAbstract(std::make_shared<crocoddyl::StateVector>(s->fk->d),s->fk->d,s->nr()+s->fk->d),spatial(s),weight(w),terminal(term){set_u_lb(lo);set_u_ub(hi);}
 void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>&data,const Eigen::Ref<const V>&x,const Eigen::Ref<const V>&u)override{ncalc++;data->xnext=u;if(terminal){data->cost=0;return;}V r;M j;spatial->eval(u,r,j);data->cost=.5*r.squaredNorm()+.5*weight*(u-x).squaredNorm();}
 void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>&data,const Eigen::Ref<const V>&x)override{data->cost=0;data->xnext=x;}
 void calcDiff(const std::shared_ptr<crocoddyl::ActionDataAbstract>&data,const Eigen::Ref<const V>&x,const Eigen::Ref<const V>&u)override{
  ndiff++;int d=spatial->fk->d;V r;M j;spatial->eval(u,r,j);data->Fx.setZero();data->Fu.setIdentity();data->Lx=weight*(x-u);data->Lu=j.transpose()*r+weight*(u-x);data->Lxx=weight*M::Identity(d,d);data->Lxu=-weight*M::Identity(d,d);data->Luu=j.transpose()*j+weight*M::Identity(d,d);
 }
 void calcDiff(const std::shared_ptr<crocoddyl::ActionDataAbstract>&data,const Eigen::Ref<const V>&x)override{data->Fx.setIdentity();data->Fu.setZero();data->Lx.setZero();data->Lu.setZero();data->Lxx.setZero();data->Lxu.setZero();data->Luu.setZero();}
};
py::array_t<double> numpy_matrix(const M&value){py::array_t<double> result({value.rows(),value.cols()});auto a=result.mutable_unchecked<2>();for(int i=0;i<value.rows();i++)for(int j=0;j<value.cols();j++)a(i,j)=value(i,j);return result;}
struct Solver{
 std::shared_ptr<FK> fk;std::vector<std::shared_ptr<crocoddyl::SolverBoxFDDP>> solvers;std::vector<int> offsets;int d;V lo,hi;
 Solver(py::dict k,py::list spatial,std::vector<int>off,py::array reference,py::array lower,py::array upper,double w,int threads):fk(std::make_shared<FK>(k)),offsets(off){d=fk->d;lo=mat(lower);hi=mat(upper);M refs=mat(reference);for(size_t b=0;b+1<offsets.size();b++){std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> actions;std::shared_ptr<Spatial> last;for(int t=offsets[b];t<offsets[b+1];t++){last=std::make_shared<Spatial>(fk,py::cast<py::dict>(spatial[t]));actions.push_back(std::make_shared<Action>(last,w,lo,hi));}auto terminal=std::make_shared<Action>(last,0,lo,hi,true);auto problem=std::make_shared<crocoddyl::ShootingProblem>(refs.row(b).transpose(),actions,terminal);problem->set_nthreads(threads);auto solver=std::make_shared<crocoddyl::SolverBoxFDDP>(problem);solver->set_th_stop(1e-9);solvers.push_back(solver);}}
 py::dict solve(py::array initial,int iterations){M q=mat(initial);std::vector<int> counts;std::vector<double> costs;std::vector<bool> converged;size_t ncalc=0,ndiff=0;for(auto&solver:solvers)for(auto&m:solver->get_problem()->get_runningModels()){auto a=std::dynamic_pointer_cast<Action>(m);a->ncalc=0;a->ndiff=0;}auto begin=std::chrono::steady_clock::now();{py::gil_scoped_release release;for(size_t b=0;b<solvers.size();b++){std::vector<V> us,xs;xs.push_back(solvers[b]->get_problem()->get_x0());for(int t=offsets[b];t<offsets[b+1];t++){us.push_back(q.row(t).transpose());xs.push_back(q.row(t).transpose());}bool ok=solvers[b]->solve(xs,us,iterations,true,1e-3);auto result=solvers[b]->get_us();for(size_t i=0;i<result.size();i++)q.row(offsets[b]+i)=result[i].transpose();counts.push_back(solvers[b]->get_iter());costs.push_back(solvers[b]->get_cost());converged.push_back(ok);}}double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();py::array_t<double> out({q.rows(),q.cols()});auto a=out.mutable_unchecked<2>();for(int i=0;i<q.rows();i++)for(int j=0;j<q.cols();j++)a(i,j)=q(i,j);py::dict result;result["q"]=out;result["iterations"]=counts;result["cost"]=costs;result["converged"]=converged;for(auto&solver:solvers)for(auto&m:solver->get_problem()->get_runningModels()){auto a=std::dynamic_pointer_cast<Action>(m);ncalc+=a->ncalc;ndiff+=a->ndiff;}result["solver_seconds"]=seconds;result["function_evaluations"]=ncalc;result["gradient_evaluations"]=ndiff;return result;}
 void update(py::list payload){for(size_t b=0;b<solvers.size();b++){const auto&models=solvers[b]->get_problem()->get_runningModels();for(size_t i=0;i<models.size();i++){auto a=std::dynamic_pointer_cast<Action>(models[i]);a->spatial=std::make_shared<Spatial>(fk,py::cast<py::dict>(payload[offsets[b]+i]));}}}
 py::dict action_check(int index,py::array x,py::array u){auto a=std::dynamic_pointer_cast<Action>(solvers[0]->get_problem()->get_runningModels()[index]);auto data=a->createData();V xv=mat(x),uv=mat(u);a->calc(data,xv,uv);a->calcDiff(data,xv,uv);M g(2*d,1),h(2*d,2*d),dynamics(d,2*d);g.topRows(d)=data->Lx;g.bottomRows(d)=data->Lu;h.topLeftCorner(d,d)=data->Lxx;h.topRightCorner(d,d)=data->Lxu;h.bottomLeftCorner(d,d)=data->Lxu.transpose();h.bottomRightCorner(d,d)=data->Luu;dynamics.leftCols(d)=data->Fx;dynamics.rightCols(d)=data->Fu;py::dict result;result["cost"]=data->cost;result["gradient"]=numpy_matrix(g);result["hessian"]=numpy_matrix(h);result["dynamics"]=numpy_matrix(dynamics);return result;}
 py::dict spatial_check(int index,py::array q){auto action=std::dynamic_pointer_cast<Action>(solvers[0]->get_problem()->get_runningModels()[index]);V r;M j;action->spatial->eval(mat(q),r,j);py::array_t<double> rr(r.size());py::array_t<double> jj({j.rows(),j.cols()});std::copy(r.data(),r.data()+r.size(),rr.mutable_data());auto a=jj.mutable_unchecked<2>();for(int i=0;i<j.rows();i++)for(int k=0;k<j.cols();k++)a(i,k)=j(i,k);py::dict result;result["residual"]=rr;result["jacobian"]=jj;return result;}
};
PYBIND11_MODULE(wuji_crocoddyl,m){py::class_<Solver>(m,"Solver").def(py::init<py::dict,py::list,std::vector<int>,py::array,py::array,py::array,double,int>()).def("solve",&Solver::solve).def("spatial_check",&Solver::spatial_check).def("action_check",&Solver::action_check).def("update",&Solver::update);}
