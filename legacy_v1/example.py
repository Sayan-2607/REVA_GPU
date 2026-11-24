import numpy as np, itertools, rava_sim as R
R.CAP = np.array([8, 64, 256], float); R.T = 4
spec = [ # name, d, L, a, e (1-indexed incl), bid, rho, eta
 ("J1 LLM fine-tune",      [6,24,128], 3, 1, 4, 130, 0.55, 0.90),
 ("J2 Vision training",    [4,16, 64], 2, 1, 4, 48, 0.95, 0.85),
 ("J3 Inference burst",    [2, 8, 32], 2, 2, 3, 30, 0.98, 0.70),
 ("J4 Hyper-param sweep",  [4,16, 64], 2, 1, 4, 40, 0.90, 0.95),
 ("J5 CPU preprocessing",  [0,32,128], 2, 1, 4, 18, 0.97, 0.80),
 ("J6 Student RL project", [2, 8, 32], 4, 1, 4, 22, 0.92, 0.60)]
jobs=[]
for k,(nm,d,L,a,e,b,rho,eta) in enumerate(spec):
    d=np.array(d,float); D=L*float(np.sum(d/R.CAP)); win=e-a+1
    jobs.append(dict(id=k,name=nm,d=d,L=L,a=a-1,e=e-1,b=b,v=b,rho=rho,p=rho,eta=eta,u=eta,D=D,phi=(win-L)/win,c=0))
for j in jobs:
    w=R.rava_mult(j); print(f"{j['name']:24s} D={j['D']:.3f} phi={j['phi']:.2f} w={w:.3f} sigma={j['b']*w:.2f}")
def show(nm,a,p):
    ew=sum(jobs[i]['rho']*jobs[i]['v'] for i in a); bw=sum(jobs[i]['v'] for i in a)
    print(nm, {jobs[i]['name'][:2]:(a[i]+1, round(p.get(i,0),2)) for i in sorted(a)}, "declared W=",bw,"expected W=",round(ew,2),"revenue=",round(sum(p.values()),2))
a,p=R.greedy_mech(jobs,lambda j:1.0); show("SPA ",a,p)
a,p=R.greedy_mech(jobs,lambda j:1.0,pay_as_bid=True); show("FPA(bids as given)",a,p)
a,p=R.greedy_mech(jobs,lambda j:1/np.sqrt(j['D'])); show("LOS ",a,p)
a,p,ok=R.vcg(jobs); show("VCG ",a,p)
a,p=R.rava(jobs); show("RAVA",a,p)
a,_,_=R.ilp_opt(jobs,[j['rho']*j['b'] for j in jobs]); show("RA-OPT",a,{})
