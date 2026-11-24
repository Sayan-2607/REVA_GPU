import json, time, numpy as np, rava_sim as R
out=[]
for s in range(10):
    rng=np.random.default_rng(40000+s); jobs=R.gen_jobs(40,rng)
    for j in jobs: j['b']=j['v']
    t=time.time(); a,p,ok=R.vcg(jobs,tl=30); m=R.metrics(jobs,a,p); m['t']=time.time()-t; m['exact']=bool(ok)
    a2,_,ok2=R.ilp_opt(jobs,[j['rho']*j['b'] for j in jobs],tl=30); m2=R.metrics(jobs,a2,{}); m2['exact']=bool(ok2)
    out.append({'VCG':m,'RA-OPT':m2}); print(s, round(m['t'],1), ok, ok2, flush=True)
    json.dump(out,open('results_vcg.json','w'),default=float)
