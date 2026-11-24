import json, rava_sim as R
out={}
for n in [40,80,120,160]:
    out[n]=[R.run_one(n,1000*n+s,do_vcg=False) for s in range(20)]
    print('done',n,flush=True)
json.dump(out,open('results_fast.json','w'),default=float)
