"""P2 (計画§19.4): 回復 demo の冒頭 10 行動の xyz と、v1 方策が同状態で実行した xyz の cos。"""
import json,csv,glob,h5py,numpy as np,collections
R='/path/to/kadai_work/'
summ={}
for l in open(R+'kadai/state/rec_collect_20261016/summary.jsonl'):
    r=json.loads(l); summ[r['id']]=r
sel=[json.loads(l) for l in open(R+'kadai/state/rec_collect_20261016/selected.jsonl')]
rows=[];tc={}
for s in sel:
    m=summ[s['id']]; task,seed,t=s['task'],s['seed'],s['t']
    k=(task,seed)
    if k not in tc: tc[k]=np.load(f"{R}data/trace_rec/{task}_s{seed}_r0.npz",allow_pickle=True)
    z=tc[k]
    h=h5py.File(R+m['demo'],'r'); g=h[list(h.keys())[0]]
    act=g['actions'][:]; st=g['state'][:]
    a=act[:10,:3].mean(0)
    for key in ('act_exec','act_raw'):
        pass
    be=z['act_exec'][t:t+10,:3].mean(0); br=z['act_raw'][t:t+10,:3].mean(0)
    b=be
    cos=lambda x,y:float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
    eeerr=float(np.linalg.norm(st[0,:3]-z['ee'][t-1])*1000)
    rows.append(dict(id=s['id'],task=task,seed=seed,t=t,origin='S' if s['origin_success'] else 'F',
        cos=cos(a,b),cos_raw=cos(a,br),az=a[2],bz=b[2],oppz=int(np.sign(a[2])!=np.sign(b[2])),
        na=np.linalg.norm(a),nb=np.linalg.norm(b),ee_err_mm=eeerr,state0=st[0,:3].tolist(),
        amax=float(np.abs(act[:,:6]).max()),bmax=float(np.abs(z['act_exec'][:,:6]).max()),n_trace=len(z['act_exec'])))
flds=['id','task','seed','t','origin','cos','cos_raw','az','bz','oppz','na','nb','ee_err_mm']
with open(R+'kadai/state/rec_mode_conflict_20261016.csv','w') as f:
    w=csv.DictWriter(f,flds,extrasaction='ignore');w.writeheader();w.writerows(rows)
def q(x,f):x=np.array(x);return f(x)
out=[]
out.append("| タスク | 由来 | n | cos 中央値 | cos<0.5 | cos<0 | z 符号反対 | 中央|a| | 中央|b| | 中央 cos(raw) |\n|---|---|---|---|---|---|---|---|---|---|")
for task in ('drawer','k4','k6'):
  for o in 'SF':
    r=[x for x in rows if x['task']==task and x['origin']==o]
    c=np.array([x['cos'] for x in r])
    out.append(f"| {task} | {o} | {len(r)} | {np.median(c):.3f} | {np.mean(c<.5):.3f} | {np.mean(c<0):.3f} | {np.mean([x['oppz'] for x in r]):.3f} | {np.median([x['na'] for x in r]):.3f} | {np.median([x['nb'] for x in r]):.3f} | {np.median([x['cos_raw'] for x in r]):.3f} |")
out.append("\n")
for task in ('drawer','k4','k6'):
  out.append(f"\n{task} S の t 別\n\n| t | n | cos 中央値 | cos<0.5 | cos<0 | z 反対 | 中央|a| | 中央|b| |\n|---|---|---|---|---|---|---|---|")
  for t in range(5,45,5):
    r=[x for x in rows if x['task']==task and x['origin']=='S' and x['t']==t]
    if not r: continue
    c=np.array([x['cos'] for x in r])
    out.append(f"| {t} | {len(r)} | {np.median(c):.3f} | {np.mean(c<.5):.3f} | {np.mean(c<0):.3f} | {np.mean([x['oppz'] for x in r]):.3f} | {np.median([x['na'] for x in r]):.3f} | {np.median([x['nb'] for x in r]):.3f} |")
ee=np.array([x['ee_err_mm'] for x in rows])
out.append(f"\nee 照合(demo state[0] xyz vs trace ee[t]): 中央 {np.median(ee):.2f} mm, p90 {np.percentile(ee,90):.2f}, max {ee.max():.2f}")
out.append(f"demo |action|max 中央 {np.median([x['amax'] for x in rows]):.3f}, trace act_exec |action|max 中央 {np.median([x['bmax'] for x in rows]):.3f}")
print("\n".join(out))
