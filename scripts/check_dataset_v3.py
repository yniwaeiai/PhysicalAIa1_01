import sys, json, collections, h5py
sys.path.insert(0, "/workspace/Kadai")
from pathlib import Path
from parc.perturbed_dataset import PerturbedDemoDataset, _hdf5_files
root = Path(sys.argv[1])
ds = PerturbedDemoDataset(root, only_success=True)
files = _hdf5_files(root)
print("files", len(files))
fr = collections.Counter(); ep = collections.Counter(); miss = []
for p in files:
    with h5py.File(p, "r") as f:
        base = str(f.attrs["base_task"]); src = "rec" if p.stem.endswith("_rec") else "v1"
        for k in f:
            g = f[k]
            for a in ("category", "source_demo"):
                if a not in g.attrs and a not in f.attrs:
                    miss.append((p.name, k, a))
            if not bool(g.attrs["success"]): continue
            ep[(base, src)] += 1; fr[(base, src)] += int(g.attrs["length"])
print("n_variants", len(files), "n_episodes", sum(ep.values()), "n_frames", sum(fr.values()), "dataset index", len(ds.index))
print("missing attrs", len(miss), miss[:3])
tot = collections.defaultdict(int)
for (b, s), n in sorted(fr.items()):
    print(f"{b:60s} {s:4s} ep {ep[(b,s)]:4d} frames {n:6d}")
    tot[s] += n
print("by source", dict(tot), "rec share %.3f" % (tot["rec"] / sum(tot.values())))
for b in sorted({b for b, _ in fr}):
    t = fr[(b, "v1")] + fr[(b, "rec")]
    print(f"{b:60s} total {t:6d} rec share {fr[(b,'rec')]/t:.3f}")
