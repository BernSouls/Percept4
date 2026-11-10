"""U27 self-test for the patched depth.py / surface_fragments.py (run from the tree root:
python _u27_selftest.py).  New-behaviour assertions only; no baseline copy needed."""
import math
from perception import surface_fragments as sf, depth as dp

def chain(name, pts, signed=0.5, nrm=(0.0, 1.0)):
    return [dict(chain=name, vertex=i, n_chain=len(pts), pos=p, closed=False, alpha=(name, i),
                 signed=signed, normal=nrm, neighbor=None) for i, p in enumerate(pts)]
def line(x0, y0, x1, y1, step=2.0):
    n = max(1, int(round(math.hypot(x1 - x0, y1 - y0) / step)))
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]
def scene(bar_normal):
    return (chain("stem", line(0, 0, 40, 0)) + chain("bar", line(42, -30, 42, 30), nrm=bar_normal)
            + chain("cont", line(90, 0, 130, 0)))

# 1 true occlusion kept, same-object crease rejected, undecided bar kept + counted
assert len(sf.t_junction_sites(scene((1.0, 0.0)))[0]) == 1
s, d = sf.t_junction_sites(scene((-1.0, 0.0)))
assert len(s) == 0 and d["rejected_stem_on_owner_side"] == 1
r = scene((1.0, 0.0))
for x in r:
    if x["chain"] == "bar": x["signed"] = 0.0
s, d = sf.t_junction_sites(r); assert len(s) == 1 and d["unchecked_bar_owner"] == 1
# 2 switches off restores the old decision
assert len(sf.t_junction_sites(scene((-1.0, 0.0)), require_stem_outside_bar=False)[0]) == 1
# 3 continuation must be past the bar
r = chain("stem", line(0, 0, 30, 0)) + chain("bar", line(34, -30, 34, 30), nrm=(1.0, 0.0)) \
    + chain("pre", [(31.0, 0.0), (32.0, 0.0)])
assert not [x for x in sf.t_junction_sites(r)[0] if x["chain_end"] == "stem"]
assert [x for x in sf.t_junction_sites(r, continuation_beyond_bar=False)[0] if x["chain_end"] == "stem"]
# 4 chain_arms totality
rs = chain("A", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]); rs[0]["pos"] = None
assert len(sf.chain_arms(rs)) == 2
rs = chain("B", [(10, 10), (11, 10), (12, 10), (13, 10), (14, 10)]); del rs[0]["pos"]
assert (0.0, 0.0) not in [a["pos"] for a in sf.chain_arms(rs)]
rs = chain("C", [(float(i), 0.0) for i in range(8)])[2:]
assert [a["vertex"] for a in sf.chain_arms(rs, true_ends_only=True)] == [7]
# 5 alpha=None never matches
def frag(i, alpha, pos, own):
    return sf.SurfaceFragment(index=i, position=pos, tangent=(own[1], -own[0]), owner_normal=own, owner_side=1.0,
        curvature=None, curvature_sign=0.0, depth_layer=None, alpha=alpha, chain=f"k{i}", vertex=0, enclosure=None)
fr = [frag(0, "P", (0.0, 0.0), (1.0, 0.0)), frag(1, None, (9.0, 0.0), (1.0, 0.0))]
recs = [dict(chain="k0", vertex=0, pos=(0.0, 0.0), alpha="P", neighbor=None, normal=(1, 0), signed=1),
        dict(chain="k1", vertex=0, pos=(9.0, 0.0), alpha=None, neighbor=None, normal=(1, 0), signed=1)]
assert sf.occlusion_order_edges(fr, recs) == []
# 6 t_junction_order no longer drops silently
_, d = sf.t_junction_order([], scene((1.0, 0.0)), return_diag=True)
assert d["sites_no_fragment"] == 1
# 7 depth.py
k, rem = dp.resolve_depth_constraints([("a", "b", float("nan"), "s"), ("b", "a", 1.0, "t")])
assert k == [("b", "a", 1.0, "t")] and rem == []
a = dp.resolve_depth_constraints([("a", "b", 3.0, "s"), ("a", "b", 5.0, "s")], {"s": 0.0})
b = dp.resolve_depth_constraints([("a", "b", 5.0, "s"), ("a", "b", 3.0, "s")], {"s": 0.0})
assert a == b
try: dp.depth_layout([("a", "b", 1, "s"), ("b", "a", 1, "s")]); raise SystemExit("no raise")
except ValueError: pass
kept, _ = dp.resolve_depth_constraints([("A", "B", 1, "s"), ("B", "C", 1, "s"), ("D", "C", 1, "s")])
lo, hi, _ = dp.depth_layout_bounds(kept); assert (lo["D"], hi["D"]) == (0, 1) and lo["B"] == hi["B"] == 1
class Det:
    def __init__(s, rho): s.rho = rho
assert dp._extract_member_node_ids(Det([("member_node",), (), ("member_node", (5,), 1), ("member_node", (3,), 1)])) == [3, 5]
print("U27 selftest: all pass")
