"""Dev-only: Optuna study over all retrieval hyperparameters.

Searches on cached score matrices (dev/optuna_prep.npz) so each trial is pure
numpy. Tunes: BM25 (k1,b) [grid], fusion alpha, reranker blend weight, and
reranker pool depth. Objective = mean NDCG@10 on the 29 public queries.

NOTE: 29 queries is a tiny validation set; the winner is re-checked with the
real scripts/eval_public.py before adoption, and a bootstrap CI quantifies how
much of any gain is noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402

PREP = ROOT / "dev" / "cache" / "optuna_prep.npz"
N_TRIALS = 600
SEED = 13


def minmax_rows(m):
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (m - lo) / rng


def mm1(v):
    lo = float(v.min()); rng = float(v.max()) - lo
    return (v - lo) / (rng if rng > 1e-12 else 1.0)


def ndcg10(ranked_ids, gold_set):
    seen, gains = set(), []
    for pid in ranked_ids:
        if pid in seen:
            continue
        seen.add(pid)
        gains.append(1.0 if pid in gold_set else 0.0)
        if len(gains) >= 10:
            break
    dcg = gains[0] if gains else 0.0
    for i, g in enumerate(gains[1:], start=2):
        dcg += g / np.log2(i)
    n_rel = min(len(gold_set), 10)
    if n_rel == 0:
        return 0.0
    idcg = 1.0 + sum(1.0 / np.log2(i) for i in range(2, n_rel + 1))
    return dcg / idcg


class Data:
    def __init__(self):
        z = np.load(PREP)
        self.dense = z["dense"]
        self.bm25_grid = z["bm25_grid"]
        self.grid = z["grid"]
        self.ce = z["ce"]
        self.page_ids = z["page_ids"]
        gl = z["gold_len"]
        gp = z["gold"]
        self.gold = [set(int(x) for x in gp[i, : gl[i]]) for i in range(len(gl))]
        self.Dn = minmax_rows(self.dense)
        self.Bn = [minmax_rows(self.bm25_grid[g]) for g in range(len(self.grid))]
        self.Q = self.dense.shape[0]

    def score(self, g, alpha, weight, pool, count_miss=False):
        fused = alpha * self.Dn + (1 - alpha) * self.Bn[g]
        total, miss = 0.0, 0
        for qi in range(self.Q):
            row = fused[qi]
            cols = np.argpartition(-row, pool - 1)[:pool]
            cp = self.ce[qi, cols].copy()
            fin = np.isfinite(cp)
            miss += int((~fin).sum())
            if not fin.all():
                cp[~fin] = cp[fin].min() if fin.any() else 0.0
            blended = weight * mm1(cp) + (1 - weight) * mm1(row[cols])
            order = np.argsort(-blended)[:10]
            ranked = [int(self.page_ids[cols[i]]) for i in order]
            total += ndcg10(ranked, self.gold[qi])
        mean = total / self.Q
        return (mean, miss) if count_miss else mean


def main():
    if not PREP.exists():
        print("missing prep cache; run dev/optuna_prep.py first")
        return
    data = Data()

    # Reference: current production config within this cached approximation.
    cur_g = next(i for i, (k1, b) in enumerate(data.grid)
                 if abs(k1 - 1.5) < 1e-6 and abs(b - 0.75) < 1e-6)
    cur, cur_miss = data.score(cur_g, 0.6, 0.5, 100, count_miss=True)
    print(f"current config (cached approx) = {cur:.4f}  (pool CE misses={cur_miss})\n")

    def objective(trial):
        g = trial.suggest_categorical("grid", list(range(len(data.grid))))
        alpha = trial.suggest_float("alpha", 0.30, 0.80)
        weight = trial.suggest_float("weight", 0.0, 1.0)
        pool = trial.suggest_int("pool", 20, 150)
        return data.score(g, alpha, weight, pool)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    bp = study.best_params
    g = bp["grid"]; k1, b = data.grid[g]
    best, miss = data.score(g, bp["alpha"], bp["weight"], bp["pool"], count_miss=True)
    print(f"best value      = {study.best_value:.4f}")
    print(f"best params     = k1={k1}, b={b}, alpha={bp['alpha']:.3f}, "
          f"weight={bp['weight']:.3f}, pool={bp['pool']}  (CE misses={miss})")

    # Bootstrap CI of (best - current) over queries to gauge noise.
    rng = np.random.default_rng(0)
    fb = _per_query(data, g, bp["alpha"], bp["weight"], bp["pool"])
    fc = _per_query(data, cur_g, 0.6, 0.5, 100)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, data.Q, data.Q)
        diffs.append(float(fb[idx].mean() - fc[idx].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nbest - current  = {fb.mean()-fc.mean():+.4f}  "
          f"(95% bootstrap CI [{lo:+.4f}, {hi:+.4f}])")
    if lo > 0:
        print("=> gain is beyond the 29-query noise floor.")
    else:
        print("=> gain is within noise; prefer robust defaults / re-validate.")

    # Top-10 trials to eyeball stability of the optimum.
    print("\ntop trials:")
    for t in sorted(study.trials, key=lambda t: -(t.value or 0))[:10]:
        p = t.params; k1, b = data.grid[p["grid"]]
        print(f"  {t.value:.4f}  k1={k1} b={b} alpha={p['alpha']:.2f} "
              f"weight={p['weight']:.2f} pool={p['pool']}")


def _per_query(data, g, alpha, weight, pool):
    fused = alpha * data.Dn + (1 - alpha) * data.Bn[g]
    res = np.zeros(data.Q)
    for qi in range(data.Q):
        row = fused[qi]
        cols = np.argpartition(-row, pool - 1)[:pool]
        cp = data.ce[qi, cols].copy()
        fin = np.isfinite(cp)
        if not fin.all():
            cp[~fin] = cp[fin].min() if fin.any() else 0.0
        blended = weight * mm1(cp) + (1 - weight) * mm1(row[cols])
        order = np.argsort(-blended)[:10]
        ranked = [int(data.page_ids[cols[i]]) for i in order]
        res[qi] = ndcg10(ranked, data.gold[qi])
    return res


if __name__ == "__main__":
    main()
