import numpy as np

from scipy.stats import wilcoxon



def _paired_arrays(rows, metric, *, seed_key="seed"):
    """Return paired (cyl, sph) arrays matched by seed."""
    cyl = {int(r[seed_key]): float(r[metric]) for r in rows if int(r["scenario"]) == 0 and metric in r}
    sph = {int(r[seed_key]): float(r[metric]) for r in rows if int(r["scenario"]) == 1 and metric in r}

    common_seeds = sorted(set(cyl.keys()) & set(sph.keys()))
    x = np.array([cyl[s] for s in common_seeds], dtype=float)  # cylinder
    y = np.array([sph[s] for s in common_seeds], dtype=float)  # sphere
    return common_seeds, x, y

def paired_wilcoxon_report(rows, metrics=("bb_acc", "cbm_acc", "F_pair_raw_mean", "E_pair_raw_mean")):
    print("Paired Wilcoxon: sphere vs cylinder (paired by seed)")
    print("H1: median(sphere - cylinder) != 0\n")

    for m in metrics:
        seeds, x_cyl, y_sph = _paired_arrays(rows, m)
        if len(seeds) < 5:
            print(f"{m}: not enough paired seeds (n={len(seeds)})")
            continue

        diff = y_sph - x_cyl
        median_diff = float(np.median(diff))
        mean_cyl = float(np.mean(x_cyl))
        mean_sph = float(np.mean(y_sph))

        # Wilcoxon needs nonzero diffs; scipy handles 'zero_method'
        stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")

        print(f"--- {m} ---")
        print(f"n (paired seeds)          : {len(seeds)}")
        print(f"mean cylinder / sphere    : {mean_cyl:.4f} / {mean_sph:.4f}")
        print(f"median(sphere - cylinder) : {median_diff:.6f}")
        print(f"Wilcoxon W                : {float(stat):.3f}")
        print(f"p-value                   : {float(p):.6g}\n")