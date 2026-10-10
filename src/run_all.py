"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all            (full run, ~15-20 min on a laptop)

Steps
  1. load + clean the data (cleaning log printed), EDA figures
  2. features, stratified train/val/test split, tier label from TRAIN median
  3. tune OUR decision tree (Task A regression, Task B classification) on VAL
  4. tune OUR Pegasos SVM (linear + random-Fourier-feature) on VAL
  5. final test evaluation vs scikit-learn baselines with matched settings,
     5-fold CV mean±std on the development set, bootstrap std on test,
     fit/predict times + a scaling experiment
  6. error analysis (importances, SVM weights, residuals by price/district)
  7. writes results/*.json|txt, report/tables/*.tex and report/figures/*.png

Everything is deterministic (SEED everywhere). The test split is touched
only in step 5, after all hyperparameters are frozen.
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep as dp
from . import evaluate as ev
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RFFPegasosSVM

SEED = dp.SEED
FIG_DIR = "report/figures"
TAB_DIR = "report/tables"
RES_DIR = "results"
SVC_SUBSAMPLE = 10_000   # kernel SVC is O(n^2)-O(n^3): fit on a stratified subsample
F1_TOL = 0.002           # SVM selection: most-regularised config within F1_TOL of the best val F1
                         # (≈ the bootstrap std of a val F1; differences below it are noise)

# --- figure style (reference palette; recessive grid, thin marks) ---------------
C_OURS, C_SK, C_THIRD, C_FOURTH = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 5, "legend.frameon": False, "figure.facecolor": "white",
})

RESULTS: dict = {}


def say(*a):
    print(*a, flush=True)


def section(title):
    say("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches="tight")
    plt.close(fig)


_TR = str.maketrans("əƏıİşŞçÇğĞöÖüÜ", "eEiIsScCgGoOuU")


def tex(s) -> str:
    """ASCII-safe LaTeX text (Azerbaijani letters transliterated)."""
    s = str(s).translate(_TR)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("±", r"$\pm$"), ("²", r"$^2$"), ("≤", r"$\le$"), ("λ", r"$\lambda$")]:
        s = s.replace(a, b)
    return s


def write_tex_table(name, header, rows, colspec=None):
    colspec = colspec or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{tabular}{@{}" + colspec + r"@{}}", r"\toprule",
             " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_tex_macros(macros: dict):
    """\\newcommand for every number quoted in the report text."""
    with open(os.path.join(TAB_DIR, "numbers.tex"), "w", encoding="utf-8") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{tex(v)}}}\n")


def ms(mean_std, fmt="{:.3f}"):
    m, s = mean_std
    return f"{fmt.format(m)} ± {fmt.format(s)}"


# ================================================================================
# 1. data
# ================================================================================
def step_data():
    section("1. LOAD + CLEAN")
    raw = dp.load_raw()
    df = dp.clean(raw)
    for line in dp.CLEANING_LOG:
        say("  -", line)
    RESULTS["cleaning_log"] = list(dp.CLEANING_LOG)
    RESULTS["n_raw"], RESULTS["n_clean"] = len(raw), len(df)
    return df


def step_eda(df, thr):
    section("1b. EDA figures")
    price = df["price"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].hist(price / 1e3, bins=120, color=C_OURS)
    ax[0].set(xlabel="price (thousand AZN)", ylabel="listings", title="Price: heavy right tail")
    ax[1].hist(np.log(price), bins=80, color=C_OURS)
    ax[1].axvline(np.log(thr), color=INK, lw=1, ls="--")
    ax[1].text(np.log(thr) + 0.08, ax[1].get_ylim()[1] * 0.92, "tier threshold\n(train median)",
               fontsize=7, color=INK2, va="top")
    ax[1].set(xlabel="log(price)", title="log(price): close to symmetric")
    save(fig, "eda_price_distribution")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    hb = ax[0].hexbin(df["area_m2"], price, gridsize=60, xscale="log", yscale="log",
                      bins="log", cmap="Blues", mincnt=1, linewidths=0)
    fig.colorbar(hb, ax=ax[0], label="listings (log)")
    ax[0].set(xlabel="area (m², log)", ylabel="price (AZN, log)", title="Area vs price")
    rooms = np.clip(df["rooms"].fillna(-1).astype(int), -1, 8)
    groups = [np.log(price[rooms == r]) for r in range(1, 9)]
    ax[1].boxplot(groups, tick_labels=[*map(str, range(1, 8)), "8+"], showfliers=False,
                  medianprops=dict(color=C_SK, lw=1.5), boxprops=dict(color=INK2),
                  whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ax[1].set(xlabel="rooms", ylabel="log(price)", title="Price by number of rooms")
    save(fig, "eda_area_rooms")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    order = np.argsort(price)
    sc = ax[0].scatter(df["lng"].to_numpy()[order], df["lat"].to_numpy()[order],
                       c=np.log(price[order]), s=1.2, cmap="Blues", linewidths=0)
    fig.colorbar(sc, ax=ax[0], label="log(price)")
    ax[0].set(xlabel="longitude", ylabel="latitude", title="Location effect (lat/lng)")
    ax[0].set_aspect("equal")
    med = df.groupby("district")["price"].agg(["median", "size"]).query("size >= 100")
    med = med.sort_values("median")
    ax[1].barh([d.replace(" r.", "") for d in med.index], med["median"] / 1e3, color=C_OURS, height=0.7)
    ax[1].set(xlabel="median price (thousand AZN)", title="Median price by district")
    ax[1].grid(axis="y", visible=False)
    save(fig, "eda_location")

    by_cat = df.groupby("category")["price"].median()
    RESULTS["median_price_by_category"] = by_cat.to_dict()
    RESULTS["median_price_by_district"] = med["median"].to_dict()


# ================================================================================
# 2. features + split
# ================================================================================
def step_split(df):
    section("2. FEATURES + SPLIT")
    price = df["price"].to_numpy()
    strata = dp.price_strata(price)
    tr, va, te = dp.stratified_split_indices(strata, 0.15, 0.15, SEED)
    builder = dp.FeatureBuilder().fit(df.iloc[tr])          # TRAIN statistics only
    X, y_log, names = builder.transform(df)
    for leak in dp.LEAKAGE_COLUMNS + ["price"]:
        assert not any(leak in n for n in names), f"leakage column {leak} in features!"

    y_tr_tier, thr = dp.make_tier_label(price[tr])          # threshold from TRAIN only
    y_tier = dp.make_tier_label(price, thr)[0]
    assert thr == float(np.median(price[tr]))

    sizes = {"train": len(tr), "val": len(va), "test": len(te)}
    bal = {k: float(y_tier[i].mean()) for k, i in zip(sizes, (tr, va, te))}
    say(f"  features: {X.shape[1]}  | split sizes {sizes}")
    say(f"  tier threshold (train median price) = {thr:,.0f} AZN")
    say("  premium share: " + ", ".join(f"{k} {v:.3f}" for k, v in bal.items()))
    RESULTS.update(split_sizes=sizes, tier_threshold=thr, tier_balance=bal,
                   n_features=X.shape[1], feature_names=names)

    write_tex_table("dataset", ["Property", "Value"], [
        ["Raw rows (scraped records)", f"{RESULTS['n_raw']:,}"],
        ["Rows after cleaning", f"{len(df):,}"],
        ["Features (after one-hot)", f"{X.shape[1]}"],
        ["Train / Val / Test", f"{len(tr):,} / {len(va):,} / {len(te):,}"],
        ["Tier threshold (train median)", f"{thr:,.0f} AZN"],
        ["Premium share train / val / test (%)",
         " / ".join(f"{100 * bal[k]:.1f}" for k in ("train", "val", "test"))],
    ])
    return dict(X=X, y_log=y_log, y_tier=y_tier, price=price, names=names,
                tr=tr, va=va, te=te, thr=thr)


# ================================================================================
# 3. tree tuning (validation only)
# ================================================================================
DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20, None]


def dlabel(d):
    return "∞" if d is None else str(d)


def step_tune_tree(D):
    X, tr, va = D["X"], D["tr"], D["va"]
    section("3a. TREE — Task A (regression on log price): max_depth × min_samples_leaf")
    yA = D["y_log"]
    leafs = [1, 5, 20, 50]
    gridA = {}
    for msl in leafs:
        for d in DEPTHS:
            m = DecisionTree("regression", "mse", max_depth=d, min_samples_leaf=msl).fit(X[tr], yA[tr])
            gridA[(d, msl)] = (ev.rmse(yA[tr], m.predict(X[tr])), ev.rmse(yA[va], m.predict(X[va])),
                               m.get_n_leaves())
        say(f"  msl={msl:>3}: val RMSE(log) " +
            " ".join(f"d{dlabel(d)}={gridA[(d, msl)][1]:.4f}" for d in DEPTHS))
    bestA = min(gridA, key=lambda k: gridA[k][1])
    say(f"  -> chosen: max_depth={dlabel(bestA[0])}, min_samples_leaf={bestA[1]} "
        f"(val RMSE {gridA[bestA][1]:.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    xs = np.arange(len(DEPTHS))
    ax[0].plot(xs, [gridA[(d, 1)][0] for d in DEPTHS], "-o", color=C_OURS, label="train")
    ax[0].plot(xs, [gridA[(d, 1)][1] for d in DEPTHS], "-o", color=C_SK, label="validation")
    ax[0].set(xticks=xs, xticklabels=[dlabel(d) for d in DEPTHS], xlabel="max_depth",
              ylabel="RMSE (log price)", title="Task A, min_samples_leaf=1")
    ax[0].legend()
    for msl, c in zip(leafs, [C_OURS, C_SK, C_THIRD, C_FOURTH]):
        ax[1].plot(xs, [gridA[(d, msl)][1] for d in DEPTHS], "-o", color=c, label=f"leaf ≥ {msl}")
    ax[1].set(xticks=xs, xticklabels=[dlabel(d) for d in DEPTHS], xlabel="max_depth",
              ylabel="validation RMSE (log)", title="Task A: min_samples_leaf regularises")
    ax[1].set_ylim(top=0.40)
    ax[1].legend(ncol=2, fontsize=7)
    save(fig, "tree_taskA_validation_curves")

    section("3b. TREE — Task B (price tier): gini vs entropy × max_depth, then min_samples_leaf")
    yB = D["y_tier"]
    gridB = {}
    for crit in ("gini", "entropy"):
        for d in DEPTHS:
            m = DecisionTree("classification", crit, max_depth=d).fit(X[tr], yB[tr])
            gridB[(crit, d)] = (ev.accuracy(yB[tr], m.predict(X[tr])),
                                ev.precision_recall_f1(yB[va], m.predict(X[va]))[2],
                                ev.accuracy(yB[va], m.predict(X[va])))
        say(f"  {crit:>7}: val F1 " + " ".join(f"d{dlabel(d)}={gridB[(crit, d)][1]:.4f}" for d in DEPTHS))
    crit_best, d_best = max(gridB, key=lambda k: gridB[k][1])
    leafsB = [1, 5, 10, 20, 50, 100]
    gridB2 = {}
    for msl in leafsB:
        m = DecisionTree("classification", crit_best, max_depth=d_best, min_samples_leaf=msl).fit(X[tr], yB[tr])
        gridB2[msl] = ev.precision_recall_f1(yB[va], m.predict(X[va]))[2]
    msl_best = max(gridB2, key=gridB2.get)
    say(f"  min_samples_leaf sweep at {crit_best}, depth {dlabel(d_best)}: "
        + " ".join(f"{k}={v:.4f}" for k, v in gridB2.items()))
    bestB = (crit_best, d_best, msl_best)
    say(f"  -> chosen: criterion={crit_best}, max_depth={dlabel(d_best)}, min_samples_leaf={msl_best} "
        f"(val F1 {gridB2[msl_best]:.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for crit, c in (("gini", C_OURS), ("entropy", C_SK)):
        ax[0].plot(xs, [1 - gridB[(crit, d)][0] for d in DEPTHS], "--", color=c, lw=1.2,
                   label=f"{crit} train error")
        ax[0].plot(xs, [1 - gridB[(crit, d)][2] for d in DEPTHS], "-o", color=c, label=f"{crit} val error")
    ax[0].set(xticks=xs, xticklabels=[dlabel(d) for d in DEPTHS], xlabel="max_depth",
              ylabel="error rate", title="Task B: gini vs entropy")
    ax[0].legend(fontsize=7)
    ax[1].plot(range(len(leafsB)), [gridB2[k] for k in leafsB], "-o", color=C_OURS)
    ax[1].set(xticks=range(len(leafsB)), xticklabels=leafsB, xlabel="min_samples_leaf",
              ylabel="validation F1", title=f"Task B: {crit_best}, max_depth={dlabel(d_best)}")
    save(fig, "tree_taskB_validation_curves")

    RESULTS["tree_tuning"] = {
        "taskA_grid": {f"d={dlabel(d)},msl={m}": v for (d, m), v in gridA.items()},
        "taskB_grid": {f"{c},d={dlabel(d)}": v for (c, d), v in gridB.items()},
        "taskB_msl": gridB2,
        "taskA_best": {"max_depth": bestA[0], "min_samples_leaf": bestA[1]},
        "taskB_best": {"criterion": crit_best, "max_depth": d_best, "min_samples_leaf": msl_best},
    }
    gini_best = max(gridB[("gini", d)][1] for d in DEPTHS)
    ent_best = max(gridB[("entropy", d)][1] for d in DEPTHS)
    RESULTS["gini_vs_entropy_best_val_f1"] = {"gini": gini_best, "entropy": ent_best}
    return {"A": {"max_depth": bestA[0], "min_samples_leaf": bestA[1]},
            "B": {"criterion": crit_best, "max_depth": d_best, "min_samples_leaf": msl_best}}


# ================================================================================
# 4. SVM tuning (validation only)
# ================================================================================
def step_tune_svm(D):
    X, tr, va = D["X"], D["tr"], D["va"]
    yB = D["y_tier"]
    Xtr, Xva = dp.standardize(X[tr], X[va])                 # scaling: TRAIN stats only
    section("4a. PEGASOS SVM — λ sweep (linear)")
    lams = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    rows = {}
    for lam in lams:
        m = PegasosSVM(lambda_=lam, n_epochs=30, random_state=SEED).fit(Xtr, yB[tr])
        s = m.decision_function(Xva)
        rows[lam] = dict(f1=ev.precision_recall_f1(yB[va], m.predict(Xva))[2], auc=ev.roc_auc(yB[va], s),
                         norm_w=m.norm_w_, margin=m.margin_width_, J=m.loss_history_[-1],
                         sv_frac=float(m.support_vector_mask(Xtr, yB[tr]).mean()))
        say(f"  λ={lam:.0e}: val F1 {rows[lam]['f1']:.4f}  AUC {rows[lam]['auc']:.4f}  ‖w‖ {m.norm_w_:7.3f}"
            f"  margin 2/‖w‖ {m.margin_width_:.4f}  SV {rows[lam]['sv_frac']:.1%}  J {rows[lam]['J']:.4f}")
    f1_max = max(r["f1"] for r in rows.values())
    lam_best = max(l for l in lams if rows[l]["f1"] >= f1_max - F1_TOL)   # largest λ = widest margin
    say(f"  -> chosen λ = {lam_best:g} (largest λ with val F1 within {F1_TOL} of the best {f1_max:.4f}; "
        f"smaller λ gain < noise but converge far slower: J above)")

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    lx = np.log10(lams)
    ax[0].plot(lx, [rows[l]["f1"] for l in lams], "-o", color=C_OURS, label="F1")
    ax[0].plot(lx, [rows[l]["auc"] for l in lams], "-o", color=C_SK, label="ROC-AUC")
    ax[0].set(xlabel="log10 λ", title="validation metric")
    ax[0].legend(fontsize=7)
    ax[1].plot(lx, [rows[l]["margin"] for l in lams], "-o", color=C_OURS)
    ax[1].set(xlabel="log10 λ", yscale="log", title="margin width 2/‖w‖")
    ax[2].plot(lx, [100 * rows[l]["sv_frac"] for l in lams], "-o", color=C_OURS)
    ax[2].set(xlabel="log10 λ", title="support vectors (% train)")
    save(fig, "svm_lambda_sweep")

    section("4b. PEGASOS SVM — epochs, learning-rate schedule, batch size (at chosen λ)")
    sched = {
        "pegasos 1/(λt), averaged": dict(lr_schedule="pegasos"),
        "pegasos 1/(λt), last iterate": dict(lr_schedule="pegasos", average=False),
        "invscaling 0.1/√t": dict(lr_schedule="invscaling", eta0=0.1),
        "constant 0.01": dict(lr_schedule="constant", eta0=0.01),
        "full batch 1/(λt)": dict(lr_schedule="pegasos", batch_size=None),
    }
    sched_res = {}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for (name, kw), c in zip(sched.items(), [C_OURS, C_FOURTH, C_SK, C_THIRD, INK2]):
        m = PegasosSVM(lambda_=lam_best, n_epochs=50, random_state=SEED, **kw).fit(Xtr, yB[tr])
        sched_res[name] = dict(f1=ev.precision_recall_f1(yB[va], m.predict(Xva))[2], J=m.loss_history_[-1],
                               margin=m.margin_width_)
        ax[0].plot(range(len(m.loss_history_)), m.loss_history_, color=c, lw=1.5, label=name)
        say(f"  {name:<30} final J {m.loss_history_[-1]:.4f}  val F1 {sched_res[name]['f1']:.4f}  "
            f"margin {m.margin_width_:.4f}")
    ax[0].set(xlabel="epoch", ylabel="objective J(w)", yscale="log", title=f"Convergence, λ={lam_best:g}")
    ax[0].legend(fontsize=6)
    ep_list = [1, 2, 5, 10, 20, 50]
    ep_res = {}
    for ep in ep_list:
        m = PegasosSVM(lambda_=lam_best, n_epochs=ep, random_state=SEED).fit(Xtr, yB[tr])
        ep_res[ep] = dict(f1=ev.precision_recall_f1(yB[va], m.predict(Xva))[2], margin=m.margin_width_,
                          J=m.loss_history_[-1])
    ax[1].plot(range(len(ep_list)), [ep_res[e]["J"] for e in ep_list], "-o", color=C_OURS)
    ax[1].set(xticks=range(len(ep_list)), xticklabels=ep_list, xlabel="n_epochs",
              ylabel="final objective J", title="More epochs → lower J")
    save(fig, "svm_convergence")
    say("  epochs: " + " ".join(f"{e}:F1={v['f1']:.4f}/margin={v['margin']:.3f}" for e, v in ep_res.items()))

    section("4c. RFF-PEGASOS — γ × λ grid (D = 1000 random features)")
    gammas = [1e-3, 3e-3, 1e-2, 3e-2]
    lams_rff = [1e-4, 1e-5, 1e-6]
    grid = {}
    for g in gammas:
        for lam in lams_rff:
            m = RFFPegasosSVM(n_components=1000, gamma=g, lambda_=lam, n_epochs=30,
                              random_state=SEED).fit(Xtr, yB[tr])
            grid[(g, lam)] = ev.precision_recall_f1(yB[va], m.predict(Xva))[2]
        say(f"  γ={g:g}: " + " ".join(f"λ={l:.0e}→F1 {grid[(g, l)]:.4f}" for l in lams_rff))
    f1_max = max(grid.values())
    ok = [k for k, v in grid.items() if v >= f1_max - F1_TOL]
    g_best, lam_rff = max(ok, key=lambda k: (k[1], -k[0]))     # most regularised: largest λ, then smoothest γ
    say(f"  -> chosen γ={g_best:g}, λ={lam_rff:g} (most regularised within {F1_TOL} of best val F1 {f1_max:.4f})")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    M = np.array([[grid[(g, l)] for l in lams_rff] for g in gammas])
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    for i in range(len(gammas)):
        for j in range(len(lams_rff)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.min() + 0.6 * (M.max() - M.min()) else INK)
    ax.set(xticks=range(len(lams_rff)), xticklabels=[f"{l:.0e}" for l in lams_rff], xlabel="λ",
           yticks=range(len(gammas)), yticklabels=[f"{g:g}" for g in gammas], ylabel="γ (RBF bandwidth)",
           title="RFF-SVM validation F1")
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    save(fig, "svm_rff_grid")

    RESULTS["svm_tuning"] = {
        "lambda_sweep": {f"{k:g}": v for k, v in rows.items()},
        "schedules": sched_res, "epochs": ep_res,
        "rff_grid": {f"gamma={g:g},lambda={l:g}": v for (g, l), v in grid.items()},
        "linear_best": {"lambda": lam_best}, "rff_best": {"gamma": g_best, "lambda": lam_rff},
    }
    write_tex_table("svm_lambda", ["λ", "val F1", "val AUC", "‖w‖", "2/‖w‖", "SV %", "J"], [
        [f"{l:.0e}", f"{r['f1']:.3f}", f"{r['auc']:.3f}", f"{r['norm_w']:.2f}", f"{r['margin']:.3f}",
         f"{100 * r['sv_frac']:.1f}", f"{r['J']:.3f}"] for l, r in rows.items()])
    return {"lambda": lam_best, "rff_gamma": g_best, "rff_lambda": lam_rff}


def step_margin_figure(D, lams=(1.0, 1e-2, 1e-4)):
    """2-D illustration: the SVM on two standardized features, margin band and SVs visible."""
    X, tr, yB, names = D["X"], D["tr"], D["y_tier"], D["names"]
    f1, f2 = names.index("area_m2"), names.index("dist_center_km")
    rng = np.random.default_rng(SEED)
    sub = rng.choice(tr, 600, replace=False)
    X2 = np.log(X[sub][:, [f1, f2]] + 0.5)
    X2 = (X2 - X2.mean(0)) / X2.std(0)
    y2 = yB[sub]
    fig, axes = plt.subplots(1, len(lams), figsize=(7.2, 2.9), sharey=True)
    gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    out = {}
    for ax, lam in zip(axes, lams):
        m = PegasosSVM(lambda_=lam, n_epochs=200, batch_size=None, random_state=SEED).fit(X2, y2)
        sv = m.support_vector_mask(X2, y2)
        Z = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contour(gx, gy, Z, levels=[-1, 0, 1], colors=[INK2, INK, INK2],
                   linestyles=["--", "-", "--"], linewidths=[1, 1.5, 1])
        for cls, c, lab in ((0, C_OURS, "standard"), (1, C_SK, "premium")):
            k = y2 == cls
            ax.scatter(X2[k & ~sv, 0], X2[k & ~sv, 1], s=8, color=c, alpha=0.35, linewidths=0)
            ax.scatter(X2[k & sv, 0], X2[k & sv, 1], s=14, facecolor=c, edgecolor=INK, linewidths=0.5,
                       label=f"{lab} (SV ringed)")
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="log area (std.)",
               title=f"λ={lam:g}\n2/‖w‖={m.margin_width_:.2f}, {sv.sum()} SVs")
        out[lam] = dict(margin=m.margin_width_, n_sv=int(sv.sum()), norm_w=m.norm_w_)
    axes[0].set_ylabel("log dist. to centre (std.)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.25)
    save(fig, "svm_margin_2d")
    RESULTS["svm_margin_2d"] = {f"{k:g}": v for k, v in out.items()}


# ================================================================================
# 5. final evaluation vs scikit-learn
# ================================================================================
def make_models(cfg, n_train):
    """Factories: name -> (fit(X, y) -> model, needs_scaling, kind)."""
    tA, tB, sv = cfg["tree"]["A"], cfg["tree"]["B"], cfg["svm"]
    lam = sv["lambda"]
    reg = {
        "Ours: DecisionTree (MSE)":
            lambda: DecisionTree("regression", "mse", **tA),
        "sklearn DecisionTreeRegressor":
            lambda: DecisionTreeRegressor(random_state=SEED, **tA),
    }
    clf = {
        f"Ours: DecisionTree ({tB['criterion']})":
            lambda: DecisionTree("classification", **tB),
        "sklearn DecisionTreeClassifier":
            lambda: DecisionTreeClassifier(random_state=SEED, **tB),
        "Ours: Pegasos linear SVM":
            lambda: PegasosSVM(lambda_=lam, n_epochs=30, random_state=SEED),
        "Ours: RFF-Pegasos SVM":
            lambda: RFFPegasosSVM(n_components=1000, gamma=sv["rff_gamma"], lambda_=sv["rff_lambda"],
                                  n_epochs=30, random_state=SEED),
        "sklearn SGDClassifier (hinge)":
            lambda: Pipeline([("scale", StandardScaler()),
                              ("sgd", SGDClassifier(loss="hinge", alpha=lam, max_iter=1000, tol=1e-6,
                                                    random_state=SEED))]),
        "sklearn SVC linear (10k subsample)":
            lambda: Pipeline([("scale", StandardScaler()),
                              ("svc", SVC(kernel="linear", C=1.0 / (sv["lambda"] * SVC_SUBSAMPLE)))]),
        "sklearn SVC RBF (10k subsample)":
            lambda: Pipeline([("scale", StandardScaler()),
                              ("svc", SVC(kernel="rbf", gamma=sv["rff_gamma"],
                                          C=1.0 / (sv["rff_lambda"] * SVC_SUBSAMPLE)))]),
    }
    return reg, clf


OURS_SCALED = ("Pegasos", "RFF")      # our SVMs get standardize() outside (sklearn's: in a Pipeline)


def fit_clf(name, factory, Xtr, ytr, rng_seed=SEED):
    """Fit one classifier, handling scaling / SVC subsampling. Returns (predict_fn, score_fn, fit_s)."""
    if "subsample" in name:
        idx = dp.stratified_split_indices(ytr, 0.0, 1 - SVC_SUBSAMPLE / len(ytr), rng_seed)[0] \
            if len(ytr) > SVC_SUBSAMPLE else np.arange(len(ytr))
        Xtr, ytr = Xtr[idx], ytr[idx]
    if any(k in name for k in OURS_SCALED):
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        tf = lambda Z: (Z - mu) / sd
    else:
        tf = lambda Z: Z
    model, t_fit = ev.timed(factory().fit, tf(Xtr), ytr)
    if hasattr(model, "decision_function"):
        score = lambda Z: model.decision_function(tf(Z))
    else:
        score = lambda Z: model.predict_proba(tf(Z))[:, 1]
    return model, (lambda Z: model.predict(tf(Z))), score, t_fit


def step_final(D, cfg):
    X, tr, va, te = D["X"], D["tr"], D["va"], D["te"]
    yA, yB = D["y_log"], D["y_tier"]
    dev = np.sort(np.r_[tr, va])
    reg, clf = make_models(cfg, len(tr))
    final = {"A": {}, "B": {}}
    models = {}

    section("5a. TASK A — test set (models trained on TRAIN, configs from VAL)")
    for name, fac in {"Baseline: predict train mean": None, **reg}.items():
        if fac is None:
            mean = yA[tr].mean()
            pred, t_fit, t_pred = np.full(len(te), mean), 0.0, 0.0
        else:
            m, t_fit = ev.timed(fac().fit, X[tr], yA[tr])
            pred, t_pred = ev.timed(m.predict, X[te])
            models[name] = m
        r = ev.regression_report(yA[te], pred)
        r["RMSE_log_boot_std"] = ev.bootstrap_std(ev.rmse, yA[te], pred)
        r.update(fit_s=t_fit, predict_s=t_pred)
        final["A"][name] = r
        final["A"][name]["_pred"] = pred
    ev.compare({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in final["A"].items()},
               keys=["RMSE_log", "MAE_log", "R2_log", "RMSE_AZN", "MAE_AZN", "R2_AZN", "MAPE", "fit_s"])

    section("5b. TASK B — test set")
    for name, fac in clf.items():
        m, predict, score, t_fit = fit_clf(name, fac, X[tr], yB[tr])
        pred, t_pred = ev.timed(predict, X[te])
        s = score(X[te])
        r = ev.classification_report(yB[te], pred, s)
        r["F1_boot_std"] = ev.bootstrap_std(lambda a, b: ev.precision_recall_f1(a, b)[2], yB[te], pred)
        r.update(fit_s=t_fit, predict_s=t_pred)
        r["_pred"], r["_score"], r["_cm"] = pred, s, ev.confusion_matrix(yB[te], pred)
        final["B"][name] = r
        models[name] = m
    ev.compare({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in final["B"].items()},
               keys=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "fit_s", "predict_s"])

    section("5c. 5-fold CV on the development set (train+val), mean ± std")
    cv = {"A": {}, "B": {}}
    for name, fac in reg.items():
        def fp(Xa, ya, Xb, yb, fac=fac):
            return ev.regression_report(yb, fac().fit(Xa, ya).predict(Xb))
        cv["A"][name] = ev.cross_validate(fp, X[dev], yA[dev], k=5, seed=SEED)
        say(f"  A | {name:<36} RMSE_log {ms(cv['A'][name]['RMSE_log'])}  R2_log {ms(cv['A'][name]['R2_log'])}"
            f"  time/fold {cv['A'][name]['time_s'][0]:.1f}s")
    for name, fac in clf.items():
        def fp(Xa, ya, Xb, yb, name=name, fac=fac):
            _, predict, score, _ = fit_clf(name, fac, Xa, ya)
            return ev.classification_report(yb, predict(Xb), score(Xb))
        cv["B"][name] = ev.cross_validate(fp, X[dev], yB[dev], k=5, seed=SEED, strata=yB[dev])
        say(f"  B | {name:<36} F1 {ms(cv['B'][name]['F1'])}  AUC {ms(cv['B'][name]['ROC_AUC'])}"
            f"  time/fold {cv['B'][name]['time_s'][0]:.1f}s")

    # results tables (the report's headline tables)
    rowsA = []
    for name, r in final["A"].items():
        c = cv["A"].get(name)
        rowsA.append([name, f"{r['RMSE_log']:.3f} ± {r['RMSE_log_boot_std']:.3f}", f"{r['MAE_log']:.3f}",
                      f"{r['R2_log']:.3f}", f"{r['RMSE_AZN'] / 1e3:,.1f}k", f"{r['MAE_AZN'] / 1e3:,.1f}k",
                      f"{r['R2_AZN']:.3f}", ms(c["RMSE_log"]) if c else "n/a", f"{r['fit_s']:.2f}"])
    write_tex_table("results_taskA", ["Model", "RMSE log", "MAE log", "R² log", "RMSE AZN", "MAE AZN",
                                      "R² AZN", "CV RMSE log", "fit s"], rowsA)
    rowsB = []
    for name, r in final["B"].items():
        c = cv["B"][name]
        rowsB.append([name, f"{r['Precision']:.3f}", f"{r['Recall']:.3f}",
                      f"{r['F1']:.3f} ± {r['F1_boot_std']:.3f}", f"{r['ROC_AUC']:.3f}", f"{r['PR_AUC']:.3f}",
                      ms(c["F1"]), f"{r['fit_s']:.2f}", f"{r['predict_s']:.3f}"])
    write_tex_table("results_taskB", ["Model", "Prec.", "Rec.", "F1 (test)", "ROC-AUC", "PR-AUC",
                                      "CV F1", "fit s", "pred. s"], rowsB)
    # compact combined table: all models on both tasks
    rowsAll = []
    for name, r in final["A"].items():
        c = cv["A"].get(name)
        rowsAll.append([name, f"{r['RMSE_log']:.3f}", f"{r['R2_log']:.3f}",
                        ms(c["RMSE_log"]) if c else "n/a", "—", "—", "—"])
    for name, r in final["B"].items():
        rowsAll.append([name, "—", "—", "—", f"{r['F1']:.3f}", f"{r['ROC_AUC']:.3f}", ms(cv["B"][name]["F1"])])
    write_tex_table("results_all", ["Model", "A: RMSE log", "A: R² log", "A: CV RMSE",
                                    "B: F1", "B: AUC", "B: CV F1"], rowsAll)

    RESULTS["final"] = {t: {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                for kk, vv in v.items() if kk not in ("_pred", "_score")}
                            for k, v in final[t].items()} for t in final}
    RESULTS["cv"] = cv
    return final, models


def step_structure(D, cfg, models):
    section("5d. STRUCTURE: our tree vs sklearn (matched hyperparameters)")
    X, te, names = D["X"], D["te"], D["names"]
    tA, tB = cfg["tree"]["A"], cfg["tree"]["B"]
    pairs = [("A", "Ours: DecisionTree (MSE)", "sklearn DecisionTreeRegressor"),
             ("B", f"Ours: DecisionTree ({tB['criterion']})", "sklearn DecisionTreeClassifier")]
    out, rows = {}, []
    for task, a, b in pairs:
        o, s = models[a], models[b]
        po, ps = o.predict(X[te]), s.predict(X[te])
        same = float(np.mean(np.isclose(po, ps))) if task == "A" else float(np.mean(po == ps))
        so = o.splits()
        # first splits in sklearn order (pre-order)
        t = s.tree_
        ss, stack = [], [(0, 0)]
        while stack:
            n, d = stack.pop()
            if t.children_left[n] == -1:
                continue
            ss.append((d, int(t.feature[n]), float(t.threshold[n])))
            stack += [(t.children_right[n], d + 1), (t.children_left[n], d + 1)]
        top_same = sum(1 for x, y in zip(so[:7], ss[:7]) if x[1] == y[1] and abs(x[2] - y[2]) < 1e-3 * max(1, abs(x[2])))
        out[task] = dict(ours_depth=o.get_depth(), sk_depth=int(s.get_depth()),
                         ours_leaves=o.get_n_leaves(), sk_leaves=int(s.get_n_leaves()),
                         identical_test_predictions=same, first7_splits_identical=top_same,
                         root_ours=(names[so[0][1]], so[0][2]), root_sk=(names[ss[0][1]], ss[0][2]))
        say(f"  Task {task}: depth {o.get_depth()} vs {s.get_depth()}, leaves {o.get_n_leaves()} vs "
            f"{s.get_n_leaves()}, identical test predictions {same:.2%}, first 7 splits identical {top_same}/7")
        say(f"           root split ours: {names[so[0][1]]} <= {so[0][2]:.4f} | sklearn: "
            f"{names[ss[0][1]]} <= {ss[0][2]:.4f}")
        rows.append([f"Task {task}", f"{o.get_depth()} / {s.get_depth()}",
                     f"{o.get_n_leaves()} / {s.get_n_leaves()}", f"{top_same}/7", f"{100 * same:.1f}%"])
    write_tex_table("tree_structure", ["", "depth (ours/sk)", "leaves (ours/sk)", "first 7 splits equal",
                                       "same test pred."], rows)
    RESULTS["tree_structure"] = out


SCALING_STYLE = {"Ours: tree (Task A)": (C_OURS, "-"), "sklearn tree (Task A)": (C_SK, "-"),
                 "Ours: Pegasos SVM": (C_OURS, "--"), "sklearn SGDClassifier": (C_SK, "--"),
                 "sklearn SVC (RBF)": (C_THIRD, ":")}


def _plain_log_axis(axis):
    """Log axis with plain major labels only (no overlapping minor labels)."""
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1000:g}k" if v >= 1000 else (f"{v:,.0f}" if v >= 1 else f"{v:g}")))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_scaling(times):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name, (c, ls) in SCALING_STYLE.items():
        ns = sorted(int(n) for n in times[name])
        ax.plot(ns, [times[name][n] if n in times[name] else times[name][str(n)] for n in ns], ls,
                marker="o", color=c, label=name)
    ax.set(xscale="log", yscale="log", xlabel="training rows", ylabel="fit time (s)",
           title="Fit time vs training size")
    ax.set_xticks([2_500, 5_000, 10_000, 20_000, 40_000])
    _plain_log_axis(ax.xaxis)
    _plain_log_axis(ax.yaxis)
    ax.legend(fontsize=6)
    save(fig, "scaling_fit_time")


def step_scaling(D, cfg):
    section("5e. SCALING: fit time vs training-set size")
    X, tr, yA, yB = D["X"], D["tr"], D["y_log"], D["y_tier"]
    rng = np.random.default_rng(SEED)
    sizes = [2_500, 5_000, 10_000, 20_000, len(tr)]
    tA, tB, sv = cfg["tree"]["A"], cfg["tree"]["B"], cfg["svm"]
    cases = {
        "Ours: tree (Task A)": (lambda: DecisionTree("regression", "mse", **tA), "A", False, C_OURS, "-"),
        "sklearn tree (Task A)": (lambda: DecisionTreeRegressor(random_state=SEED, **tA), "A", False, C_SK, "-"),
        "Ours: Pegasos SVM": (lambda: PegasosSVM(lambda_=sv["lambda"], n_epochs=30, random_state=SEED),
                              "B", True, C_OURS, "--"),
        "sklearn SGDClassifier": (lambda: SGDClassifier(loss="hinge", alpha=sv["lambda"], max_iter=1000,
                                                        tol=1e-6, random_state=SEED), "B", True, C_SK, "--"),
        "sklearn SVC (RBF)": (lambda: SVC(kernel="rbf", gamma=sv["rff_gamma"], C=1.0), "B", True, C_THIRD, ":"),
    }
    times = {k: {} for k in cases}
    perm = rng.permutation(tr)
    for n in sizes:
        idx = perm[:n]
        Xs = dp.standardize(X[idx])
        for name, (fac, task, scaled, _, _) in cases.items():
            if "SVC" in name and n > 20_000:
                continue                                    # O(n^2) memory/time; skip the largest size
            y = yA[idx] if task == "A" else yB[idx]
            _, t = ev.timed(fac().fit, Xs if scaled else X[idx], y)
            times[name][n] = t
        say(f"  n={n:>6}: " + "  ".join(f"{k}={v[n]:.2f}s" for k, v in times.items() if n in v))
    plot_scaling(times)
    RESULTS["scaling"] = times


# ================================================================================
# 6. error analysis
# ================================================================================
def step_errors(D, cfg, final, models, df):
    section("6. ERROR ANALYSIS")
    X, tr, te, names = D["X"], D["tr"], D["te"], D["names"]
    yA, yB, price, thr = D["y_log"], D["y_tier"], D["price"], D["thr"]
    tB = cfg["tree"]["B"]
    oursA = models["Ours: DecisionTree (MSE)"]
    skA = models["sklearn DecisionTreeRegressor"]
    oursB_name = f"Ours: DecisionTree ({tB['criterion']})"
    svm = models["Ours: Pegasos linear SVM"]

    # --- feature importances (ours vs sklearn) -------------------------------
    imp_o, imp_s = oursA.feature_importances_, skA.feature_importances_
    top = np.argsort(imp_o)[::-1][:12][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    yy = np.arange(len(top))
    ax[0].barh(yy + 0.2, imp_o[top], height=0.38, color=C_OURS, label="ours")
    ax[0].barh(yy - 0.2, imp_s[top], height=0.38, color=C_SK, label="sklearn")
    ax[0].set(yticks=yy, yticklabels=[names[i] for i in top], xlabel="impurity-decrease importance",
              title="Task A tree: feature importance")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].grid(axis="y", visible=False)
    w = svm.w
    topw = np.argsort(np.abs(w))[::-1][:12][::-1]
    ax[1].barh(np.arange(len(topw)), w[topw], height=0.7,
               color=[C_SK if v > 0 else C_OURS for v in w[topw]])
    ax[1].axvline(0, color=INK, lw=0.8)
    ax[1].set(yticks=np.arange(len(topw)), yticklabels=[names[i] for i in topw],
              xlabel="weight (standardized features)", title="Linear SVM: largest |w|  (+ → premium)")
    ax[1].grid(axis="y", visible=False)
    save(fig, "importance_and_weights")
    RESULTS["importances_taskA"] = {names[i]: float(imp_o[i]) for i in np.argsort(imp_o)[::-1][:15]}
    impB = models[oursB_name].feature_importances_
    RESULTS["importances_taskB"] = {names[i]: float(impB[i]) for i in np.argsort(impB)[::-1][:15]}
    RESULTS["svm_weights_top"] = {names[i]: float(w[i]) for i in np.argsort(np.abs(w))[::-1][:15]}
    say("  Task A top importances: " + ", ".join(f"{k} {v:.3f}" for k, v in list(RESULTS["importances_taskA"].items())[:6]))
    say("  SVM largest weights:    " + ", ".join(f"{k} {v:+.3f}" for k, v in list(RESULTS["svm_weights_top"].items())[:6]))

    # --- ROC curves + confusion matrices ---------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    for name, c, ls in ((oursB_name, C_OURS, "-"), ("Ours: Pegasos linear SVM", C_SK, "-"),
                        ("Ours: RFF-Pegasos SVM", C_THIRD, "-"), ("sklearn SVC RBF (10k subsample)", C_FOURTH, "--")):
        fpr, tpr = ev.roc_curve(yB[te], final["B"][name]["_score"])
        ax[0].plot(fpr, tpr, ls, color=c, lw=1.5,
                   label=f"{name.replace('Ours: ', '').replace(' (10k subsample)', '')} "
                         f"({final['B'][name]['ROC_AUC']:.3f})")
    ax[0].plot([0, 1], [0, 1], color=GRID, lw=1)
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (test)")
    ax[0].legend(fontsize=5.5, loc="lower right")
    for a, name in zip(ax[1:], (oursB_name, "Ours: Pegasos linear SVM")):
        cm = final["B"][name]["_cm"]
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                       color="white" if cm[i, j] > cm.max() * 0.6 else INK)
        a.set(xticks=[0, 1], xticklabels=["std", "prem"], yticks=[0, 1], yticklabels=["std", "prem"],
              xlabel="predicted", ylabel="true", title=name.replace("Ours: ", ""))
        a.grid(False)
    save(fig, "roc_confusion")

    # --- regression residuals: by price decile, category, district --------------
    res = final["A"]["Ours: DecisionTree (MSE)"]["_pred"] - yA[te]
    dte = df.iloc[te].reset_index(drop=True)
    dec = np.clip((np.searchsorted(np.quantile(price[tr], np.linspace(0, 1, 11)[1:-1]), price[te])), 0, 9)
    by_dec = [(np.mean(np.abs(res[dec == k])), np.mean(res[dec == k])) for k in range(10)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1.2, 0.8]})
    ax[0].bar(np.arange(10), [b[1] for b in by_dec], color=[C_SK if b[1] > 0 else C_OURS for b in by_dec], width=0.7)
    ax[0].axhline(0, color=INK, lw=0.8)
    ax[0].set(xticks=range(10), xticklabels=range(1, 11), xlabel="true-price decile (1 = cheapest)",
              ylabel="mean residual (log)", title="Shrinkage to the middle")
    g = dte.assign(abs_err=np.abs(res)).groupby("district").agg(mae=("abs_err", "mean"), n=("abs_err", "size"))
    dtr = df.iloc[tr]
    g["n_train"] = dtr["district"].value_counts().reindex(g.index).fillna(0)
    g["house_share"] = (dtr["category"] == "house").groupby(dtr["district"]).mean().reindex(g.index)
    g = g[g["n"] >= 10]                                     # too few test rows to estimate an MAE
    r_house = float(np.corrcoef(g["mae"], g["house_share"])[0, 1])
    r_size = float(np.corrcoef(g["mae"], np.log(g["n_train"]))[0, 1])
    ax[1].scatter(100 * g["house_share"], g["mae"], s=18, color=C_OURS, edgecolor="white", linewidths=0.8)
    for d, r in g.iterrows():
        if r["mae"] > 0.15:
            ax[1].annotate(d.replace(" r.", ""), (100 * r["house_share"], r["mae"]), fontsize=6, color=INK2,
                           xytext=(3, -8 if d.startswith("Qarada") else 2), textcoords="offset points")
    ax[1].set(xlabel="houses among district listings (%)", ylabel="test MAE (log)",
              title=f"District error tracks house share (r={r_house:.2f})")
    gc = dte.assign(abs_err=np.abs(res)).groupby("category")["abs_err"].mean()
    ax[2].bar(range(len(gc)), gc.values, color=C_OURS, width=0.6)
    ax[2].set(xticks=range(len(gc)), xticklabels=[c.replace("_", "\n") for c in gc.index],
              ylabel="test MAE (log)", title="By category")
    save(fig, "errors_regression")
    RESULTS["errors_by_decile"] = {f"decile_{k + 1}": {"mae_log": float(a), "bias_log": float(b)}
                                   for k, (a, b) in enumerate(by_dec)}
    RESULTS["errors_by_district"] = g.round(4).to_dict(orient="index")
    RESULTS["errors_by_category"] = gc.round(4).to_dict()
    RESULTS["district_mae_corr"] = {"house_share": r_house, "log_n_train": r_size}
    say(f"  district MAE vs house share r={r_house:.2f}; vs log(train size) r={r_size:.2f}")
    say("  residual bias by price decile (log): " + " ".join(f"{b[1]:+.2f}" for b in by_dec))
    say("  MAE(log) by category: " + ", ".join(f"{k} {v:.3f}" for k, v in gc.items()))

    # --- classification errors vs distance from the tier threshold ---------------
    dist = np.abs(np.log(price[te]) - np.log(thr))
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 5]
    lab = ["<5%", "5–10%", "10–20%", "20–30%", "30–50%", ">50%"]
    k = np.digitize(dist, bins[1:-1])
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    w_ = 0.27
    err_rates = {}
    for off, (name, c) in zip((-w_, 0, w_), ((oursB_name, C_OURS), ("Ours: Pegasos linear SVM", C_SK),
                                             ("Ours: RFF-Pegasos SVM", C_THIRD))):
        e = final["B"][name]["_pred"] != yB[te]
        rates = [e[k == i].mean() for i in range(len(lab))]
        err_rates[name] = rates
        ax.bar(np.arange(len(lab)) + off, 100 * np.array(rates), width=w_ - 0.03, color=c,
               label=name.replace("Ours: ", ""))
    ax.set(xticks=range(len(lab)), xticklabels=lab, xlabel="|log price − log threshold| (≈ % from threshold)",
           ylabel="test error rate (%)", title="Tier errors concentrate near the threshold")
    ax.legend(fontsize=6)
    save(fig, "errors_classification_threshold")
    share_near = float(np.mean(dist[final["B"]["Ours: Pegasos linear SVM"]["_pred"] != yB[te]] < 0.2))
    RESULTS["tier_error_rate_by_distance"] = {"bins": lab, **err_rates}
    RESULTS["svm_errors_within_20pct_of_threshold"] = share_near
    say(f"  share of SVM tier errors within ~20% of the threshold price: {share_near:.1%}")

    # --- largest regression errors (no personal data: only listing attributes) ---
    big = np.argsort(-np.abs(np.exp(final["A"]["Ours: DecisionTree (MSE)"]["_pred"]) - price[te]))[:8]
    rows = []
    for i in big:
        r = dte.iloc[i]
        rows.append([r["category"].replace("_", " "), r["district"], f"{r['area_m2']:.0f}",
                     f"{r['price'] / 1e3:,.0f}k",
                     f"{np.exp(final['A']['Ours: DecisionTree (MSE)']['_pred'][i]) / 1e3:,.0f}k"])
    write_tex_table("worst_errors", ["category", "district", "m²", "true", "pred."], rows, "llrrr")
    RESULTS["worst_errors"] = rows
    svm_train_sv = float(svm.support_vector_mask(dp.standardize(X[tr]), yB[tr]).mean())
    RESULTS["final_svm"] = dict(norm_w=svm.norm_w_, margin=svm.margin_width_, sv_frac_train=svm_train_sv,
                                bias=svm.b, final_J=svm.loss_history_[-1])
    say(f"  final linear SVM: ‖w‖={svm.norm_w_:.3f}, margin 2/‖w‖={svm.margin_width_:.4f}, "
        f"support vectors {svm_train_sv:.1%} of train")


def plot_learning_curves(curves, depth):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    titles = {"tree_shallow": "Tree, max_depth=6", "tree": f"Tree, tuned (depth {dlabel(depth)})",
              "svm": "Linear SVM"}
    for a, key in zip(ax, ("tree_shallow", "tree", "svm")):
        c = np.array(curves[key], dtype=float)
        a.plot(c[:, 0], 100 * c[:, 1], "--o", color=C_OURS, lw=1.5, label="train")
        a.plot(c[:, 0], 100 * c[:, 2], "-o", color=C_SK, label="validation")
        a.set(xscale="log", xlabel="training rows", title=titles[key])
        a.set_xticks([2_000, 5_000, 10_000, 20_000, 40_000])
        _plain_log_axis(a.xaxis)
    ax[0].set_ylabel("error rate (%)")
    ax[0].legend(fontsize=7)
    save(fig, "learning_curves")


def step_learning_curves(D, cfg):
    section("6b. LEARNING CURVES (train vs validation)")
    X, tr, va, yB = D["X"], D["tr"], D["va"], D["y_tier"]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(tr)
    fracs = [0.05, 0.1, 0.25, 0.5, 1.0]
    tB, sv = cfg["tree"]["B"], cfg["svm"]
    curves = {"tree": [], "svm": [], "tree_shallow": []}
    for f in fracs:
        idx = perm[: int(f * len(tr))]
        for key, fac, scaled in (
                ("tree", lambda: DecisionTree("classification", **tB), False),
                ("tree_shallow", lambda: DecisionTree("classification", tB["criterion"], max_depth=6), False),
                ("svm", lambda: PegasosSVM(lambda_=sv["lambda"], n_epochs=30, random_state=SEED), True)):
            if scaled:
                a, b = dp.standardize(X[idx], X[va])
            else:
                a, b = X[idx], X[va]
            m = fac().fit(a, yB[idx])
            curves[key].append((len(idx), 1 - ev.accuracy(yB[idx], m.predict(a)), 1 - ev.accuracy(yB[va], m.predict(b))))
    plot_learning_curves(curves, tB["max_depth"])
    RESULTS["learning_curves"] = curves
    for key in curves:
        say(f"  {key:<16} " + "  ".join(f"n={n}: tr {100 * a:.1f}% / val {100 * b:.1f}%" for n, a, b in curves[key]))


# ================================================================================
def numbers_for_report(cfg):
    R = RESULTS
    fA, fB = R["final"]["A"], R["final"]["B"]
    tB = cfg["tree"]["B"]
    oursB = f"Ours: DecisionTree ({tB['criterion']})"
    m = {
        "NRaw": f"{R['n_raw']:,}", "NClean": f"{R['n_clean']:,}", "NFeat": R["n_features"],
        "NTrain": f"{R['split_sizes']['train']:,}", "NVal": f"{R['split_sizes']['val']:,}",
        "NTest": f"{R['split_sizes']['test']:,}", "TierThr": f"{R['tier_threshold']:,.0f}",
        "TreeADepth": dlabel(cfg["tree"]["A"]["max_depth"]), "TreeALeaf": cfg["tree"]["A"]["min_samples_leaf"],
        "TreeBCrit": tB["criterion"], "TreeBDepth": dlabel(tB["max_depth"]), "TreeBLeaf": tB["min_samples_leaf"],
        "SvmLam": f"{cfg['svm']['lambda']:g}", "RffGamma": f"{cfg['svm']['rff_gamma']:g}",
        "RffLam": f"{cfg['svm']['rff_lambda']:g}",
        "OursARmse": f"{fA['Ours: DecisionTree (MSE)']['RMSE_log']:.3f}",
        "SkARmse": f"{fA['sklearn DecisionTreeRegressor']['RMSE_log']:.3f}",
        "OursARsq": f"{fA['Ours: DecisionTree (MSE)']['R2_log']:.3f}",
        "OursAMape": f"{100 * fA['Ours: DecisionTree (MSE)']['MAPE']:.1f}",
        "OursAMae": f"{fA['Ours: DecisionTree (MSE)']['MAE_AZN'] / 1e3:,.1f}",
        "BaseARmse": f"{fA['Baseline: predict train mean']['RMSE_log']:.3f}",
        "OursBTreeFone": f"{fB[oursB]['F1']:.3f}",
        "SkBTreeFone": f"{fB['sklearn DecisionTreeClassifier']['F1']:.3f}",
        "OursSvmFone": f"{fB['Ours: Pegasos linear SVM']['F1']:.3f}",
        "OursSvmAuc": f"{fB['Ours: Pegasos linear SVM']['ROC_AUC']:.3f}",
        "OursRffFone": f"{fB['Ours: RFF-Pegasos SVM']['F1']:.3f}",
        "OursRffAuc": f"{fB['Ours: RFF-Pegasos SVM']['ROC_AUC']:.3f}",
        "SgdFone": f"{fB['sklearn SGDClassifier (hinge)']['F1']:.3f}",
        "SvcLinFone": f"{fB['sklearn SVC linear (10k subsample)']['F1']:.3f}",
        "SvcRbfFone": f"{fB['sklearn SVC RBF (10k subsample)']['F1']:.3f}",
        "OursTreeAFit": f"{fA['Ours: DecisionTree (MSE)']['fit_s']:.1f}",
        "SkTreeAFit": f"{fA['sklearn DecisionTreeRegressor']['fit_s']:.2f}",
        "OursSvmFit": f"{fB['Ours: Pegasos linear SVM']['fit_s']:.1f}",
        "SvcRbfFit": f"{fB['sklearn SVC RBF (10k subsample)']['fit_s']:.1f}",
        "SvmNormW": f"{R['final_svm']['norm_w']:.3f}", "SvmMargin": f"{R['final_svm']['margin']:.3f}",
        "SvmSvFrac": f"{100 * R['final_svm']['sv_frac_train']:.1f}",
        "SvmErrNear": f"{100 * R['svm_errors_within_20pct_of_threshold']:.0f}",
        "DistHouseR": f"{R['district_mae_corr']['house_share']:.2f}",
        "DistSizeR": f"{R['district_mae_corr']['log_n_train']:.2f}",
        "SameA": f"{100 * R['tree_structure']['A']['identical_test_predictions']:.1f}",
        "SameB": f"{100 * R['tree_structure']['B']['identical_test_predictions']:.1f}",
        "GiniBest": f"{R['gini_vs_entropy_best_val_f1']['gini']:.4f}",
        "EntropyBest": f"{R['gini_vs_entropy_best_val_f1']['entropy']:.4f}",
    }
    write_tex_macros(m)
    return m


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    np.random.seed(SEED)
    for d in (FIG_DIR, TAB_DIR, RES_DIR):
        os.makedirs(d, exist_ok=True)
    t0 = time.time()

    df = step_data()
    D = step_split(df)
    step_eda(df, D["thr"])
    cfg = {"tree": step_tune_tree(D), "svm": step_tune_svm(D)}
    step_margin_figure(D)
    RESULTS["final_configs"] = {"tree": cfg["tree"], "svm": cfg["svm"]}
    section("FINAL CONFIGURATIONS (chosen on validation only)")
    say(json.dumps(RESULTS["final_configs"], indent=2, default=str))
    final, models = step_final(D, cfg)
    step_structure(D, cfg, models)
    step_scaling(D, cfg)
    step_errors(D, cfg, final, models, df)
    step_learning_curves(D, cfg)
    nums = numbers_for_report(cfg)

    with open(os.path.join(RES_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                  ensure_ascii=False)
    section("HEADLINE NUMBERS (test set)")
    say(f"  Task A  our tree RMSE(log) {nums['OursARmse']} (sklearn {nums['SkARmse']}, mean-baseline "
        f"{nums['BaseARmse']}), R²(log) {nums['OursARsq']}, MAPE {nums['OursAMape']}%")
    say(f"  Task B  our tree F1 {nums['OursBTreeFone']} (sklearn {nums['SkBTreeFone']}) | our linear SVM F1 "
        f"{nums['OursSvmFone']} (SGD {nums['SgdFone']}, SVC-lin {nums['SvcLinFone']}) | our RFF-SVM F1 "
        f"{nums['OursRffFone']} (SVC-RBF {nums['SvcRbfFone']})")
    say(f"\nDone in {time.time() - t0:.0f}s. Figures -> {FIG_DIR}/, tables -> {TAB_DIR}/, metrics -> {RES_DIR}/")


if __name__ == "__main__":
    main()
