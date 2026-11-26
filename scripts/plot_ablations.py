# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Readable table + plots for a run written by scripts/ablate_uncertain_push.py.

Everything is recomputed from the raw episodes.csv (one row per ablation x seed), not from summary.csv.

Usage (from the repo root). matplotlib is not a judo dependency, so run this in a throwaway pixi env; it only reads
CSV/JSON and never imports judo:
    pixi exec -s python -s numpy -s matplotlib -s rich python scripts/plot_ablations.py                # newest run
    pixi exec -s python -s numpy -s matplotlib -s rich python scripts/plot_ablations.py --run out/ablations/<dir>

Writes, into the run folder:
    summary.md                      the table below, as GitHub markdown
    figures/success.png             success rate per config with 95% CIs (grouped bars for a grid, sorted bars otherwise)
    figures/main_effects.png        success rate per value of each parameter
    figures/seed_outcomes.png       success / fail / never-touched for every (config, seed)
    figures/secondary_metrics.png   first-contact time, final belief error, planning time (+ identify time and final
                                    angle error for the CoM task)
    figures/baselines*.png          stage 3 only: success per baseline, overall and split by prior error
    figures/belief_error.png        stage 3 only: belief error over time per baseline (from belief_error.json)

Works for both tasks of the runner (run_config["task_key"]: "position" or "com"). Every config is paired against
"baseline", or against "ours" in a run without one (CoM stage 3).

Error bars on success rates are 95% Wilson intervals: with n=20 the usual p +- 1.96*SE leaves [0, 1] near p=1 and
collapses to zero width at p=1, while Wilson stays inside [0, 1]. Continuous metrics use mean +- standard error.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_PLAN_BUDGET_MS = 50.0  # the live app plans at 20 Hz

# Colours (light theme), checked with a colour-vision-deficiency validator.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # one hue, light -> dark, for ordered values

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "font.size": 9,
        "legend.frameon": False,
    }
)
ERR = {"ecolor": INK_2, "elinewidth": 1.0, "capsize": 3}
BAR = {"edgecolor": SURFACE, "linewidth": 1.5}  # thin surface-coloured gap between adjacent bars


# ============================================================================================================== #
# Loading and statistics                                                                                         #
# ============================================================================================================== #


def short(key: str) -> str:
    """'task.w_info' -> 'w_info'."""
    return key.split(".", 1)[1]


def fmt(value: float) -> str:
    """Format a parameter value compactly: 10.0 -> '10', 0.1 -> '0.1'."""
    return f"{value:g}"


CONTINUOUS_METRICS = (
    "first_contact_time",
    "final_belief_err",
    "mean_plan_ms",
    "identify_time",
    "final_pos_err",
    "final_yaw_err",
)


def task_key(run_config: dict) -> str:
    """'position' or 'com' (runs from before the CoM task have no task_key and are position runs)."""
    return run_config.get("task_key", "position")


def reference_name(configs: list[dict]) -> str:
    """The config every other one is paired against: "baseline", or "ours" in a run without one, else the first."""
    names = [c["name"] for c in configs]
    return next((n for n in ("baseline", "ours") if n in names), names[0])


def load_run(run_dir: Path) -> tuple[list[dict], list[str], dict]:
    """Return (configs, swept parameter keys, run_config).

    Each config holds its overrides, the full value of every swept parameter (baseline value where not overridden)
    and per-seed arrays of the episode metrics.
    """
    run_config = json.loads((run_dir / "run_config.json").read_text())
    ablations: dict[str, dict] = dict(run_config["ablations"])
    baseline: dict = run_config["baseline"]
    params = list(dict.fromkeys(k for overrides in ablations.values() for k in overrides))

    with open(run_dir / "episodes.csv") as f:
        rows = list(csv.DictReader(f))

    configs = []
    for name, overrides in ablations.items():
        eps = sorted((r for r in rows if r["ablation"] == name), key=lambda r: int(r["seed"]))
        if not eps:
            continue
        configs.append(
            {
                "name": name,
                "label": "baseline" if name == "baseline" else name.replace(",", ", "),
                "overrides": overrides,
                "values": {k: overrides.get(k, baseline.get(k)) for k in params},
                "order": len(configs),  # position in run_config["ablations"] (the ladder order for stage 3)
                "seed": np.array([int(r["seed"]) for r in eps]),
                "success": np.array([r["success"] == "True" for r in eps]),
                **{
                    m: np.array([float(r[m]) for r in eps])
                    for m in (*CONTINUOUS_METRICS, "prior_err")
                    if m in eps[0]  # newer metrics are missing from older runs
                },
            }
        )
    return configs, params, run_config


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, center - half), min(1.0, center + half)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    """Mean and standard error, ignoring NaN (e.g. first-contact time of episodes that never touched)."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def summarize(configs: list[dict]) -> list[dict]:
    """Add summary statistics to each config and return them sorted best first."""
    ref = reference_name(configs)
    base = next(c for c in configs if c["name"] == ref)
    base_success = dict(zip(base["seed"], base["success"], strict=True))
    for c in configs:
        k, n = int(c["success"].sum()), len(c["success"])
        paired = [(s, base_success[s]) for s, s_ok in zip(c["seed"], c["success"], strict=True) if s in base_success]
        c["rate"] = k / n
        c["ci"] = wilson(k, n)
        c["delta"] = c["rate"] - base["success"].mean()
        c["wins"] = sum(ok and not b for (_, b), ok in zip(paired, c["success"], strict=True))
        c["losses"] = sum(b and not ok for (_, b), ok in zip(paired, c["success"], strict=True))
        c["never_touched"] = int(np.isnan(c["first_contact_time"]).sum())
        c["n"] = n
        c["is_ref"] = c["name"] == ref
    return sorted(configs, key=lambda c: (-c["rate"], -(c["wins"] - c["losses"])))


def grid_axes(configs: list[dict], params: list[str]) -> dict[str, list] | None:
    """If the configs form a full grid over 2-3 parameters, return {key: sorted values}; otherwise None."""
    if not 2 <= len(params) <= 3:
        return None
    axes = {k: sorted({c["values"][k] for c in configs}) for k in params}
    cells = {tuple(c["values"][k] for k in params) for c in configs}
    return axes if cells == set(itertools.product(*axes.values())) and len(cells) == len(configs) else None


# ============================================================================================================== #
# Table                                                                                                          #
# ============================================================================================================== #


def table_rows(configs: list[dict], params: list[str], baseline: dict, grid: bool) -> tuple[list[str], list[list]]:
    """Header and cells shared by the terminal and markdown tables. Each cell is (text, is_default_value)."""
    header = ([short(k) for k in params] if grid else ["config"]) + [
        "success",
        "95% CI",
        "Δ pts",
        "W/L",
        "no touch",
        "contact s",
        "belief m",
        "plan ms",
    ]
    rows = []
    for c in configs:
        if grid:
            cells = [(fmt(c["values"][k]), c["values"][k] == baseline[k]) for k in params]
        else:
            cells = [(c["label"], False)]
        lo, hi = c["ci"]
        contact, belief, plan = (mean_se(c[m])[0] for m in ("first_contact_time", "final_belief_err", "mean_plan_ms"))
        cells += [
            (f"{c['rate']:.0%}", False),
            (f"{100 * lo:.0f}–{100 * hi:.0f}%", False),
            ("—" if c["is_ref"] else f"{100 * c['delta']:+.0f}", False),
            ("—" if c["is_ref"] else f"{c['wins']}/{c['losses']}", False),
            (f"{c['never_touched']}/{c['n']}", False),
            (f"{contact:.2f}", False),
            (f"{belief:.3f}", False),
            (f"{plan:.0f}", False),
        ]
        rows.append(cells)
    return header, rows


def print_table(configs: list[dict], header: list[str], rows: list[list], title: str) -> None:
    """Print the table to the terminal. Parameter values equal to the baseline are dimmed."""
    table = Table(title=title, title_justify="left")
    for i, h in enumerate(header):
        widest = max(len(cells[i][0]) for cells in rows)  # values never wrap; long headers may
        table.add_column(h, justify="left" if h == "config" else "right", min_width=widest, overflow="fold")
    for c, cells in zip(configs, rows, strict=True):
        texts = [f"[dim]{t}[/dim]" if is_default else t for t, is_default in cells]
        table.add_row(*texts, style="bold yellow" if c["is_ref"] else None)
    Console().print(table)


def write_markdown(
    path: Path, configs: list[dict], header: list[str], rows: list[list], title: str, belief_what: str
) -> None:
    """Write the same table as GitHub markdown; the reference row (baseline, or ours) is bold."""
    ref = next(c["name"] for c in configs if c["is_ref"])
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c, cells in zip(configs, rows, strict=True):
        texts = [t for t, _ in cells]
        if c["is_ref"]:
            texts = [f"**{t}**" for t in texts]
        lines.append("| " + " | ".join(texts) + " |")
    lines += [
        "",
        f"- **95% CI**: Wilson interval of the success rate. **Δ pts**: difference to {ref} in percentage points.",
        f"- **W/L**: seeds this config solved and {ref} did not / {ref} solved and this config did not.",
        "- **no touch**: episodes in which the pusher never touched the cart.",
        "- **contact s**: mean time of first contact, over episodes that touched. **belief m**: mean final distance",
        f"  between belief mean and {belief_what}. Standard errors for both are in figures/secondary_metrics.png.",
        f"- **plan ms**: mean planning time per step (the live app has {LIVE_PLAN_BUDGET_MS:.0f} ms).",
    ]
    path.write_text("\n".join(lines) + "\n")


# ============================================================================================================== #
# Plots                                                                                                          #
# ============================================================================================================== #


def yerr(cs: list[dict]) -> np.ndarray:
    """Asymmetric error-bar lengths (2, n) for the success rates of the given configs."""
    return np.array([[c["rate"] - c["ci"][0], c["ci"][1] - c["rate"]] for c in cs]).T


def ramp(n: int) -> list[str]:
    """Return n colours spread over the light -> dark ramp."""
    return [RAMP[round(i)] for i in np.linspace(0, len(RAMP) - 1, n)]


def baseline_line(ax: plt.Axes, rate: float, horizontal: bool = True) -> None:
    """Dashed reference line at the baseline success rate."""
    kw = {"color": INK_2, "linestyle": "--", "linewidth": 1.0, "zorder": 3}
    (ax.axhline if horizontal else ax.axvline)(rate, **kw)


def plot_success_grid(configs: list[dict], axes: dict[str, list], baseline: dict, path: Path) -> None:
    """Grouped bars: x = 1st parameter, one panel per value of the 2nd (if 3 params), colour = last parameter."""
    keys = list(axes)
    x_key, colour_key = keys[0], keys[-1]
    panel_key = keys[1] if len(keys) == 3 else None
    panel_values = axes[panel_key] if panel_key else [None]
    colours = ramp(len(axes[colour_key]))
    base_rate = next(c["rate"] for c in configs if c["name"] == "baseline")
    lookup = {tuple(c["values"][k] for k in keys): c for c in configs}

    fig, axs = plt.subplots(1, len(panel_values), figsize=(5.2 * len(panel_values), 4.2), sharey=True, squeeze=False)
    width = 0.8 / len(axes[colour_key])
    for ax, pv in zip(axs[0], panel_values, strict=True):
        for j, (cv, colour) in enumerate(zip(axes[colour_key], colours, strict=True)):
            cs = [lookup[tuple(xv if k == x_key else pv if k == panel_key else cv for k in keys)] for xv in axes[x_key]]
            xs = np.arange(len(cs)) + (j - (len(colours) - 1) / 2) * width
            bars = ax.bar(xs, [c["rate"] for c in cs], width, color=colour, yerr=yerr(cs), error_kw=ERR, **BAR)
            for bar, c in zip(bars, cs, strict=True):
                ax.text(bar.get_x() + width / 2, c["ci"][1] + 0.015, f"{c['rate']:.0%}", ha="center", fontsize=7,
                        color=INK_2)  # fmt: skip
                if c["name"] == "baseline":
                    bar.set_hatch("////")
                    bar.set_edgecolor(INK_2)
        baseline_line(ax, base_rate)
        ax.set_xticks(np.arange(len(axes[x_key])), [fmt(v) for v in axes[x_key]])
        ax.set_xlabel(short(x_key))
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
        if panel_key:
            ax.set_title(f"{short(panel_key)} = {fmt(pv)}" + (" (default)" if pv == baseline[panel_key] else ""))
    axs[0][0].set_ylabel("success rate")
    handles = [Patch(color=c, label=fmt(v)) for v, c in zip(axes[colour_key], colours, strict=True)]
    handles += [
        Patch(facecolor="none", edgecolor=INK_2, hatch="////", label="current defaults"),
        plt.Line2D([], [], color=INK_2, linestyle="--", label=f"baseline {base_rate:.0%}"),
    ]
    fig.legend(handles=handles, title=short(colour_key), loc="upper center", ncol=len(handles), fontsize=8,
               title_fontsize=8, bbox_to_anchor=(0.5, 1.0))  # fmt: skip
    n = configs[0]["n"]
    fig.suptitle(f"Success rate per config ({n} seeds each, 95% Wilson CI)", y=1.07, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_success_bars(configs: list[dict], path: Path) -> None:
    """Horizontal bars sorted by success rate (for one-at-a-time sweeps); baseline in grey."""
    cs = configs[::-1]  # best at the top
    base_rate = next(c["rate"] for c in configs if c["name"] == "baseline")
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(cs) + 1.2))
    colours = [MUTED if c["name"] == "baseline" else BLUE for c in cs]
    ax.barh(np.arange(len(cs)), [c["rate"] for c in cs], 0.75, color=colours, xerr=yerr(cs), error_kw=ERR, **BAR)
    for i, c in enumerate(cs):
        ax.text(c["ci"][1] + 0.015, i, f"{c['rate']:.0%}", va="center", fontsize=7, color=INK_2)
    baseline_line(ax, base_rate, horizontal=False)
    ax.set_yticks(np.arange(len(cs)), [c["label"] for c in cs])
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Success rate per config ({cs[0]['n']} seeds each, 95% Wilson CI; dashed = baseline {base_rate:.0%})")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_main_effects(configs: list[dict], params: list[str], baseline: dict, grid: bool, path: Path) -> None:
    """Success rate per value of each parameter.

    Grid: pooled over every config with that value (all other parameters vary). Sweep: the configs that change only
    this parameter, plus the baseline for its default value.
    """
    ncols = min(len(params), 5)
    nrows = int(np.ceil(len(params) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), sharey=True, squeeze=False)
    for ax, key in zip(axs.flat, params, strict=False):
        if grid:
            pool = list(configs)
        else:
            pool = [c for c in configs if c["name"] == "baseline" or set(c["overrides"]) == {key}]
        values = sorted({c["values"][key] for c in pool})
        ks = [sum(int(c["success"].sum()) for c in pool if c["values"][key] == v) for v in values]
        ns = [sum(c["n"] for c in pool if c["values"][key] == v) for v in values]
        rates = np.array(ks) / np.array(ns)
        cis = [wilson(k, n) for k, n in zip(ks, ns, strict=True)]
        err = np.array([[r - lo, hi - r] for r, (lo, hi) in zip(rates, cis, strict=True)]).T
        colours = [MUTED if v == baseline[key] else BLUE for v in values]
        ax.bar(np.arange(len(values)), rates, 0.7, color=colours, yerr=err, error_kw=ERR, **BAR)
        for i, (r, (_, hi), n) in enumerate(zip(rates, cis, ns, strict=True)):
            ax.text(i, hi + 0.02, f"{r:.0%}\nn={n}", ha="center", fontsize=7, color=INK_2)
        ax.set_xticks(np.arange(len(values)), [fmt(v) for v in values])
        ax.set_title(short(key))
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
    for ax in list(axs.flat)[len(params) :]:
        ax.set_visible(False)
    for row in axs:
        row[0].set_ylabel("success rate")
    note = (
        "Pooled over all other grid parameters. Pooled episodes share seeds, so the CIs are somewhat optimistic."
        if grid
        else "Each bar is one config that changes only this parameter."
    )
    fig.suptitle(f"Success rate per parameter value (95% Wilson CI; grey = current default)\n{note}", color=INK,
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Stage 3 (baselines): readable names and roles. Labels only; every number still comes from episodes.csv.
BASELINE_LABELS = {
    "position": {
        "baseline": "defaults\n(stage 1/2 ref.)",
        "oracle": "oracle\n(knows cart)",
        "blind": "blind\n(ignores touch)",
        "certainty_equivalent": "plan vs\nbelief mean",
        "posterior_sampling": "plan vs one\nsampled guess",
        "no_info": "average guesses,\nno info term",
        "ours": "ours: average\n+ info gain",
    },
    "com": {
        "oracle": "oracle\n(knows CoM)",
        "blind": "blind\n(ignores motion)",
        "certainty_equivalent": "plan with\nmean CoM",
        "posterior_sampling": "plan with one\nsampled CoM",
        "no_info": "average CoMs,\nno info term",
        "ours": "ours: average\n+ info gain",
    },
}
# What the belief tracks, and what "prior error" means, per task (for axis labels and legends).
BELIEF_WHAT = {"position": "true cart position", "com": "true CoM"}
PRIOR_ERROR_WHAT = {"position": "prior error", "com": "CoM offset from centre"}
BOUNDS, OURS = {"oracle", "blind"}, {"ours"}


def role_colour(name: str) -> str:
    """Grey for the bounds, orange for the method, blue for the baselines in between."""
    return MUTED if name in BOUNDS else ORANGE if name in OURS else BLUE


def plot_baselines(configs: list[dict], labels: dict[str, str], path: Path) -> None:
    """Success rate per baseline in ladder order (as listed in the run), with Wilson CIs."""
    cs = sorted(configs, key=lambda c: c["order"])
    fig, ax = plt.subplots(figsize=(1.25 * len(cs) + 1.5, 4.4))
    xs = np.arange(len(cs))
    ax.bar(xs, [c["rate"] for c in cs], 0.7, color=[role_colour(c["name"]) for c in cs], yerr=yerr(cs), error_kw=ERR,
           **BAR)  # fmt: skip
    for x, c in zip(xs, cs, strict=True):
        ax.text(x, c["ci"][1] + 0.02, f"{c['rate']:.0%}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(xs, [labels.get(c["name"], c["name"]) for c in cs], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("success rate")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[Patch(color=MUTED, label="bounds"), Patch(color=BLUE, label="baselines"),
                 Patch(color=ORANGE, label="our method")],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8,
    )  # fmt: skip
    ax.set_title(f"Success rate per method ({cs[0]['n']} seeds each, 95% Wilson CI)", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baselines_by_prior_error(
    configs: list[dict], threshold: float, labels: dict[str, str], what: str, path: Path
) -> None:
    """Success split by how wrong the starting guess was (|prior centre - truth| vs the run's hard_threshold)."""
    cs = sorted(configs, key=lambda c: c["order"])
    hard = cs[0]["prior_err"] > threshold  # prior error depends only on the seed, so the split is the same for all
    groups = [("easy", ~hard, RAMP[1]), ("hard", hard, RAMP[4])]
    fig, ax = plt.subplots(figsize=(1.25 * len(cs) + 1.5, 4.4))
    xs = np.arange(len(cs))
    width = 0.38
    for j, (label, mask, colour) in enumerate(groups):
        n = int(mask.sum())
        rates, errs = [], []
        for c in cs:
            k = int(c["success"][mask].sum())
            lo, hi = wilson(k, n) if n else (np.nan, np.nan)
            rates.append(k / n if n else np.nan)
            errs.append([rates[-1] - lo, hi - rates[-1]])
        bar_x = xs + (j - 0.5) * width
        ax.bar(bar_x, rates, width, color=colour, yerr=np.array(errs).T, error_kw=ERR, **BAR,
               label=f"{label}: {what} {'>' if label == 'hard' else '≤'} {threshold:.3g} m  (n={n})")  # fmt: skip
        for x, r in zip(bar_x, rates, strict=True):
            ax.text(x, 1.03, f"{r:.0%}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(xs, [labels.get(c["name"], c["name"]) for c in cs], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("success rate")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    ax.set_title("Success by how wrong the starting guess was (95% Wilson CI)", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seed_outcomes(configs: list[dict], path: Path) -> None:
    """Heatmap of the outcome of every (config, seed): shows which seeds fail for every config."""
    seeds = sorted({int(s) for c in configs for s in c["seed"]})
    outcome = np.full((len(configs), len(seeds)), np.nan)
    for i, c in enumerate(configs):
        for s, ok, t in zip(c["seed"], c["success"], c["first_contact_time"], strict=True):
            outcome[i, seeds.index(s)] = 0 if ok else (2 if np.isnan(t) else 1)
    fig, ax = plt.subplots(figsize=(0.32 * len(seeds) + 4.5, 0.3 * len(configs) + 1.6))
    ax.pcolormesh(outcome, cmap=ListedColormap([BLUE, ORANGE, MUTED]), vmin=-0.5, vmax=2.5, edgecolors=SURFACE,
                  linewidth=1.5)  # fmt: skip
    for i, j in zip(*np.where(outcome == 2), strict=True):
        ax.text(j + 0.5, i + 0.5, "×", ha="center", va="center", color=SURFACE, fontsize=8)
    solved = (outcome == 0).sum(axis=0)
    ax.set_xticks(np.arange(len(seeds)) + 0.5, [f"{s}\n{k}/{len(configs)}" for s, k in zip(seeds, solved, strict=True)],
                  fontsize=7)  # fmt: skip
    ax.set_yticks(np.arange(len(configs)) + 0.5, [c["label"] for c in configs], fontsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("seed  (configs that solved it)")
    ax.legend(
        handles=[
            Patch(color=BLUE, label="success"),
            Patch(color=ORANGE, label="failed after touching the cart"),
            Patch(color=MUTED, label="× never touched the cart"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_secondary(configs: list[dict], task: str, path: Path) -> None:
    """First-contact time, final belief error, planning time (+ CoM-task metrics) per config (mean +- SE over seeds)."""
    cs = configs[::-1]
    metrics = [
        ("first_contact_time", "first contact (s)\nepisodes that touched"),
        ("final_belief_err", "final CoM error (m)" if task == "com" else "final belief error (m)"),
        ("identify_time", "time to identify (s)\nbelief std < info gate"),
        ("final_yaw_err", "final angle error (rad)"),
        ("mean_plan_ms", "planning time per step (ms)"),
    ]
    # keep a metric only if this run logged it and it is not NaN everywhere (e.g. no angle goal in the position task)
    metrics = [(m, t) for m, t in metrics if m in cs[0] and any(np.any(~np.isnan(c[m])) for c in cs)]
    fig, axs = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 0.28 * len(cs) + 1.4), sharey=True)
    colours = [MUTED if c["is_ref"] else BLUE for c in cs]
    for ax, (key, title) in zip(axs, metrics, strict=True):
        stats = [mean_se(c[key]) for c in cs]
        ax.barh(np.arange(len(cs)), [m for m, _ in stats], 0.75, color=colours, xerr=[se for _, se in stats],
                error_kw=ERR, **BAR)  # fmt: skip
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        if key == "mean_plan_ms":
            ax.axvline(LIVE_PLAN_BUDGET_MS, color=INK_2, linestyle="--", linewidth=1.0)
            ax.text(LIVE_PLAN_BUDGET_MS, len(cs) - 0.3, " live budget", fontsize=7, color=INK_2, va="bottom")
    axs[0].set_yticks(np.arange(len(cs)), [c["label"] for c in cs], fontsize=7)
    ref = next(c["name"] for c in cs if c["is_ref"])
    fig.suptitle(f"Secondary metrics (mean ± SE over seeds; grey = {ref})", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_belief_error(run_dir: Path, configs: list[dict], labels: dict[str, str], key: str, path: Path) -> bool:
    """Belief error over time per baseline (mean +- SE over seeds), from belief_error.json. False if not logged."""
    source = run_dir / "belief_error.json"
    if not source.exists():
        return False
    data = json.loads(source.read_text())
    cs = sorted(configs, key=lambda c: c["order"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # bounds dotted/dashed grey, ours solid orange; the baselines in between get distinct blues and dash patterns
    middle = [c["name"] for c in cs if c["name"] not in BOUNDS | OURS]
    styles = {"oracle": ":", "blind": "--", **dict(zip(middle, ["-.", (0, (6, 2)), "-", (0, (2, 1))], strict=False))}
    shades = dict(zip(middle, ramp(len(middle)), strict=True))
    for c in cs:
        traces = data["traces"].get(c["name"], {})
        if not traces:
            continue
        length = min(len(t) for t in traces.values())
        errs = np.array([t[:length] for t in traces.values()])
        t = data["plan_dt"] * np.arange(length)
        mean, se = errs.mean(0), errs.std(0, ddof=1) / np.sqrt(len(errs)) if len(errs) > 1 else np.zeros(length)
        colour = shades.get(c["name"], role_colour(c["name"]))
        ax.plot(t, mean, color=colour, linestyle=styles.get(c["name"], "-"), linewidth=1.6,
                label=labels.get(c["name"], c["name"]).replace("\n", " "))  # fmt: skip
        ax.fill_between(t, mean - se, mean + se, color=colour, alpha=0.15, linewidth=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|belief mean − " + BELIEF_WHAT[key] + "| (m)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Belief error over time (mean ± SE over {cs[0]['n']} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ============================================================================================================== #
# Main                                                                                                           #
# ============================================================================================================== #


def main() -> None:
    """Load a run, print and save the table, and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="out/ablations/<dir> (default: the newest run)")
    args = parser.parse_args()
    run_dir = args.run or max((REPO_ROOT / "out" / "ablations").iterdir(), key=lambda p: p.name)

    configs, params, run_config = load_run(run_dir)
    configs = summarize(configs)
    baseline = run_config["baseline"]
    axes = grid_axes(configs, params)
    git = run_config.get("git", {})
    title = (
        f"{run_dir.name}: {len(configs)} configs x {configs[0]['n']} seeds, {run_config['duration']:g} s episodes "
        f"(commit {git.get('commit', '?')[:7]}{', dirty' if git.get('dirty') else ''})"
    )

    key = task_key(run_config)
    labels = BASELINE_LABELS[key]
    header, rows = table_rows(configs, params, baseline, grid=axes is not None)
    print_table(configs, header, rows, title)
    write_markdown(run_dir / "summary.md", configs, header, rows, title, BELIEF_WHAT[key])

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    if run_config.get("stage") == 3:  # baselines: ladder-ordered bars instead of parameter plots
        plot_baselines(configs, labels, fig_dir / "baselines.png")
        if "prior_err" in configs[0]:
            threshold = run_config.get("hard_threshold", baseline.get("task.prior_sigma"))
            plot_baselines_by_prior_error(
                configs, threshold, labels, PRIOR_ERROR_WHAT[key], fig_dir / "baselines_by_prior_error.png"
            )
        plot_belief_error(run_dir, configs, labels, key, fig_dir / "belief_error.png")
    else:
        if axes:
            plot_success_grid(configs, axes, baseline, fig_dir / "success.png")
        else:
            plot_success_bars(configs, fig_dir / "success.png")
        plot_main_effects(configs, params, baseline, axes is not None, fig_dir / "main_effects.png")
    plot_seed_outcomes(configs, fig_dir / "seed_outcomes.png")
    plot_secondary(configs, key, fig_dir / "secondary_metrics.png")

    print(f"\nTable:   {run_dir / 'summary.md'}")
    print(f"Figures: {fig_dir}/ ({', '.join(sorted(p.name for p in fig_dir.glob('*.png')))})")


if __name__ == "__main__":
    main()
