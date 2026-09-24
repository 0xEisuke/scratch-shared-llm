import argparse
import json

STANDARD_COLOR = "#0072B2"
SHARED_FULL_COLOR = "#E69F00"
SHARED_ATTN_COLOR = "#D55E00"
GUIDE_COLOR = "0.55"


def _classify(row: dict) -> str:
    if row["attn_sharing"] == "none" and row["ffn_sharing"] == "none":
        return "standard"
    if row["attn_sharing"] == "full" and row["ffn_sharing"] == "full":
        return "shared_full_layer"
    if row["attn_sharing"] == "full" and row["ffn_sharing"] == "none":
        return "shared_attn_only"
    return "other"


def load_rows(results_path: str) -> list:
    rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_trend(standard_rows: list, x_key: str):
    """Fit val_loss = m * log10(x_key) + c over the standard models.

    Returns (m, c, r2, equiv_x_fn) where equiv_x_fn(val_loss) inverts the
    fit to answer: "what x_key value would a standard model need to reach
    this val_loss?" -- used to place a weight-sharing model's loss onto the
    standard trend in x_key terms.
    """
    import numpy as np

    log_x = np.array([__import__("math").log10(r[x_key]) for r in standard_rows])
    losses = np.array([r["val_loss"] for r in standard_rows])
    m, c = np.polyfit(log_x, losses, 1)
    pred = m * log_x + c
    ss_res = float(((losses - pred) ** 2).sum())
    ss_tot = float(((losses - losses.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def equiv_x(val_loss: float) -> float:
        return 10 ** ((val_loss - c) / m)

    return m, c, r2, equiv_x


_X_AXIS = {
    "d_model": {
        "label": "d_model", "title": "d_model", "effective_label": "effective d_model",
        "log_scale": False,
    },
    "total_params": {
        "label": "Total parameters (log scale)",
        "title": "total parameters",
        "effective_label": "effective params",
        "log_scale": True,
    },
}


def find_intersection(m1, c1, m2, c2):
    """Return (x, y) where two val_loss = m*log10(x)+c lines cross, or None if parallel."""
    if abs(m2 - m1) < 1e-12:
        return None
    log_x = (c1 - c2) / (m2 - m1)
    x = 10 ** log_x
    y = m1 * log_x + c1
    return x, y


def plot(results_path: str, output_path: str, x_key: str = "d_model", show_intersection: bool = False) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    axis = _X_AXIS[x_key]
    rows = load_rows(results_path)
    for row in rows:
        row["_kind"] = _classify(row)

    standard = sorted((r for r in rows if r["_kind"] == "standard"), key=lambda r: r[x_key])
    shared_full = [r for r in rows if r["_kind"] == "shared_full_layer"]
    shared_attn = sorted((r for r in rows if r["_kind"] == "shared_attn_only"), key=lambda r: r[x_key])

    if not standard:
        raise ValueError("No rows with attn_sharing=none, ffn_sharing=none found -- nothing to plot as the trend line")

    m, c, r2, equiv_x = fit_trend(standard, x_key)
    m2 = c2 = r2_2 = equiv_x2 = None
    if len(shared_attn) >= 2:
        m2, c2, r2_2, equiv_x2 = fit_trend(shared_attn, x_key)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    xs = [r[x_key] for r in standard]
    ys = [r["val_loss"] for r in standard]

    # Smooth fitted trend curve underneath the actual standard-model points.
    curve_x = np.linspace(min(xs) * 0.9, max(xs) * 1.05, 200)
    curve_y = m * np.log10(curve_x) + c
    ax.plot(curve_x, curve_y, color=STANDARD_COLOR, linewidth=1.5, alpha=0.5, zorder=2)
    ax.plot(
        xs, ys, marker="o", markersize=8, color=STANDARD_COLOR, linewidth=0, zorder=4,
        label="Standard (independent layers)",
    )

    # Same treatment for the attention-only-sharing series' own fitted trend.
    if m2 is not None:
        xs2 = [r[x_key] for r in shared_attn]
        curve_x2 = np.linspace(min(xs2) * 0.9, max(xs2) * 1.05, 200)
        curve_y2 = m2 * np.log10(curve_x2) + c2
        ax.plot(curve_x2, curve_y2, color=SHARED_ATTN_COLOR, linewidth=1.5, alpha=0.5, zorder=2)

    def annotate_equivalent(row, color, marker, label):
        ax.scatter(
            row[x_key], row["val_loss"], marker=marker, s=120,
            color=color, edgecolors="white", linewidths=1, label=label, zorder=5,
        )
        eq_x = equiv_x(row["val_loss"])
        # horizontal guide from the shared point to the fitted trend curve,
        # then a vertical drop to the x-axis, marking the "effective" x_key
        # value a standard model would need to match this loss.
        ax.plot([eq_x, row[x_key]], [row["val_loss"], row["val_loss"]],
                color=color, linestyle="--", linewidth=1, alpha=0.7, zorder=3)
        ax.plot([eq_x, eq_x], [ax.get_ylim()[0], row["val_loss"]],
                color=color, linestyle="--", linewidth=1, alpha=0.7, zorder=3)
        ax.annotate(
            f"{axis['effective_label']}≈{eq_x:.0f}",
            xy=(eq_x, row["val_loss"]), xytext=(4, 6), textcoords="offset points",
            fontsize=9, color=color,
        )
        return eq_x

    all_losses = ys + [r["val_loss"] for r in shared_full + shared_attn]
    pad = 0.08 * (max(all_losses) - min(all_losses) + 1e-9)
    ax.set_ylim(bottom=min(all_losses) - pad, top=max(all_losses) + pad)

    for r in shared_full:
        annotate_equivalent(r, SHARED_FULL_COLOR, "^", "Shared: full layer (attn+FFN)")
    for r in shared_attn:
        annotate_equivalent(r, SHARED_ATTN_COLOR, "s", "Shared: attention only")

    intersection = None
    if show_intersection and m2 is not None:
        intersection = find_intersection(m, c, m2, c2)
        if intersection is not None:
            ix, iy = intersection
            all_x = xs + [r[x_key] for r in shared_attn]
            x_lo, x_hi = min(all_x), max(all_x)
            # Widen the axes to fit the crossing point if it falls outside the
            # tested range, so the two trend lines visibly meet on the plot.
            plot_x_lo, plot_x_hi = min(x_lo, ix) * 0.9, max(x_hi, ix) * 1.1
            extended_x = np.linspace(plot_x_lo, plot_x_hi, 400)
            in_range = x_lo <= ix <= x_hi
            style = dict(linewidth=1.5, alpha=0.5 if in_range else 0.3, zorder=1)
            if not in_range:
                ax.plot(extended_x, m * np.log10(extended_x) + c, color=STANDARD_COLOR,
                         linestyle="--", **style)
                ax.plot(extended_x, m2 * np.log10(extended_x) + c2, color=SHARED_ATTN_COLOR,
                         linestyle="--", **style)
                ax.set_xlim(plot_x_lo, plot_x_hi)
            ax.scatter(
                [ix], [iy], marker="X", s=140, color="black", edgecolors="white",
                linewidths=1, zorder=6, label="Trend intersection",
            )
            note = "" if in_range else " (extrapolated outside tested range)"
            ax.annotate(
                f"intersection: {ix:,.0f} params, loss={iy:.4f}{note}",
                xy=(ix, iy), xytext=(-6, -16), textcoords="offset points",
                fontsize=9, color="black", ha="right",
            )
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(min(y_lo, iy - 0.02), max(y_hi, iy + 0.02))

    if axis["log_scale"]:
        ax.set_xscale("log")
    ax.set_xlabel(axis["label"])
    ax.set_ylabel("Validation cross-entropy loss")
    ax.set_title(f"Validation loss vs. {axis['title']}: standard trend vs. weight-sharing variants")
    ax.grid(True, axis="both", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")
    print(f"Standard trend fit:     val_loss = {m:.5f} * log10({x_key}) + {c:.5f} (R^2={r2:.4f}, n={len(standard)})")
    if m2 is not None:
        print(f"Shared-attn trend fit:  val_loss = {m2:.5f} * log10({x_key}) + {c2:.5f} (R^2={r2_2:.4f}, n={len(shared_attn)})")
    for r in shared_full:
        print(f"  {r['run_name']}: actual {x_key}={r[x_key]}, {axis['effective_label']} vs standard={equiv_x(r['val_loss']):.1f}")
    for r in shared_attn:
        print(f"  {r['run_name']}: actual {x_key}={r[x_key]}, {axis['effective_label']} vs standard={equiv_x(r['val_loss']):.1f}")
    if intersection is not None:
        ix, iy = intersection
        print(f"Standard/shared-attn trend intersection: {x_key}={ix:,.1f}, val_loss={iy:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot d_model (or total_params) vs val_loss for the standard models (with a "
        "fitted trend curve), overlaying weight-sharing variant points and their 'effective' value "
        "-- what a standard model would need to match that variant's loss."
    )
    parser.add_argument("--results-path", required=True, help="JSONL file written by train.py")
    parser.add_argument("--output", default="checkpoints/compare/dmodel_vs_val_loss.png")
    parser.add_argument("--x-key", choices=["d_model", "total_params"], default="d_model")
    args = parser.parse_args()

    # Marking where the standard and shared-attention trend lines cross only
    # makes sense on the total_params axis (see mdfiles/sharing_comparison_report.md).
    plot(args.results_path, args.output, args.x_key, show_intersection=args.x_key == "total_params")


if __name__ == "__main__":
    main()
