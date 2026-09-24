import argparse

from scratch_llm.plot_results import STANDARD_COLOR, _classify, fit_trend, load_rows

FFN_SWEEP_COLOR = "#009E73"

# d_model=512, attn_sharing=full FFN-unique-count sweep: run_name -> FFN
# unique count. n_layers=16 is fixed across this whole comparison track (see
# CLAUDE.md), so "none" (every layer independent) means 16 unique instances
# and "full" (all layers share one instance) means 1.
_SWEEP_RUNS = {
    "shared_attn_512": 16,
    "model_512_shared_attn_ffn8": 8,
    "model_512_shared_attn_ffn4": 4,
    "model_512_shared_attn_ffn2": 2,
    "model_512_shared_attn_ffn1": 1,
}


def load_sweep_rows(results_path: str) -> list:
    rows = load_rows(results_path)
    by_name = {r["run_name"]: r for r in rows}
    sweep = []
    for name, n_unique in _SWEEP_RUNS.items():
        if name not in by_name:
            raise ValueError(f"Missing expected run '{name}' in {results_path}")
        row = dict(by_name[name])
        row["ffn_unique"] = n_unique
        sweep.append(row)
    return sorted(sweep, key=lambda r: r["ffn_unique"])


def plot_unique_vs_loss(sweep: list, output_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = [r["ffn_unique"] for r in sweep]
    ys = [r["val_loss"] for r in sweep]

    ax.plot(xs, ys, marker="D", markersize=9, color=FFN_SWEEP_COLOR, linewidth=1.5, zorder=4)
    for r in sweep:
        ax.annotate(
            f"{r['total_params']:,}",
            xy=(r["ffn_unique"], r["val_loss"]), xytext=(0, 8), textcoords="offset points",
            fontsize=8, color=FFN_SWEEP_COLOR, ha="center",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("FFN unique count (out of 16 layers, attention shared across all 16)")
    ax.set_ylabel("Validation cross-entropy loss")
    ax.set_title("Validation loss vs. FFN unique count (d_model=512)")
    ax.grid(True, axis="both", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


def plot_totalparams_vs_loss(all_rows: list, sweep: list, output_path: str) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    for row in all_rows:
        row["_kind"] = _classify(row)
    standard = sorted((r for r in all_rows if r["_kind"] == "standard"), key=lambda r: r["total_params"])
    if not standard:
        raise ValueError("No standard (attn_sharing=none, ffn_sharing=none) rows found for the trend curve")
    m, c, r2, _ = fit_trend(standard, "total_params")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    xs_std = [r["total_params"] for r in standard]
    ys_std = [r["val_loss"] for r in standard]
    curve_x = np.linspace(min(xs_std) * 0.9, max(xs_std) * 1.05, 200)
    curve_y = m * np.log10(curve_x) + c
    ax.plot(curve_x, curve_y, color=STANDARD_COLOR, linewidth=1.5, alpha=0.5, zorder=2)
    ax.plot(
        xs_std, ys_std, marker="o", markersize=8, color=STANDARD_COLOR, linewidth=0, zorder=4,
        label="Standard (independent layers)",
    )

    xs = [r["total_params"] for r in sweep]
    ys = [r["val_loss"] for r in sweep]
    ax.plot(
        xs, ys, marker="D", markersize=9, color=FFN_SWEEP_COLOR, linewidth=1.5, zorder=4,
        label="Attn shared + FFN-unique-count sweep (d_model=512)",
    )
    for r in sweep:
        ax.annotate(
            f"N={r['ffn_unique']}",
            xy=(r["total_params"], r["val_loss"]), xytext=(6, -4), textcoords="offset points",
            fontsize=9, color=FFN_SWEEP_COLOR,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total parameters (log scale)")
    ax.set_ylabel("Validation cross-entropy loss")
    ax.set_title("Validation loss vs. total parameters: FFN-unique-count sweep vs. standard trend")
    ax.grid(True, axis="both", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")
    print(f"Standard trend fit: val_loss = {m:.5f} * log10(total_params) + {c:.5f} (R^2={r2:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the d_model=512 FFN-unique-count sweep (16/8/4/2/1, attention shared "
        "across all 16 layers): FFN-unique-count vs val_loss, and total_params vs val_loss "
        "overlaid with the standard series' fitted trend."
    )
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--unique-output", default="checkpoints/compare/ffn_unique_vs_val_loss.png")
    parser.add_argument("--params-output", default="checkpoints/compare/ffn_sweep_totalparams_vs_val_loss.png")
    args = parser.parse_args()

    all_rows = load_rows(args.results_path)
    sweep = load_sweep_rows(args.results_path)

    plot_unique_vs_loss(sweep, args.unique_output)
    plot_totalparams_vs_loss(all_rows, sweep, args.params_output)

    print("\nFFN-unique-count sweep (d_model=512, attention shared across all 16 layers):")
    for r in sweep:
        print(
            f"  N={r['ffn_unique']:2d}: params={r['total_params']:,}, "
            f"val_loss={r['val_loss']:.4f}, ppl={r['val_perplexity']:.3f}"
        )


if __name__ == "__main__":
    main()
