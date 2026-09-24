import argparse

from scratch_llm.plot_results import load_rows

WIDENING_COLOR = "#CC79A7"
GUIDE_COLOR = "0.55"

# Parameter-matched widening chain at d_model=512, attn_sharing=full, total
# FFN params held constant at 31,457,280 (= FFN unique count * 3*d_model*d_ff):
# run_name -> (FFN unique count, d_ff, batch_size actually used in training).
_WIDENING_RUNS = {
    "shared_attn_512": (16, 1280, 24),
    "model_512_shared_attn_ffn8_dff2560": (8, 2560, 24),
    "model_512_shared_attn_ffn4_dff5120": (4, 5120, 24),
    "model_512_shared_attn_ffn2_dff10240": (2, 10240, 8),
}

# ffn2_dff10240 at bs=24 overflows the RTX 4090's 24GB and falls into
# Windows' GPU-memory-oversubscription paging (see
# project_headdim128_vram_cliff memory / mdfiles/sharing_comparison_report.md);
# this is the preflight-measured peak VRAM at bs=24 for that one config,
# kept only for the "at matched batch size" reference line/annotation.
_FFN2_DFF10240_BS24_PREFLIGHT_VRAM_MB = 26010.0


def load_widening_rows(results_path: str) -> list:
    rows = load_rows(results_path)
    by_name = {r["run_name"]: r for r in rows}
    out = []
    for name, (n_unique, d_ff, bs) in _WIDENING_RUNS.items():
        if name not in by_name:
            raise ValueError(f"Missing expected run '{name}' in {results_path}")
        row = dict(by_name[name])
        row["ffn_unique"] = n_unique
        row["d_ff"] = d_ff
        row["batch_size_used"] = bs
        out.append(row)
    return sorted(out, key=lambda r: r["ffn_unique"])


def plot_val_loss(rows: list, output_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = [r["ffn_unique"] for r in rows]
    ys = [r["val_loss"] for r in rows]

    ax.plot(xs, ys, marker="D", markersize=9, color=WIDENING_COLOR, linewidth=1.5, zorder=4)
    for r in rows:
        ax.annotate(
            f"d_ff={r['d_ff']}",
            xy=(r["ffn_unique"], r["val_loss"]), xytext=(0, 8), textcoords="offset points",
            fontsize=8, color=WIDENING_COLOR, ha="center",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("FFN unique count (total FFN params held constant at 31,457,280)")
    ax.set_ylabel("Validation cross-entropy loss")
    ax.set_title("Parameter-matched widening: Val Loss vs. FFN unique count (d_model=512)")
    ax.grid(True, axis="both", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


def plot_cost(rows: list, output_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(12.5, 5.5))
    xs = [r["ffn_unique"] for r in rows]
    hours = [r["train_time_sec"] / 3600 for r in rows]
    vram = [r["peak_vram_mb"] / 1000 for r in rows]

    for ax, ys, ylabel, title in [
        (ax_t, hours, "Training time (h, full 1 epoch)", "Training time vs. FFN unique count"),
        (ax_v, vram, "Peak VRAM (GB)", "Peak VRAM vs. FFN unique count"),
    ]:
        ax.plot(xs, ys, marker="D", markersize=9, color=WIDENING_COLOR, linewidth=1.5, zorder=4)
        for r, y in zip(rows, ys):
            ax.annotate(
                f"bs={r['batch_size_used']}",
                xy=(r["ffn_unique"], y), xytext=(0, 8), textcoords="offset points",
                fontsize=8, color=WIDENING_COLOR, ha="center",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("FFN unique count")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="both", color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)

    # ffn_unique=2 (d_ff=10240) had to drop to batch_size=8 (grad_accum=3) to
    # fit under the RTX 4090's 24GB -- at batch_size=24 it measured ~26.0GB
    # in preflight (would overflow into slow system-RAM paging). Mark that
    # "at matched batch_size=24" reference point for an honest comparison.
    ffn2_row = next(r for r in rows if r["ffn_unique"] == 2)
    ax_v.scatter(
        [2], [_FFN2_DFF10240_BS24_PREFLIGHT_VRAM_MB / 1000], marker="x", s=100,
        color=GUIDE_COLOR, zorder=5,
    )
    ax_v.plot([2, 2], [ffn2_row["peak_vram_mb"] / 1000, _FFN2_DFF10240_BS24_PREFLIGHT_VRAM_MB / 1000],
              color=GUIDE_COLOR, linestyle=":", linewidth=1, zorder=3)
    ax_v.annotate(
        "at bs=24 (preflight est.,\nexceeds 24GB budget)",
        xy=(2, _FFN2_DFF10240_BS24_PREFLIGHT_VRAM_MB / 1000), xytext=(8, 0),
        textcoords="offset points", fontsize=8, color=GUIDE_COLOR,
    )
    ax_v.axhline(24.0, color="0.7", linestyle="--", linewidth=1, zorder=1)
    ax_v.annotate("RTX 4090 VRAM (24GB)", xy=(xs[-1], 24.0), xytext=(0, 4),
                  textcoords="offset points", fontsize=8, color="0.5", ha="right")
    ax_v.set_ylim(top=_FFN2_DFF10240_BS24_PREFLIGHT_VRAM_MB / 1000 * 1.12)

    fig.suptitle("Parameter-matched widening: training cost vs. FFN unique count (d_model=512, fixed effective batch size 24)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the parameter-matched FFN widening sweep (16x1280 / 8x2560 / 4x5120 / "
        "2x10240 at d_model=512, total FFN params held constant): val_loss and training-cost "
        "(time/VRAM) vs FFN unique count."
    )
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--loss-output", default="checkpoints/compare/ffn_widening_unique_vs_val_loss.png")
    parser.add_argument("--cost-output", default="checkpoints/compare/ffn_widening_unique_vs_cost.png")
    args = parser.parse_args()

    rows = load_widening_rows(args.results_path)
    plot_val_loss(rows, args.loss_output)
    plot_cost(rows, args.cost_output)

    print("\nParameter-matched widening sweep (d_model=512, total FFN params=31,457,280 fixed):")
    for r in rows:
        print(
            f"  N={r['ffn_unique']:2d} (d_ff={r['d_ff']:5d}, bs={r['batch_size_used']}): "
            f"params={r['total_params']:,}, val_loss={r['val_loss']:.4f}, ppl={r['val_perplexity']:.3f}, "
            f"time={r['train_time_sec']/3600:.2f}h, peak_vram={r['peak_vram_mb']/1000:.2f}GB"
        )


if __name__ == "__main__":
    main()
