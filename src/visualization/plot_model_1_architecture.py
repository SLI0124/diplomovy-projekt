from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torchview import draw_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_SRC = PROJECT_ROOT / "src" / "dl"
if str(DL_SRC) not in sys.path:
    sys.path.insert(0, str(DL_SRC))


def _build_model(*, covariate_dim: int, prediction_length: int) -> torch.nn.Module:
    model_module = importlib.import_module("adapters.custom.model_1")
    model_cls = model_module._Model1Forecaster
    return model_cls(
        feature_dim=1 + covariate_dim,
        future_covariate_dim=covariate_dim,
        prediction_length=prediction_length,
        hidden_dim=128,
        dropout=0.1,
        attention_heads=4,
        ff_hidden_dim=256,
    )


def _render_with_torchview(
    *,
    output_path: Path,
    context_length: int,
    prediction_length: int,
    covariate_dim: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(
        covariate_dim=covariate_dim,
        prediction_length=prediction_length,
    ).to(device)
    model.eval()

    context = torch.randn(1, context_length, device=device, dtype=torch.float32)
    context_covariates = torch.randn(
        1,
        context_length,
        covariate_dim,
        device=device,
        dtype=torch.float32,
    )
    future_covariates = torch.randn(
        1,
        prediction_length,
        covariate_dim,
        device=device,
        dtype=torch.float32,
    )

    graph = draw_graph(
        model,
        input_data={
            "context": context,
            "context_covariates": context_covariates,
            "future_covariates": future_covariates,
        },
        expand_nested=True,
        graph_name="Model 1 Architecture",
        depth=4,
        device=str(device),
        save_graph=False,
    )

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    graph.visual_graph.render(
        filename=output_path.stem,
        directory=str(output_dir),
        cleanup=True,
        format="png",
    )


def _render_fallback_diagram(*, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = [
        "Input: context + covariates",
        "LayerNorm",
        "Linear projection",
        "Temporal Conv1D + BatchNorm + Dropout",
        "BiGRU encoder",
        "Multihead self-attention",
        "Feed-forward refinement",
        "Mean/Max pooling",
        "Future covariate summary",
        "Prediction head",
        "Output: forecast horizon",
    ]

    fig, ax = plt.subplots(figsize=(9, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(blocks) + 1)
    ax.axis("off")

    for idx, text in enumerate(blocks, start=1):
        y = len(blocks) + 1 - idx
        ax.add_patch(
            Rectangle(
                (1.5, y - 0.35),
                7.0,
                0.7,
                fill=True,
                linewidth=1.2,
                edgecolor="#2f3d4a",
                facecolor="#d6eaf8",
            )
        )
        ax.text(5.0, y, text, ha="center", va="center", fontsize=10)
        if idx < len(blocks):
            ax.annotate(
                "",
                xy=(5.0, y - 0.65),
                xytext=(5.0, y - 0.95),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#2f3d4a"},
            )

    ax.set_title("Model 1 Architecture", fontsize=14, pad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Model 1 architecture to PNG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "plots" / "model_1_architecture.png",
        help="Target PNG path.",
    )
    parser.add_argument("--context-length", type=int, default=168)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--covariate-dim", type=int, default=8)
    args = parser.parse_args()

    try:
        _render_with_torchview(
            output_path=args.output,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            covariate_dim=args.covariate_dim,
        )
        print(f"Saved architecture image: {args.output}")
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(
            "Torchview render failed, using matplotlib fallback. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        _render_fallback_diagram(output_path=args.output)
        print(f"Saved architecture image (fallback): {args.output}")


if __name__ == "__main__":
    main()
