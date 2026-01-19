from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ladder.fitting.step1_flops import fit_step1 as fit_step1_flops
from ladder.fitting.step2 import fit_step2


def get_bench_names(df: pd.DataFrame) -> list[str]:
    bpb_sub_cols = [c for c in df.columns if c.startswith("correct_bpb_sub_")]
    return sorted({c[len("correct_bpb_sub_"):] for c in bpb_sub_cols})


def build_step1_data(df: pd.DataFrame, bpb_col: str) -> dict:
    step1_df = df.loc[df["FLOP"].notna(), ["FLOP", bpb_col]].dropna()
    if step1_df.empty:
        return {}
    return {
        "all": {
            "fs": step1_df["FLOP"].tolist(),
            "xs": step1_df[bpb_col].tolist(),
            "mode": "train",
        }
    }


def build_step2_data(df: pd.DataFrame, bpb_col: str, metric_col: str) -> dict:
    step2_df = df.loc[:, [bpb_col, metric_col]].dropna()
    if step2_df.empty:
        return {}
    return {
        "all": {
            "xs": step2_df[bpb_col].tolist(),
            "ys": step2_df[metric_col].tolist(),
            "mode": "train",
        }
    }


def fit_baseline_for_bench(
    df_mix: pd.DataFrame,
    bench: str,
    metric: str,
) -> dict | None:
    bpb_col = f"correct_bpb_sub_{bench}"
    metric_col = (
        f"acc_sub_{bench}" if metric == "acc_sub" else f"p_correct_choice_sub_{bench}"
    )
    if bpb_col not in df_mix.columns or metric_col not in df_mix.columns:
        return None

    step1_data = build_step1_data(df_mix, bpb_col)
    if not step1_data:
        return None

    step2_data = build_step2_data(df_mix, bpb_col, metric_col)
    if not step2_data:
        return None

    step1_coeffs, _ = fit_step1_flops(step1_data, y_metric="rc_bpb", use_two_param=False)
    step2_coeffs, _ = fit_step2(
        step2_data,
        task_name=None,
        y_metric="rc_bpb",
        use_log_sigmoid=False,
        use_helper_points=False,
    )

    return {
        "model_data_mix": str(df_mix["model_data_mix"].iloc[0]),
        "bench": bench,
        "metric": metric,
        "step1_a": float(step1_coeffs[0]),
        "step1_alpha": float(step1_coeffs[1]),
        "step1_E": float(step1_coeffs[2]),
        "step2_a": float(step2_coeffs[0]),
        "step2_x0": float(step2_coeffs[1]),
        "step2_k": float(step2_coeffs[2]),
        "step2_b": float(step2_coeffs[3]),
    }


def run_baseline(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    bench_names = get_bench_names(df)
    mixes = sorted(df["model_data_mix"].unique())

    results: list[dict] = []
    for mix in mixes:
        df_mix = df[df["model_data_mix"] == mix]
        for bench in bench_names:
            for metric in ("acc_sub", "p_correct_choice_sub"):
                try:
                    result = fit_baseline_for_bench(df_mix, bench, metric)
                except Exception:
                    result = None
                if result is not None:
                    results.append(result)

    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-step baseline fit for DataDecide.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-path",
        type=Path,
        default=base_dir / "data" / "6_long.parquet",
        help="Path to 6_long.parquet produced by 6_organize_data.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=base_dir / "data" / "7_baseline.parquet",
        help="Path to write baseline fit results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = run_baseline(
        input_path=args.input_path,
        output_path=args.output_path,
    )
    print(f"Wrote {len(results_df):,} rows to {args.output_path}")


if __name__ == "__main__":
    main()
