from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LayeredAnalysisConfig:
    scores_path: str
    output_path: str
    bins: int = 5


def analyze_score_layers(config: LayeredAnalysisConfig) -> dict[str, object]:
    frame = pd.read_parquet(config.scores_path).sort_values(["trade_date", "prediction"], ascending=[True, False])
    grouped_records: list[dict[str, object]] = []

    for trade_date, day_frame in frame.groupby("trade_date"):
        if len(day_frame) < config.bins:
            continue
        working = day_frame.copy()
        working["layer"] = pd.qcut(
            working["prediction"].rank(method="first"),
            q=config.bins,
            labels=False,
        )
        working["layer"] = config.bins - 1 - working["layer"].astype(int)
        layer_mean = working.groupby("layer")["label"].mean().to_dict()
        grouped_records.append(
            {
                "trade_date": pd.Timestamp(trade_date).date().isoformat(),
                "top_layer_return": float(layer_mean.get(0, 0.0)),
                "bottom_layer_return": float(layer_mean.get(config.bins - 1, 0.0)),
                "top_bottom_spread": float(layer_mean.get(0, 0.0) - layer_mean.get(config.bins - 1, 0.0)),
            }
        )

    analysis_frame = pd.DataFrame(grouped_records)
    if analysis_frame.empty:
        raise ValueError("no valid layered analysis rows were generated")

    summary = {
        "rows": int(len(analysis_frame)),
        "mean_top_layer_return": float(analysis_frame["top_layer_return"].mean()),
        "mean_bottom_layer_return": float(analysis_frame["bottom_layer_return"].mean()),
        "mean_top_bottom_spread": float(analysis_frame["top_bottom_spread"].mean()),
        "positive_spread_ratio": float((analysis_frame["top_bottom_spread"] > 0).mean()),
    }
    payload = {"summary": summary, "by_date": grouped_records}

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
