from __future__ import annotations

from ashare_backtest.qlib_integration.config import QlibFeatureSpec


def build_feature_field_map(feature_specs: tuple[QlibFeatureSpec, ...]) -> list[tuple[str, str]]:
    return [(spec.name, spec.expression) for spec in feature_specs]
