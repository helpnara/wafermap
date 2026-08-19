"""스키마 검증기 자체를 검사한다.

왜 검증기를 검사하나: 이 모듈은 다른 모든 데이터의 안전망이다. 안전망이 조용히
통과시키는 버그가 있으면 잘못된 데이터가 그대로 흘러가고, 검사를 했다는 사실이
오히려 잘못된 안심을 준다. 그래서 '틀린 데이터를 실제로 잡아내는지'를 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafermap.data import schema


def _valid_master(n=5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wafer_id": [f"W{i}" for i in range(n)],
            "lot_id": ["LOT1"] * n,
            "slot_no": np.arange(1, n + 1, dtype="int64"),
            "product": ["DDR5-16Gb"] * n,
            "tech_node": ["1z-nm"] * n,
            "fab_in_time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "eds_time": pd.date_range("2026-01-05", periods=n, freq="h"),
            # 검사 설비 축 (M5.5-①)
            "tester_id": ["ATE-01"] * n,
            "probe_card_id": ["PC-03"] * n,
            "probe_touchdown": np.full(n, 12_500, dtype="int64"),
            "die_total": np.full(n, 1584, dtype="int64"),
            "die_pass": np.full(n, 1500, dtype="int64"),
            "yield_pct": np.full(n, 94.7),
            "pattern_label": ["none"] * n,
            "data_source": ["synthetic"] * n,
            "is_labeled": np.ones(n, dtype=bool),
        }
    )


def test_valid_frame_passes():
    assert schema.validate(_valid_master(), schema.WAFER_MASTER) == []


def test_missing_column_is_detected():
    df = _valid_master().drop(columns=["yield_pct"])
    with pytest.raises(schema.SchemaError, match="누락 컬럼"):
        schema.validate(df, schema.WAFER_MASTER)


def test_extra_column_is_detected_when_not_allowed():
    df = _valid_master()
    df["surprise"] = 1
    with pytest.raises(schema.SchemaError, match="정의되지 않은 컬럼"):
        schema.validate(df, schema.WAFER_MASTER)


def test_extra_column_allowed_for_fdc():
    """FDC는 스텝마다 파라미터 컬럼이 달라 추가 컬럼을 허용해야 한다."""
    df = pd.DataFrame(
        {
            "wafer_id": ["W0"],
            "step_id": ["P020"],
            "step_name": ["Etch"],
            "equip_id": ["ETCH-A"],
            "chamber_id": ["ETCH-A/ch1"],
            "recipe_id": ["RCP-STD-01"],
            "run_time": pd.to_datetime(["2026-01-01"]),
            "rf_power_mean": [1500.0],  # 선언되지 않은 컬럼
        }
    )
    assert schema.validate(df, schema.FDC_SUMMARY) == []


def test_bad_dtype_is_detected():
    df = _valid_master()
    df["slot_no"] = df["slot_no"].astype(str)
    with pytest.raises(schema.SchemaError, match="dtype"):
        schema.validate(df, schema.WAFER_MASTER)


def test_null_in_non_nullable_column_is_detected():
    df = _valid_master()
    df.loc[0, "pattern_label"] = None
    with pytest.raises(schema.SchemaError, match="결측"):
        schema.validate(df, schema.WAFER_MASTER)


def test_disallowed_category_is_detected():
    df = _valid_master()
    df.loc[0, "pattern_label"] = "Blob"
    with pytest.raises(schema.SchemaError, match="허용되지 않은 값"):
        schema.validate(df, schema.WAFER_MASTER)


def test_out_of_range_value_is_detected():
    df = _valid_master()
    df.loc[0, "yield_pct"] = 120.0
    with pytest.raises(schema.SchemaError, match="최댓값"):
        schema.validate(df, schema.WAFER_MASTER)


def test_duplicate_unique_key_is_detected():
    df = pd.concat([_valid_master(2), _valid_master(2)], ignore_index=True)
    with pytest.raises(schema.SchemaError, match="중복"):
        schema.validate(df, schema.WAFER_MASTER)


def test_non_strict_returns_problem_list():
    df = _valid_master().drop(columns=["yield_pct"])
    problems = schema.validate(df, schema.WAFER_MASTER, strict=False)
    assert problems and "누락 컬럼" in problems[0]


def test_nullable_column_accepts_na():
    """원인이 없는 패턴은 true_root_step이 결측인 게 정상이다."""
    df = pd.DataFrame(
        {
            "wafer_id": ["W0", "W1"],
            "pattern_label": ["none", "Center"],
            "true_root_step": [None, "P050"],
            "true_root_equip": [None, "CMP-01/ch2"],
            "true_root_params": [None, ["down_force"]],
            "severity": [np.nan, 1.1],
            "is_confounded": [False, False],
            "is_test_induced": [False, True],
            "is_unexplained": [False, False],
            "is_false_positive": [False, False],
        }
    )
    assert schema.validate(df, schema.GROUND_TRUTH) == []


# ── 테이블 간 정합성 ─────────────────────────────────────────────────────


def test_consistency_detects_missing_fdc():
    master = _valid_master(3)
    gt = pd.DataFrame(
        {
            "wafer_id": master["wafer_id"],
            "pattern_label": "none",
            "true_root_step": None,
            "true_root_equip": None,
            "true_root_params": None,
            "severity": np.nan,
            "is_confounded": False,
            "is_test_induced": False,
            "is_unexplained": False,
            "is_false_positive": False,
        }
    )
    fdc = pd.DataFrame({"wafer_id": ["W0"]})  # W1, W2 누락
    with pytest.raises(schema.SchemaError, match="FDC 데이터가 없는"):
        schema.validate_consistency(master, fdc, gt)


def test_consistency_detects_die_pass_exceeding_total():
    master = _valid_master(2)
    master.loc[0, "die_pass"] = 99_999
    gt = pd.DataFrame(
        {
            "wafer_id": master["wafer_id"],
            "pattern_label": "none",
            "true_root_step": None,
            "true_root_equip": None,
            "true_root_params": None,
            "severity": np.nan,
            "is_confounded": False,
            "is_test_induced": False,
            "is_unexplained": False,
            "is_false_positive": False,
        }
    )
    fdc = pd.DataFrame({"wafer_id": master["wafer_id"]})
    with pytest.raises(schema.SchemaError, match="die_pass > die_total"):
        schema.validate_consistency(master, fdc, gt)
