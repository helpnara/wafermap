"""피처 추출 검증.

핵심 질문 두 가지:
  1. 피처가 패턴을 **구분하는가** — 설계에서 노린 피처가 실제로 갈라내는가?
  2. 피처가 맵 크기에 **불변인가** — 형상이 아니라 크기를 학습할 위험이 없는가?

2번이 특히 중요하다. 합성 데이터는 맵 크기가 전부 같아서 이 문제가 드러나지 않지만,
실측 WM-811K는 26×26부터 300×200까지 제각각이다. 크기 의존 피처가 하나라도 남아
있으면 모델이 "이 크기의 맵은 보통 Edge-Ring"이라는 지름길을 학습하고, 합성에서는
멀쩡하던 성능이 실데이터에서 무너진다. 그래서 크기 불변성을 테스트로 못박는다.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafermap.config import DIE_FAIL, DIE_NONE, DIE_PASS
from wafermap.data.synth_wafer import build_geometry, generate_wafer_map
from wafermap.features import build, connectivity, geometry, radon_feat

#: 크기 의존이 의심되면 반드시 걸러야 할 금지 피처명 조각
FORBIDDEN_NAME_PARTS = ("map_rows", "map_cols", "map_aspect", "die_total")


@pytest.fixture(scope="module")
def maps(geom):
    """패턴별 대표 맵 몇 장씩."""
    rng = np.random.default_rng(20)
    return {
        pat: [generate_wafer_map(pat, rng, geom) for _ in range(12)]
        for pat in ("Center", "Donut", "Edge-Ring", "Edge-Loc", "Loc", "Scratch", "Random", "none")
    }


# ── 기본 동작 ────────────────────────────────────────────────────────────


def test_extract_returns_finite_values(geom, rng):
    """모든 피처가 유한한 실수여야 한다 (inf/NaN은 LightGBM 학습을 깨뜨린다)."""
    for pattern in ("Center", "Scratch", "none", "Near-full"):
        feats = build.extract_one(generate_wafer_map(pattern, rng, geom))
        for name, value in feats.items():
            assert np.isfinite(value), f"{pattern}.{name} = {value}"


def test_feature_set_is_stable(geom, rng):
    """어떤 패턴이든 같은 피처 집합이 나와야 한다 (컬럼이 들쭉날쭉하면 안 된다)."""
    keys = None
    for pattern in ("Center", "none", "Near-full", "Scratch"):
        feats = build.extract_one(generate_wafer_map(pattern, rng, geom))
        if keys is None:
            keys = set(feats)
        else:
            assert set(feats) == keys, f"{pattern}에서 피처 집합이 달라짐"


def test_empty_and_degenerate_maps_do_not_crash():
    """불량이 0개이거나 die가 거의 없는 맵에서도 죽지 않아야 한다."""
    all_pass = np.full((20, 20), DIE_PASS, dtype=np.int8)
    all_pass[0, :] = DIE_NONE
    feats = build.extract_one(all_pass)
    assert all(np.isfinite(v) for v in feats.values())

    tiny = np.zeros((3, 3), dtype=np.int8)
    tiny[1, 1] = DIE_FAIL
    feats = build.extract_one(tiny)
    assert all(np.isfinite(v) for v in feats.values())


# ── 크기 불변성 (누출 방지) ★ ───────────────────────────────────────────


def test_no_absolute_size_features(geom, rng):
    """맵 크기를 직접 노출하는 피처가 존재하면 안 된다."""
    feats = build.extract_one(generate_wafer_map("Center", rng, geom))
    leaked = [
        name for name in feats if any(part in name for part in FORBIDDEN_NAME_PARTS)
    ]
    assert not leaked, (
        f"맵 크기를 인코딩하는 피처가 남아 있습니다: {leaked}\n"
        f"  실측 WM-811K는 맵 크기가 제각각이라 모델이 지름길을 학습합니다."
    )


def test_features_are_scale_invariant():
    """같은 형상을 2배 해상도로 그려도 피처가 크게 달라지면 안 된다.

    어떻게 검사하나: 동일한 Center 패턴을 작은 격자와 큰 격자에 각각 그린 뒤
    피처를 비교한다. 형상이 같으므로 값도 비슷해야 한다.
    """
    small = build_geometry(die_width_mm=9.0, die_height_mm=9.0)
    large = build_geometry(die_width_mm=4.5, die_height_mm=4.5)
    assert large.n_die > small.n_die * 3  # 확실히 다른 해상도인지 확인

    # 확률장을 결정론적으로 만들기 위해 severity를 크게 주고 여러 장 평균
    def mean_feats(g, n=15):
        rng = np.random.default_rng(77)
        rows = [build.extract_one(generate_wafer_map("Center", rng, g)) for _ in range(n)]
        return {k: np.mean([r[k] for r in rows]) for k in rows[0]}

    fs, fl = mean_feats(small), mean_feats(large)

    # 비율형 피처는 상대차로 비교한다
    for key in ("fail_ratio", "center2_ratio", "largest_cluster_ratio"):
        a, b = fs[key], fl[key]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-3)
        assert rel < 0.45, f"{key}가 해상도에 따라 크게 변함: {a:.3f} vs {b:.3f} (상대차 {rel:.0%})"

    # radial_peak_ring은 0~9 정수 인덱스라 **절대차**로 본다.
    # 왜: 0.4와 0.067은 둘 다 "ring 0"을 뜻해 사실상 동일한데, 상대차로 재면
    #     0에 가까운 값끼리라 83%라는 무의미하게 큰 수치가 나온다.
    #     인덱스형 피처는 "몇 번째 링인가"의 차이로 판단해야 한다.
    assert abs(fs["radial_peak_ring"] - fl["radial_peak_ring"]) < 1.5, (
        f"정점 링이 해상도에 따라 이동함: "
        f"{fs['radial_peak_ring']:.2f} vs {fl['radial_peak_ring']:.2f}"
    )


def test_cluster_features_are_normalized(geom, rng):
    """연결성분 피처가 절대 개수가 아니라 비율이어야 한다."""
    feats = connectivity.extract(generate_wafer_map("Loc", rng, geom))
    for name, value in feats.items():
        if "ratio" in name or "norm" in name or name in ("fail_compactness", "cluster_size_cv"):
            assert value <= 50.0, f"{name}={value} — 정규화되지 않은 것으로 보임"


def test_radon_features_are_normalized(geom, rng):
    """Radon 피처가 불량 개수에 비례해 커지면 안 된다.

    Near-full(불량 수천 개)과 Loc(불량 수십 개)의 값이 자릿수 차이로 벌어지면,
    그 피처는 형상이 아니라 불량 개수를 인코딩하고 있는 것이다.

    피처명이 radon_mean_* 에서 radon_cv_* 로 바뀐 이유는
    `tests/test_resolution_invariance.py` 에 적혀 있다 — 열평균은 정의상
    `불량수 / 검출기길이` 라서 정규화해도 맵 크기만 남는 값이었다.
    """
    few = radon_feat.extract(generate_wafer_map("Loc", rng, geom))
    many = radon_feat.extract(generate_wafer_map("Near-full", rng, geom))
    a, b = few["radon_cv_10"], many["radon_cv_10"]
    assert max(a, b) / max(min(a, b), 1e-9) < 30, (
        f"radon_cv가 불량 개수에 비례함: Loc={a:.4f} vs Near-full={b:.4f}"
    )


# ── 판별력 ───────────────────────────────────────────────────────────────


def _cohens_d(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    return float((a.mean() - b.mean()) / pooled)


def _values(maps, pattern, extractor, key):
    return [extractor(m)[key] for m in maps[pattern]]


@pytest.mark.parametrize(
    "pattern_a,pattern_b,key,extractor_name",
    [
        # Edge-Ring vs Edge-Loc: 원주 전체냐 일부냐 → 각도 집중도
        ("Edge-Ring", "Edge-Loc", "angular_concentration", "geometry"),
        ("Edge-Ring", "Edge-Loc", "angular_active_ratio", "geometry"),
        # Center vs Donut: 중심이 비었냐 → 중심 2링 비율, 정점 위치
        ("Center", "Donut", "center2_ratio", "geometry"),
        ("Center", "Donut", "radial_peak_ring", "geometry"),
        # Loc vs Scratch: 둥그냐 길쭉하냐 → 연결성분 이심률
        ("Loc", "Scratch", "largest_cluster_elongation", "connectivity"),
        ("Loc", "Scratch", "largest_cluster_eccentricity", "connectivity"),
        # Random vs Loc: 흩어졌냐 뭉쳤냐 → 최대 덩어리 비율
        ("Random", "Loc", "largest_cluster_ratio", "connectivity"),
        # none vs Random: 불량률 자체
        ("none", "Random", "fail_ratio", "geometry"),
    ],
)
def test_designed_features_separate_confusable_pairs(
    maps, pattern_a, pattern_b, key, extractor_name
):
    """설계에서 특정 쌍을 가르라고 만든 피처가 실제로 가르는지 확인한다.

    왜 이 테스트가 중요한가: 피처를 100개 넣었다고 분류가 되는 게 아니다.
    "Edge-Ring과 Edge-Loc은 각도 집중도로 가른다"는 **설계 의도**가 데이터에서
    성립하는지 확인해야, 모델 성능이 나쁠 때 피처 문제인지 모델 문제인지 알 수 있다.
    """
    extractor = {"geometry": geometry.extract, "connectivity": connectivity.extract}[
        extractor_name
    ]
    d = _cohens_d(_values(maps, pattern_a, extractor, key), _values(maps, pattern_b, extractor, key))
    assert abs(d) > 0.8, (
        f"{key}가 {pattern_a} vs {pattern_b}를 가르지 못함 (Cohen's d={d:+.2f}, 기준 0.8)"
    )


def test_radial_peak_matches_pattern_physics(maps):
    """반경 프로파일의 정점 위치가 패턴의 물리와 맞아야 한다."""
    peaks = {
        p: np.median([geometry.extract(m)["radial_peak_ring"] for m in maps[p]])
        for p in ("Center", "Donut", "Edge-Ring")
    }
    assert peaks["Center"] <= 1, f"Center 정점이 중심이 아님 (ring {peaks['Center']})"
    assert 3 <= peaks["Donut"] <= 6, f"Donut 정점이 중간이 아님 (ring {peaks['Donut']})"
    assert peaks["Edge-Ring"] >= 8, f"Edge-Ring 정점이 엣지가 아님 (ring {peaks['Edge-Ring']})"


# ── 파이프라인 ───────────────────────────────────────────────────────────


def test_build_produces_expected_columns(geom, rng):
    maps_dict = {
        f"W{i}": generate_wafer_map(p, rng, geom)
        for i, p in enumerate(("Center", "Loc", "none", "Scratch"))
    }
    df = build.build(maps_dict, verbose=False)

    assert len(df) == 4
    assert "wafer_id" in df.columns
    feature_cols = build.feature_columns(df)
    # 121개였다가 99개가 됐다 — 맵 크기만 인코딩하던 피처 22개를 걷어냈다.
    # (radon_mean_* 20개 + radon_mean_range + die_fill_ratio)
    assert len(feature_cols) == 99, f"피처 수가 바뀜: {len(feature_cols)}"
    assert df[feature_cols].isna().sum().sum() == 0
    assert np.isfinite(df[feature_cols].to_numpy()).all()


def test_build_parallel_matches_serial(geom, rng):
    """병렬 처리 결과가 단일 프로세스와 같아야 한다."""
    maps_dict = {
        f"W{i}": generate_wafer_map(p, rng, geom)
        for i, p in enumerate(("Center", "Donut", "Edge-Ring", "Loc", "none", "Scratch"))
    }
    serial = build.build(maps_dict, verbose=False, n_jobs=1)
    parallel = build.build(maps_dict, verbose=False, n_jobs=2)

    cols = build.feature_columns(serial)
    assert serial["wafer_id"].tolist() == parallel["wafer_id"].tolist()
    np.testing.assert_allclose(serial[cols].to_numpy(), parallel[cols].to_numpy(), rtol=1e-6)
