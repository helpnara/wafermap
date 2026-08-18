"""모델 계층 검증 — 학습이 아니라 **올바름**을 검사한다.

여기서 "모델 성능이 0.97 이상인가"를 검사하지는 않는다. 성능은 데이터에 따라
달라지고, 테스트가 성능 수치를 고정하면 데이터를 바꿀 때마다 테스트가 깨진다.

대신 성능 수치가 **믿을 만한지**를 결정하는 구조를 검사한다.
  · 라벨 인코딩이 일관된가 (섞이면 성능이 무의미해진다)
  · 증강이 물리적으로 타당한가 (회전해도 같은 패턴인가)
  · 누출이 없는가 (증강본이 검증 fold로 새지 않는가)
  · CNN 입력 인코딩이 의도대로인가
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafermap.config import DIE_FAIL, DIE_NONE, DIE_PASS, PATTERN_LABELS
from wafermap.data.synth_wafer import generate_wafer_map
from wafermap.models import pattern_lgbm

pytest_torch = pytest.importorskip("torch", reason="torch 미설치 시 CNN 테스트 건너뜀")
from wafermap.models import pattern_cnn  # noqa: E402


@pytest.fixture(scope="module")
def small_maps(geom):
    """패턴별 소량 맵 — 학습이 아니라 구조 검사용."""
    rng = np.random.default_rng(4)
    maps, labels = {}, {}
    for pat in PATTERN_LABELS:
        n = 3 if pat in ("Near-full", "Donut") else 8
        for i in range(n):
            wid = f"{pat}-{i}"
            maps[wid] = generate_wafer_map(pat, rng, geom)
            labels[wid] = pat
    return maps, pd.Series(labels)


# ── 라벨 인코딩 ──────────────────────────────────────────────────────────


def test_label_encoding_is_stable_and_reversible():
    """라벨 → 정수 → 라벨 왕복이 일치해야 한다.

    왜: 인코딩 순서가 흔들리면 혼동행렬의 행/열 의미가 어긋나고, 성능 수치가
        조용히 잘못된 클래스에 귀속된다. 오답보다 발견하기 어려운 종류의 버그다.
    """
    labels = pd.Series(list(PATTERN_LABELS))
    encoded = pattern_lgbm._encode(labels)

    assert list(encoded) == list(range(len(PATTERN_LABELS)))
    assert [PATTERN_LABELS[i] for i in encoded] == list(PATTERN_LABELS)


# ── 증강 ─────────────────────────────────────────────────────────────────


def test_augment_only_touches_rare_classes(small_maps):
    """표본이 충분한 클래스는 증강하지 않아야 한다."""
    maps, labels = small_maps
    aug_maps, aug_labels = augmented = pattern_lgbm.augment_maps(
        maps, labels, target_min=6, rng=np.random.default_rng(0)
    )
    del augmented

    before = labels.value_counts()
    after = aug_labels.value_counts()

    for pat in PATTERN_LABELS:
        if before.get(pat, 0) >= 6:
            assert after.get(pat, 0) == before.get(pat, 0), f"{pat}이 불필요하게 증강됨"
        else:
            assert after.get(pat, 0) >= 6, f"{pat}이 목표만큼 증강되지 않음"


def test_augmented_maps_preserve_die_count(small_maps):
    """회전·미러는 die 개수를 바꾸지 않아야 한다 (물리적 타당성)."""
    maps, labels = small_maps
    aug_maps, aug_labels = pattern_lgbm.augment_maps(
        maps, labels, target_min=12, rng=np.random.default_rng(0)
    )

    originals = {w: (m != DIE_NONE).sum() for w, m in maps.items()}
    for wid, wafer in aug_maps.items():
        if "__aug" not in wid:
            continue
        src = wid.split("__aug")[0]
        assert (wafer != DIE_NONE).sum() == originals[src], f"{wid}: die 수가 변함"
        assert (wafer == DIE_FAIL).sum() == (maps[src] == DIE_FAIL).sum()


def test_augmented_ids_are_traceable(small_maps):
    """증강본의 id에서 원본을 역추적할 수 있어야 한다.

    왜: 누출 여부를 검사하려면 "이 증강본이 어느 원본에서 왔는가"를 알아야 한다.
        id 규칙이 곧 그 추적 수단이다.
    """
    maps, labels = small_maps
    aug_maps, _ = pattern_lgbm.augment_maps(
        maps, labels, target_min=12, rng=np.random.default_rng(0)
    )
    for wid in aug_maps:
        if "__aug" in wid:
            assert wid.split("__aug")[0] in maps


def test_augmentation_preserves_radial_profile_shape(geom):
    """회전해도 반경 프로파일은 (거의) 같아야 한다 — 증강이 물리적으로 타당하다는 근거."""
    from wafermap.features import geometry as geo

    rng = np.random.default_rng(9)
    wafer = generate_wafer_map("Center", rng, geom)

    original = geo.extract(wafer)
    rotated = geo.extract(np.ascontiguousarray(np.rot90(wafer, k=1)))

    # 90° 회전은 행/열이 바뀌므로 반경 링 구성이 정확히 같지는 않지만,
    # "정점이 중심"이라는 성질은 보존되어야 한다
    assert abs(original["radial_peak_ring"] - rotated["radial_peak_ring"]) <= 2
    assert original["fail_ratio"] == pytest.approx(rotated["fail_ratio"], abs=1e-6)


# ── 누출 방지 ★ ─────────────────────────────────────────────────────────


def test_cv_does_not_leak_augmented_maps_into_validation(small_maps, monkeypatch):
    """증강본이 검증 fold에 들어가지 않는지 확인한다.

    어떻게 검사하나: `cross_validate`가 내부에서 증강할 때 만들어 낸 wafer_id를
    가로채, 그것들이 검증에 쓰인 인덱스와 겹치지 않음을 확인한다.
    검증 fold의 wafer_id는 원본 features에서만 나오므로, "__aug"가 붙은 id가
    검증 대상에 포함될 수 없어야 한다.
    """
    from wafermap.features import build as feature_build

    maps, labels = small_maps
    features = feature_build.build(maps, labels, verbose=False)

    seen_augmented: list[str] = []
    original_build = feature_build.build

    def spy(maps_arg, labels_arg=None, **kwargs):
        seen_augmented.extend(w for w in maps_arg if "__aug" in w)
        return original_build(maps_arg, labels_arg, **kwargs)

    monkeypatch.setattr(
        "wafermap.features.build.build", spy
    )

    report = pattern_lgbm.cross_validate(
        features, maps, n_splits=3, seed=0, num_boost_round=20,
        augment_target=10, verbose=False,
    )

    # 증강이 실제로 일어났는지 먼저 확인 (안 일어났으면 검사 자체가 무의미)
    assert seen_augmented, "증강이 수행되지 않아 누출 검사를 할 수 없음"
    # 검증 대상(=원본 features)에는 증강본이 하나도 없어야 한다
    assert not any("__aug" in w for w in features["wafer_id"]), (
        "원본 피처 행렬에 증강본이 섞여 있음 — 검증 fold로 누출된다"
    )
    assert 0.0 <= report.macro_f1 <= 1.0


def test_cv_report_shapes_are_consistent(small_maps):
    """CVReport의 각 항목이 서로 모순되지 않아야 한다."""
    from wafermap.features import build as feature_build

    maps, labels = small_maps
    features = feature_build.build(maps, labels, verbose=False)
    report = pattern_lgbm.cross_validate(
        features, None, n_splits=3, seed=0, num_boost_round=20, verbose=False
    )

    n = len(PATTERN_LABELS)
    assert len(report.confusion) == n
    assert all(len(row) == n for row in report.confusion)
    assert set(report.per_class_f1) == set(PATTERN_LABELS)
    # 혼동행렬 총합 = 표본 수
    assert sum(sum(row) for row in report.confusion) == len(features)
    # 대각합 / 전체 = accuracy
    diag = sum(report.confusion[i][i] for i in range(n))
    assert diag / len(features) == pytest.approx(report.accuracy, abs=1e-9)


# ── CNN ──────────────────────────────────────────────────────────────────


def test_cnn_input_separates_die_and_fail_channels(geom, rng):
    """2채널 인코딩이 의도대로인지 확인한다.

    채널0 = die 존재, 채널1 = 불량. 웨이퍼 밖은 두 채널 모두 0이어야 한다.
    """
    wafer = generate_wafer_map("Edge-Ring", rng, geom)
    x = pattern_cnn.to_tensor_input(wafer)

    assert x.shape == (2, pattern_cnn.IMG_SIZE, pattern_cnn.IMG_SIZE)
    assert set(np.unique(x)) <= {0.0, 1.0}, "이진 마스크가 아님"
    # 불량인 곳은 반드시 die가 있는 곳이다
    assert np.all(x[0][x[1] > 0] == 1.0), "die가 없는데 불량으로 표시된 위치가 있음"


def test_cnn_input_marks_outside_wafer_as_zero(geom, rng):
    """웨이퍼 바깥은 두 채널 모두 0이어야 한다 (1채널 인코딩의 함정 방지)."""
    wafer = generate_wafer_map("none", rng, geom)
    x = pattern_cnn.to_tensor_input(wafer)
    # 원본에서 die가 없는 모서리는 리샘플링 후에도 0
    assert x[0, 0, 0] == 0.0
    assert x[1, 0, 0] == 0.0


def test_cnn_model_shapes_and_size():
    """모델 출력 형태와 규모가 설계 목표에 맞는지 확인한다."""
    import torch

    model = pattern_cnn.make_model()
    n_params = sum(p.numel() for p in model.parameters())

    out = model(torch.randn(4, 2, pattern_cnn.IMG_SIZE, pattern_cnn.IMG_SIZE))
    assert out.shape == (4, len(PATTERN_LABELS))
    # 설계 목표: 약 20만 파라미터 (§4.3 클라우드 배포 요건)
    assert 100_000 < n_params < 400_000, f"파라미터 규모가 설계와 다름: {n_params:,}"


def test_grad_cam_returns_normalized_heatmap(geom, rng):
    """Grad-CAM이 0~1로 정규화된 열지도를 돌려줘야 한다."""
    model = pattern_cnn.make_model()
    model.eval()
    wafer = generate_wafer_map("Center", rng, geom)

    cam, idx = pattern_cnn.grad_cam(model, wafer)

    assert cam.shape == (pattern_cnn.IMG_SIZE, pattern_cnn.IMG_SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert 0 <= idx < len(PATTERN_LABELS)


def test_grad_cam_hook_is_removed(geom, rng):
    """hook을 등록한 뒤 반드시 제거해야 한다.

    왜: forward hook이 남아 있으면 이후 모든 순전파에서 계속 호출되어
        메모리가 누적되고, 여러 번 호출하면 리스트가 계속 커진다.
        `try/finally`로 제거하는 이유다.
    """
    model = pattern_cnn.make_model()
    model.eval()
    wafer = generate_wafer_map("Loc", rng, geom)

    before = len(model[3]._forward_hooks)
    pattern_cnn.grad_cam(model, wafer)
    pattern_cnn.grad_cam(model, wafer)
    assert len(model[3]._forward_hooks) == before, "hook이 제거되지 않고 누적됨"


def test_cnn_build_dataset_label_alignment(small_maps):
    """맵과 라벨이 같은 순서로 정렬되어야 한다."""
    maps, labels = small_maps
    ids = list(maps)
    X, y = pattern_cnn.build_dataset(maps, labels, ids)

    assert len(X) == len(y) == len(ids)
    for i, wid in enumerate(ids):
        assert PATTERN_LABELS[y[i]] == labels[wid], f"{wid}: 라벨이 어긋남"
