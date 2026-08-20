"""LightGBM 패턴 분류 모델 — 9종 불량 패턴을 판정한다.

무엇을: 99개 기하 피처를 입력으로 받아 wafer map의 불량 패턴을 분류한다.

가장 어려운 점 — **극심한 클래스 불균형**:
    none 85.25% (5,115장)  vs  Near-full 0.08% (5장)  → 약 1,000 : 1

    아무 생각 없이 학습하면 모델은 "전부 none"이라고 답한다. 그래도 정확도가
    85%나 나오기 때문이다. 그래서 이 모듈은 세 겹으로 대응한다.
      1. 평가 지표를 **macro-F1**로 — 클래스별 F1의 단순 평균이라, 희소 클래스를
         통째로 놓치면 점수가 크게 깎인다. 정확도(accuracy)는 절대 쓰지 않는다.
      2. `class_weight="balanced"` — 희소 클래스의 오답에 더 큰 벌점을 준다.
      3. **회전·미러 증강** — 웨이퍼는 회전 대칭이라 물리적으로 타당한 증강이다.
         ⚠️ 증강은 반드시 **학습 fold 안에서만** 수행한다(§누출 방지).

참고: docs/00_design.md §M1(모델), §8(성공 기준 macro-F1 ≥ 0.80)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wafermap.config import MODELS_DIR, PATTERN_LABELS

MODEL_FILE = "pattern_lgbm.txt"
META_FILE = "pattern_lgbm_meta.json"

#: 학습 기본 하이퍼파라미터. 데이터가 작고(수천 장) 피처가 많아(99개)
#: 과적합이 쉬우므로 보수적으로 잡았다.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": len(PATTERN_LABELS),
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


@dataclass
class CVReport:
    """교차검증 결과."""

    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_support: dict[str, int]
    confusion: list[list[int]]
    labels: list[str] = field(default_factory=lambda: list(PATTERN_LABELS))
    n_features: int = 0
    train_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"macro-F1 : {self.macro_f1:.4f}",
            f"accuracy : {self.accuracy:.4f}  (참고용 — 불균형 때문에 의미가 약하다)",
            "",
            f"{'클래스':<12}{'F1':>8}{'recall':>9}{'표본':>8}",
            "-" * 38,
        ]
        for label in self.labels:
            lines.append(
                f"{label:<12}{self.per_class_f1[label]:>8.3f}"
                f"{self.per_class_recall[label]:>9.3f}{self.per_class_support[label]:>8,}"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# 증강
# ──────────────────────────────────────────────────────────────────────────


def augment_maps(
    maps: dict[str, np.ndarray],
    labels: pd.Series,
    *,
    target_min: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """희소 클래스를 회전·미러로 증강한다.

    무엇을: 표본이 `target_min`장 미만인 클래스를 90°/180°/270° 회전과 좌우/상하
            반전 조합으로 늘린다.

    왜 회전이 물리적으로 타당한가 ★: 웨이퍼는 원판이고 노치(notch) 방향을 빼면
        회전 대칭이다. Center 불량을 90° 돌려도 여전히 Center 불량이다.
        이미지 분류에서 흔히 쓰는 증강(밝기 변경, 크롭)과 달리, 여기서는 증강이
        **없던 정보를 지어내는 게 아니라** 같은 물리 현상의 다른 관측일 뿐이다.

    ⚠️ 반드시 학습 fold 안에서만 호출할 것. 전체 데이터에 먼저 적용하면 원본과
       증강본이 학습/검증에 나뉘어 들어가 **검증 점수가 부풀려진다**(누출).

    Args:
        maps: {wafer_id: 2D 배열}
        labels: wafer_id → 패턴 라벨
        target_min: 클래스당 최소 목표 장수
        rng: 난수 생성기

    Returns:
        (증강된 맵 딕셔너리, 증강된 라벨 Series)
    """
    rng = rng or np.random.default_rng(0)
    counts = labels.value_counts()

    # (회전 횟수, 좌우반전 여부) 조합 — 항등변환 (0, False)은 원본이므로 제외
    transforms = [(k, flip) for k in range(4) for flip in (False, True)][1:]

    new_maps = dict(maps)
    new_labels = dict(labels)

    for label, count in counts.items():
        if count >= target_min or count == 0:
            continue
        source_ids = [w for w in maps if labels.get(w) == label]
        if not source_ids:
            continue

        needed = target_min - count
        for i in range(needed):
            src = source_ids[i % len(source_ids)]
            k, flip = transforms[(i // len(source_ids)) % len(transforms)]

            wafer = np.rot90(maps[src], k=k)
            if flip:
                wafer = np.fliplr(wafer)

            aug_id = f"{src}__aug{i:04d}"
            new_maps[aug_id] = np.ascontiguousarray(wafer)
            new_labels[aug_id] = label

    return new_maps, pd.Series(new_labels)


# ──────────────────────────────────────────────────────────────────────────
# 학습·평가
# ──────────────────────────────────────────────────────────────────────────


def _encode(labels: pd.Series) -> np.ndarray:
    """패턴명을 정수 인덱스로 바꾼다 (PATTERN_LABELS 순서 고정)."""
    lookup = {label: i for i, label in enumerate(PATTERN_LABELS)}
    return labels.map(lookup).to_numpy()


def cross_validate(
    features: pd.DataFrame,
    maps: dict[str, np.ndarray] | None = None,
    *,
    n_splits: int = 5,
    seed: int = 0,
    params: dict[str, object] | None = None,
    num_boost_round: int = 400,
    augment_target: int = 200,
    verbose: bool = True,
) -> CVReport:
    """계층 교차검증으로 성능을 측정한다.

    어떻게 누출을 막나 ★: fold를 먼저 나눈 뒤, **학습 fold의 맵만** 증강해서
        피처를 다시 뽑는다. 검증 fold는 원본 그대로 둔다. 이렇게 해야
        "같은 웨이퍼의 회전본이 학습과 검증에 동시에 존재하는" 상황을 막는다.

    Args:
        features: build.build()가 만든 피처 행렬 (wafer_id, pattern_label 포함)
        maps: 증강에 쓸 원본 맵. None이면 증강 없이 학습한다.
        n_splits: fold 수
        seed: 난수 시드
        params: LightGBM 파라미터 (None이면 DEFAULT_PARAMS)
        num_boost_round: 부스팅 라운드 수
        augment_target: 클래스당 최소 목표 장수 (증강 기준)

    Returns:
        CVReport
    """
    import lightgbm as lgb
    from sklearn.metrics import confusion_matrix, f1_score, recall_score
    from sklearn.model_selection import StratifiedKFold

    from wafermap.features import build as feature_build

    t0 = time.time()
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = feature_build.feature_columns(features)

    X = features[feature_cols].to_numpy(dtype=np.float32)
    y = _encode(features["pattern_label"])
    wafer_ids = features["wafer_id"].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(features), dtype=int)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]

        # ── 학습 fold만 증강 ────────────────────────────────────────────
        if maps is not None and augment_target > 0:
            train_ids = set(wafer_ids[train_idx])
            train_maps = {w: maps[w] for w in train_ids if w in maps}
            train_labels = features.set_index("wafer_id").loc[
                list(train_maps), "pattern_label"
            ]

            aug_maps, aug_labels = augment_maps(
                train_maps,
                train_labels,
                target_min=augment_target,
                rng=np.random.default_rng(seed + fold),
            )
            # 새로 생긴 것만 피처를 뽑는다 (원본은 이미 있다)
            new_ids = {w for w in aug_maps if w not in train_maps}
            if new_ids:
                extra = feature_build.build(
                    {w: aug_maps[w] for w in new_ids},
                    aug_labels,
                    verbose=False,
                )
                X_train = np.vstack([X_train, extra[feature_cols].to_numpy(dtype=np.float32)])
                y_train = np.concatenate([y_train, _encode(extra["pattern_label"])])

        # ── 클래스 가중치 ───────────────────────────────────────────────
        # 표본이 적은 클래스일수록 큰 가중치. 이것이 "전부 none" 붕괴를 막는다.
        counts = np.bincount(y_train, minlength=len(PATTERN_LABELS)).astype(float)
        weights_per_class = len(y_train) / (len(PATTERN_LABELS) * np.maximum(counts, 1))
        sample_weight = weights_per_class[y_train]

        model = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train, weight=sample_weight),
            num_boost_round=num_boost_round,
        )
        oof_pred[valid_idx] = model.predict(X[valid_idx]).argmax(axis=1)

        if verbose:
            fold_f1 = f1_score(y[valid_idx], oof_pred[valid_idx], average="macro", zero_division=0)
            print(f"   fold {fold}/{n_splits}  macro-F1 = {fold_f1:.4f}")

    present = sorted(set(y) | set(oof_pred))
    f1_per = f1_score(y, oof_pred, average=None, labels=range(len(PATTERN_LABELS)), zero_division=0)
    recall_per = recall_score(
        y, oof_pred, average=None, labels=range(len(PATTERN_LABELS)), zero_division=0
    )
    support = np.bincount(y, minlength=len(PATTERN_LABELS))

    return CVReport(
        macro_f1=float(f1_score(y, oof_pred, average="macro", zero_division=0)),
        accuracy=float((y == oof_pred).mean()),
        per_class_f1={PATTERN_LABELS[i]: float(f1_per[i]) for i in range(len(PATTERN_LABELS))},
        per_class_recall={
            PATTERN_LABELS[i]: float(recall_per[i]) for i in range(len(PATTERN_LABELS))
        },
        per_class_support={PATTERN_LABELS[i]: int(support[i]) for i in range(len(PATTERN_LABELS))},
        confusion=confusion_matrix(y, oof_pred, labels=range(len(PATTERN_LABELS))).tolist(),
        n_features=len(feature_cols),
        train_seconds=time.time() - t0,
    )


def fit_final(
    features: pd.DataFrame,
    maps: dict[str, np.ndarray] | None = None,
    *,
    seed: int = 0,
    params: dict[str, object] | None = None,
    num_boost_round: int = 400,
    augment_target: int = 200,
):
    """전체 데이터로 최종 모델을 학습한다 (배포용).

    왜 CV와 따로 두나: CV는 성능을 **추정**하는 절차이고, 배포에는 모든 데이터를
        쓴 모델이 낫다. 두 목적을 한 함수에 섞으면 "검증 점수를 낸 모델"과
        "배포한 모델"이 달라지는 혼란이 생긴다.
    """
    import lightgbm as lgb

    from wafermap.features import build as feature_build

    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = feature_build.feature_columns(features)

    X = features[feature_cols].to_numpy(dtype=np.float32)
    y = _encode(features["pattern_label"])

    if maps is not None and augment_target > 0:
        labels = features.set_index("wafer_id")["pattern_label"]
        aug_maps, aug_labels = augment_maps(
            maps, labels, target_min=augment_target, rng=np.random.default_rng(seed)
        )
        new_ids = {w for w in aug_maps if w not in maps}
        if new_ids:
            extra = feature_build.build(
                {w: aug_maps[w] for w in new_ids}, aug_labels, verbose=False
            )
            X = np.vstack([X, extra[feature_cols].to_numpy(dtype=np.float32)])
            y = np.concatenate([y, _encode(extra["pattern_label"])])

    counts = np.bincount(y, minlength=len(PATTERN_LABELS)).astype(float)
    weights_per_class = len(y) / (len(PATTERN_LABELS) * np.maximum(counts, 1))

    model = lgb.train(
        params,
        lgb.Dataset(X, label=y, weight=weights_per_class[y]),
        num_boost_round=num_boost_round,
    )
    return model, feature_cols


# ──────────────────────────────────────────────────────────────────────────
# 저장·로드
# ──────────────────────────────────────────────────────────────────────────


def save(model, feature_cols: list[str], report: CVReport, source: str) -> Path:
    """모델과 메타데이터를 저장한다."""
    out_dir = MODELS_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(out_dir / MODEL_FILE))
    (out_dir / META_FILE).write_text(
        json.dumps(
            {
                "feature_columns": feature_cols,
                "labels": list(PATTERN_LABELS),
                "source": source,
                "report": asdict(report),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def load(source: str = "synthetic"):
    """저장된 모델과 메타데이터를 읽는다.

    Returns:
        (Booster, meta dict)
    """
    import lightgbm as lgb

    out_dir = MODELS_DIR / source
    model_path = out_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델이 없습니다: {model_path}\n"
            f"  먼저 학습하세요: python scripts/train_pattern_model.py --source {source}"
        )
    meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(model_path)), meta


def predict(model, features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """패턴을 예측한다.

    Returns:
        wafer_id, pred_label, pred_confidence + 클래스별 확률
    """
    X = features[feature_cols].to_numpy(dtype=np.float32)
    proba = model.predict(X)
    idx = proba.argmax(axis=1)

    out = pd.DataFrame(
        {
            "wafer_id": features["wafer_id"].to_numpy(),
            "pred_label": [PATTERN_LABELS[i] for i in idx],
            "pred_confidence": proba.max(axis=1),
        }
    )
    for i, label in enumerate(PATTERN_LABELS):
        out[f"proba_{label}"] = proba[:, i]
    return out


def feature_importance(model, feature_cols: list[str], top_n: int = 25) -> pd.DataFrame:
    """피처 중요도 (gain 기준)."""
    gain = model.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": feature_cols, "gain": gain})
        .sort_values("gain", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
