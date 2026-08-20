"""경량 CNN 패턴 분류 — LightGBM의 비교군.

무엇을: wafer map을 64×64 이미지로 변환해 CNN으로 분류하고, Grad-CAM으로
        "모델이 웨이퍼의 어디를 보고 판단했는가"를 시각화한다.

왜 CNN도 만드나 ★: LightGBM은 이미 macro-F1 0.97을 낸다. 성능을 더 올리려는 게
        목적이 아니다. 두 접근의 **성격 차이**를 보여 주는 것이 목적이다.

        | | LightGBM | CNN |
        |---|---|---|
        | 입력 | 사람이 설계한 99개 피처 | 원본 이미지 |
        | 지식 | 도메인 지식이 피처에 녹아 있음 | 데이터에서 스스로 학습 |
        | 해석 | SHAP — "어떤 지표가" 판정을 만들었나 | Grad-CAM — "어디를" 보았나 |
        | 비용 | 초 단위 학습, 4MB | 분 단위 학습, GPU 선호 |

        현업에서는 "왜 이렇게 판정했나"를 엔지니어에게 설명해야 하므로 해석 가능한
        쪽이 유리한 경우가 많다. 두 모델을 나란히 두면 그 트레이드오프를 근거를
        갖고 이야기할 수 있다.

입력 정규화: 맵 크기가 제각각이므로 64×64로 리샘플링한다. 이때 **채널을 2개로**
        나눈다 — (1) die 존재 마스크, (2) 불량 마스크. 하나의 채널에 0/1/2를
        그대로 넣으면 "die 없음(0)과 pass(1)의 차이"가 "pass(1)와 fail(2)의 차이"와
        같은 크기로 취급되어, 웨이퍼 밖 영역이 불량처럼 학습에 끼어든다.

참고: docs/00_design.md §M1(모델 비교군)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from wafermap.config import DIE_FAIL, DIE_NONE, MODELS_DIR, PATTERN_LABELS
from wafermap.models.pattern_lgbm import CVReport

MODEL_FILE = "pattern_cnn.pt"
META_FILE = "pattern_cnn_meta.json"

#: CNN 입력 해상도
IMG_SIZE = 64


def to_tensor_input(wafer: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """wafer map을 CNN 입력용 2채널 이미지로 변환한다.

    Args:
        wafer: (rows, cols) 배열. 0=die 없음, 1=pass, 2=fail
        size: 출력 해상도

    Returns:
        (2, size, size) float32 — 채널0=die 마스크, 채널1=불량 마스크
    """
    die = (wafer != DIE_NONE).astype(np.float32)
    fail = (wafer == DIE_FAIL).astype(np.float32)

    # 최근접 이웃 리샘플링 — die 격자는 이산 구조라 보간하면 없던 중간값이 생긴다
    rows, cols = wafer.shape
    ri = np.clip((np.arange(size) * rows / size).astype(int), 0, rows - 1)
    ci = np.clip((np.arange(size) * cols / size).astype(int), 0, cols - 1)

    return np.stack([die[np.ix_(ri, ci)], fail[np.ix_(ri, ci)]], axis=0)


def build_dataset(
    maps: dict[str, np.ndarray], labels: pd.Series, wafer_ids: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """맵 딕셔너리를 (X, y) 배열로 변환한다."""
    ids = wafer_ids if wafer_ids is not None else list(maps)
    X = np.stack([to_tensor_input(maps[w]) for w in ids])
    lookup = {label: i for i, label in enumerate(PATTERN_LABELS)}
    y = np.array([lookup[labels[w]] for w in ids], dtype=np.int64)
    return X, y


def make_model(n_classes: int = len(PATTERN_LABELS)):
    """4개 conv 블록의 경량 CNN을 만든다.

    설계 의도: 파라미터 약 20만 개로, CPU에서 몇 분 안에 학습되고 5MB 미만으로
        저장되는 규모를 목표로 했다.
        (원래 근거는 클라우드 배포의 메모리 한도였는데 배포를 범위에서 뺐다.
         지금의 근거는 **공정한 대조**다 — LightGBM과 비슷한 규모로 맞춰야
         "딥러닝이 더 낫다/아니다"가 용량 차이가 아닌 방식 차이의 결과가 된다.)
        마지막에 Global Average Pooling을 쓰는 이유는 두 가지다.
          1. Flatten보다 파라미터가 훨씬 적어 과적합이 덜하다
          2. **Grad-CAM이 성립하려면** 마지막 conv의 공간 정보가 살아 있어야 한다
    """
    import torch.nn as nn

    def block(in_ch: int, out_ch: int, n_conv: int = 2) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(n_conv):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    # 첫 블록만 conv 1회인 이유: 연산량은 (해상도² × 채널수)에 비례하는데,
    # 64×64는 이 신경망에서 가장 해상도가 높은 지점이라 여기서 conv를 두 번 하면
    # 전체 학습 시간의 절반을 잡아먹는다. 반면 초반 층이 배우는 것은 '불량 die가
    # 인접해 있는가' 정도의 저수준 특징이라 층을 겹칠 실익이 작다.
    # 대부분의 CNN이 초반에 빠르게 다운샘플하는 것도 같은 이유다.
    return nn.Sequential(
        block(2, 16, n_conv=1),   # 64 → 32
        block(16, 32),            # 32 → 16
        block(32, 64),            # 16 → 8
        block(64, 96),            # 8 → 4   ← Grad-CAM이 참조하는 마지막 conv
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(96, n_classes),
    )


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
    verbose: bool = False,
):
    """CNN 하나를 학습한다.

    클래스 불균형 대응: LightGBM과 동일하게 클래스 가중치를 쓴다.
        가중치 없이 학습하면 손실이 none 클래스에 지배되어 "전부 none" 모델이 된다.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = make_model()

    counts = np.bincount(y_train, minlength=len(PATTERN_LABELS)).astype(np.float32)
    weights = len(y_train) / (len(PATTERN_LABELS) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * max(1, len(X_train) // batch_size + 1)
    )

    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train)
    rng = np.random.default_rng(seed)

    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(Xt))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach().item() * len(idx)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch + 1}/{epochs}  loss={total_loss / len(order):.4f}")

    model.eval()
    return model


def predict_proba(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """확률을 예측한다."""
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(torch.tensor(X[start : start + batch_size]))
            outputs.append(torch.softmax(logits, dim=1).numpy())
    return np.vstack(outputs)


def cross_validate(
    maps: dict[str, np.ndarray],
    labels: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = 0,
    epochs: int = 30,
    verbose: bool = True,
) -> CVReport:
    """계층 교차검증으로 CNN 성능을 측정한다 (LightGBM과 같은 방식·같은 지표)."""
    from sklearn.metrics import confusion_matrix, f1_score, recall_score
    from sklearn.model_selection import StratifiedKFold

    t0 = time.time()
    wafer_ids = list(maps)
    X, y = build_dataset(maps, labels, wafer_ids)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=int)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        model = train_one(X[train_idx], y[train_idx], epochs=epochs, seed=seed + fold)
        oof[valid_idx] = predict_proba(model, X[valid_idx]).argmax(axis=1)
        if verbose:
            fold_f1 = f1_score(y[valid_idx], oof[valid_idx], average="macro", zero_division=0)
            print(f"   fold {fold}/{n_splits}  macro-F1 = {fold_f1:.4f}")

    n_labels = len(PATTERN_LABELS)
    f1_per = f1_score(y, oof, average=None, labels=range(n_labels), zero_division=0)
    recall_per = recall_score(y, oof, average=None, labels=range(n_labels), zero_division=0)
    support = np.bincount(y, minlength=n_labels)

    return CVReport(
        macro_f1=float(f1_score(y, oof, average="macro", zero_division=0)),
        accuracy=float((y == oof).mean()),
        per_class_f1={PATTERN_LABELS[i]: float(f1_per[i]) for i in range(n_labels)},
        per_class_recall={PATTERN_LABELS[i]: float(recall_per[i]) for i in range(n_labels)},
        per_class_support={PATTERN_LABELS[i]: int(support[i]) for i in range(n_labels)},
        confusion=confusion_matrix(y, oof, labels=range(n_labels)).tolist(),
        n_features=IMG_SIZE * IMG_SIZE * 2,
        train_seconds=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────


def grad_cam(model, wafer: np.ndarray, class_idx: int | None = None) -> tuple[np.ndarray, int]:
    """Grad-CAM — 모델이 어느 영역을 보고 판단했는지 열지도로 만든다.

    어떻게 동작하나:
        1. 마지막 conv 층의 출력(특징 지도)과, 그 층에 대한 예측 점수의 기울기를 받는다
        2. 기울기를 공간 방향으로 평균 내 **채널별 중요도**를 구한다
        3. 그 중요도로 특징 지도를 가중합하고 ReLU를 씌운다
           (음수는 "그 클래스가 아니라는 근거"라 제외한다)

    왜 마지막 conv인가: 깊을수록 의미 있는 개념을 담지만 해상도는 떨어진다.
        마지막 conv(4×4)는 "어느 사분면/링을 보았나" 수준의 위치를 알려 준다.
        웨이퍼 패턴 판정 근거로는 이 정도면 충분하다.

    Args:
        model: 학습된 모델
        wafer: 원본 wafer map
        class_idx: 설명할 클래스. None이면 예측 1위 클래스

    Returns:
        (64×64 열지도(0~1로 정규화), 설명한 클래스 인덱스)
    """
    import torch

    x = torch.tensor(to_tensor_input(wafer)[None], requires_grad=False)

    # 마지막 conv 블록 = Sequential의 4번째 요소(index 3)
    features_holder: list[torch.Tensor] = []

    def hook(_module, _inp, out):
        out.retain_grad()
        features_holder.append(out)

    handle = model[3].register_forward_hook(hook)
    try:
        model.zero_grad()
        logits = model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))
        logits[0, class_idx].backward()

        feature_map = features_holder[0]          # (1, 96, 4, 4)
        grads = feature_map.grad                   # (1, 96, 4, 4)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # 채널별 중요도

        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().numpy()
    finally:
        handle.remove()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    return cam, int(class_idx)


# ──────────────────────────────────────────────────────────────────────────
# 저장·로드
# ──────────────────────────────────────────────────────────────────────────


def save(model, report: CVReport, source: str) -> Path:
    import torch

    out_dir = MODELS_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / MODEL_FILE)
    (out_dir / META_FILE).write_text(
        json.dumps(
            {"labels": list(PATTERN_LABELS), "img_size": IMG_SIZE,
             "source": source, "report": asdict(report)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def load(source: str = "synthetic"):
    """저장된 CNN과 메타데이터를 읽는다."""
    import torch

    out_dir = MODELS_DIR / source
    model_path = out_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN 모델이 없습니다: {model_path}\n"
            f"  먼저 학습하세요: python scripts/train_pattern_cnn.py --source {source}"
        )
    model = make_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    return model, meta
