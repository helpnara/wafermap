"""화면 3 — 패턴 분류 모델 (설계서 §4.5).

이 화면이 답하는 것:
    1. 얼마나 잘 맞히나 — 지표와 혼동행렬
    2. **어디서 틀리나** — 클래스별 성능과 혼동 쌍
    3. 무엇을 보고 판단하나 — 피처 중요도
    4. 딥러닝이 더 낫나 — LGBM vs CNN

전체 정확도(99.6%)는 의미가 없다. `none`이 85%라 **아무것도 안 하고 전부 정상이라
답해도 85%**가 나오기 때문이다. 그래서 macro-F1과 클래스별 recall을 본다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from wafermap.config import MODELS_DIR
from wafermap.ui import layout, theme

SOURCE = "synthetic"


@st.cache_data(show_spinner=False)
def _meta(name: str, source: str) -> dict | None:
    path = MODELS_DIR / source / f"{name}_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _confusion_figure(report: dict, *, normalize: bool) -> None:
    import plotly.graph_objects as go

    labels = report["labels"]
    matrix = np.array(report["confusion"], dtype=float)
    support = matrix.sum(axis=1, keepdims=True)

    if normalize:
        # 왜 행 정규화인가: 클래스별 표본이 5장~5,115장으로 1,000배 차이 난다.
        # 원시 개수로 보면 none 행만 진하고 나머지는 전부 하얗게 보인다.
        shown = np.divide(matrix, support, out=np.zeros_like(matrix), where=support > 0) * 100
        text = [[f"{v:.0f}%" if v else "" for v in row] for row in shown]
        title = "행 기준 비율 (실제 클래스 중 몇 %를 이렇게 예측했나)"
    else:
        shown, text = matrix, [[f"{int(v)}" if v else "" for v in row] for row in matrix]
        title = "원시 개수"

    fig = go.Figure(go.Heatmap(
        z=shown, x=labels, y=labels, text=text, texttemplate="%{text}",
        colorscale=[[0.0, "#ffffff"], [1.0, theme.GOOD]], showscale=False,
        hovertemplate="실제 %{y} → 예측 %{x}<br>%{text}<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="예측", yaxis_title="실제", title=title,
        plot_bgcolor="white",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


def _per_class_chart(report: dict) -> None:
    import plotly.graph_objects as go

    frame = pd.DataFrame({
        "label": report["labels"],
        "f1": [report["per_class_f1"][k] for k in report["labels"]],
        "recall": [report["per_class_recall"][k] for k in report["labels"]],
        "support": [report["per_class_support"][k] for k in report["labels"]],
    }).sort_values("f1")

    colors = [theme.BAD if v < 0.95 else theme.GOOD for v in frame["f1"]]
    fig = go.Figure(go.Bar(
        x=frame["f1"], y=frame["label"], orientation="h", marker_color=colors,
        text=[f"{v:.3f} ({n:,}장)" for v, n in zip(frame["f1"], frame["support"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>F1 %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=360, margin=dict(l=10, r=110, t=30, b=10),
        plot_bgcolor="white", xaxis_title="F1 점수", xaxis_range=[0, 1.15],
    )
    fig.update_xaxes(gridcolor=theme.BORDER)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


def _worst_pairs(report: dict, top_n: int = 5) -> None:
    """가장 많이 헷갈린 쌍 — 어디서 틀리는지가 성능 숫자보다 유용하다."""
    labels = report["labels"]
    matrix = np.array(report["confusion"], dtype=float)
    support = matrix.sum(axis=1)

    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i == j or matrix[i, j] == 0:
                continue
            rows.append({
                "실제": actual,
                "예측": predicted,
                "건수": int(matrix[i, j]),
                "그 클래스의 몇 %": matrix[i, j] / support[i] * 100 if support[i] else 0,
            })
    if not rows:
        st.success("혼동된 건이 없습니다.")
        return
    frame = pd.DataFrame(rows).sort_values("건수", ascending=False).head(top_n)
    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config={
            "그 클래스의 몇 %": st.column_config.NumberColumn(format="%.1f%%")
        },
    )


def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("패턴 분류 모델")

    lgbm = _meta("pattern_lgbm", SOURCE)
    if lgbm is None:
        st.error(
            "모델이 없습니다. `python scripts/train_pattern_model.py` 를 먼저 실행하세요."
        )
        return
    report = lgbm["report"]
    cnn = _meta("pattern_cnn", SOURCE)

    st.markdown(
        '<div class="wm-why"><b>정확도를 보지 말 것</b> ★<br>'
        f'이 모델의 정확도는 <b>{report["accuracy"]:.1%}</b>다. 그런데 이 데이터는 '
        '정상(none)이 85%라서 <b>아무것도 안 하고 전부 정상이라 답해도 85%</b>가 나온다.<br>'
        '<span style="color:#4b5563">그래서 클래스마다 같은 가중치를 주는 '
        '<b>macro-F1</b>과 <b>클래스별 recall</b>을 본다. 희소 클래스를 놓치는지가 '
        '여기서 드러난다.</span></div>',
        unsafe_allow_html=True,
    )

    layout.metric_grid([
        ("macro-F1", f"{report['macro_f1']:.4f}", "클래스 균등 가중"),
        ("정확도", f"{report['accuracy']:.1%}", "⚠️ 불균형 때문에 과대평가"),
        ("피처", f"{report['n_features']}종", "기하·연결성분·Radon"),
        ("클래스", f"{len(report['labels'])}종",
         f"최소 {min(report['per_class_support'].values())}장 ~ "
         f"최대 {max(report['per_class_support'].values()):,}장"),
    ])

    st.divider()
    with layout.responsive_split(("클래스별 성능", "혼동행렬"), ratio=(1.0, 1.2)) as (left, right):
        with left:
            st.subheader("클래스별 F1")
            _per_class_chart(report)
            st.caption(
                "주황은 0.95 미만이다. **Scratch가 가장 어렵다** — 선형 긁힘은 "
                "방향과 길이가 제각각이고 표본도 41장뿐이라, 형상 피처만으로는 "
                "Loc(국부 뭉침)과 구분이 잘 안 된다."
            )
        with right:
            st.subheader("혼동행렬")
            normalize = st.toggle("행 기준 비율로 보기", value=True)
            _confusion_figure(report, normalize=normalize)

    st.markdown("##### 가장 많이 헷갈린 쌍")
    st.caption(
        "성능 숫자보다 **어디서 틀리는지**가 유용하다. 혼동 쌍이 물리적으로 "
        "닮은 패턴이면 모델이 제대로 배운 것이고, 엉뚱한 쌍이면 피처를 의심해야 한다."
    )
    _worst_pairs(report)

    if cnn:
        st.divider()
        st.subheader("LightGBM vs 경량 CNN")
        cnn_report = cnn["report"]
        layout.metric_grid([
            ("LightGBM macro-F1", f"{report['macro_f1']:.4f}", "기하 피처 121종"),
            ("CNN macro-F1", f"{cnn_report['macro_f1']:.4f}",
             f"이미지 {cnn.get('img_size', '?')}px 직접 입력"),
        ], desktop_cols=2)
        st.markdown(
            '<div class="wm-note"><b>딥러닝이 항상 낫지는 않다</b><br>'
            '이 문제에서는 <b>사람이 만든 기하 피처가 더 잘 맞혔고 100배 이상 빨랐다.</b> '
            '웨이퍼 맵은 34×67 저해상도 이진 격자라 CNN이 배울 미세 질감이 없고, '
            '희소 클래스(Near-full 5장)에서 학습할 표본도 부족하다.<br>'
            '<span style="color:#4b5563">"이미지니까 CNN"이 아니라 '
            '<b>데이터의 성격을 보고 고르는 것</b>이 맞다.</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<div class="wm-note"><b>이 수치의 한계</b><br>'
        '합성 데이터라 패턴이 실측보다 깨끗하다. 실제 WM-811K는 사람이 라벨을 붙였고 '
        '경계가 모호한 맵이 많아 <b>이 점수는 상한선으로 봐야 한다.</b><br>'
        '그리고 피처에서 <b>맵 크기 관련 값을 모두 제거</b>했다 — 초기에 '
        '<code>map_aspect</code>가 상위 판별자로 나왔는데, 그건 패턴이 아니라 '
        '데이터 출처를 학습한 <b>지름길 학습</b>이었기 때문이다.</div>',
        unsafe_allow_html=True,
    )


render()
