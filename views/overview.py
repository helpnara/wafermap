"""화면 0 — 개요 (설계서 §4.5).

이 화면이 답하는 것:
    1. 이 시스템은 무엇을 하는가 — 파이프라인 한 눈에
    2. 어떤 데이터를 쓰는가 — 규모·기간·라벨 분포
    3. 얼마나 잘 하는가 — 단계별 측정 결과 (분포로)
    4. 어디부터 보면 되는가 — 안내
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wafermap.data import loader
from wafermap.ui import layout, milestones, theme

SOURCE = "synthetic"


@st.cache_data(show_spinner="데이터를 불러오는 중…")
def _summary(source: str) -> dict:
    return loader.summary(source)


@st.cache_data(show_spinner=False)
def _mechanisms(source: str) -> pd.Series:
    gt = loader.load_ground_truth(source)
    return gt["cause_mechanism"].value_counts()


def _pipeline() -> None:
    """파이프라인 흐름도 — 이 프로젝트의 서사를 한 줄로."""
    mobile = layout.is_mobile()
    cells = []
    for i, (icon, title, subtitle) in enumerate(milestones.PIPELINE):
        arrow = "" if i == len(milestones.PIPELINE) - 1 else (
            '<div class="wm-flow-arrow">↓</div>' if mobile
            else '<div class="wm-flow-arrow">→</div>'
        )
        cells.append(
            f'<div class="wm-flow-step"><div class="wm-flow-icon">{icon}</div>'
            f'<div class="wm-flow-title">{title}</div>'
            f'<div class="wm-flow-sub">{subtitle}</div></div>{arrow}'
        )
    direction = "column" if mobile else "row"
    st.markdown(
        f'<div class="wm-flow" style="flex-direction:{direction}">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def _label_chart(counts: dict[str, int]) -> None:
    import plotly.graph_objects as go

    # pandas 1.x 호환 — reset_index(names=...)는 2.0부터다
    frame = pd.DataFrame(
        {"label": list(counts), "n": list(counts.values())}
    ).sort_values("n", ascending=True)
    colors = [theme.MUTED if label == "none" else theme.GOOD for label in frame["label"]]
    total = frame["n"].sum()

    fig = go.Figure(
        go.Bar(
            x=frame["n"], y=frame["label"], orientation="h", marker_color=colors,
            text=[f"{n:,} ({n / total:.1%})" for n in frame["n"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:,}장<extra></extra>",
        )
    )
    fig.update_layout(
        # 위 여백을 두는 이유: 모바일에서 Plotly 툴바가 차트 위에 겹쳐 그려져
        # 맨 윗 막대의 값 라벨을 가린다.
        height=340, margin=dict(l=10, r=80, t=34, b=10),
        plot_bgcolor="white", showlegend=False, xaxis_title="웨이퍼 수",
    )
    # 로그 눈금의 보조 눈금(2,5,20,50…)은 읽기 어렵다. 10의 거듭제곱만 남긴다.
    fig.update_xaxes(
        gridcolor=theme.BORDER, type="log",
        tickmode="array", tickvals=[1, 10, 100, 1000, 10000],
        ticktext=["1", "10", "100", "1,000", "10,000"],
    )
    st.plotly_chart(
        fig, use_container_width=True,
        # 좁은 화면에서는 툴바가 자리만 차지한다
        config={"displayModeBar": not layout.is_mobile()},
    )


def _milestone_card(item: milestones.Milestone) -> None:
    rows = []
    for metric in item.metrics:
        warn = "" if metric.robust else theme.badge("시드 의존", theme.WARN)
        detail = f' <span style="color:{theme.MUTED}">· {metric.detail}</span>' if metric.detail else ""
        rows.append(
            f'<div class="wm-row"><b>{metric.label}</b> — '
            f'<span style="font-size:1.05rem;font-weight:700">{metric.value}</span>'
            f'{detail} {warn}</div>'
        )
    st.markdown(
        f'<div class="wm-card">'
        f'<h4>{item.key} · {item.title}</h4>'
        f'<div class="wm-row" style="color:#374151">“{item.question}”</div>'
        f'<div class="wm-row" style="color:{theme.MUTED};font-size:.8rem">{item.method}</div>'
        f'<div style="margin-top:.45rem">{"".join(rows)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if item.reproduce:
        st.caption(f"재현: `{item.reproduce}`")


def render() -> None:
    st.markdown(theme.CSS + OVERVIEW_CSS, unsafe_allow_html=True)
    st.title("EDS Wafer Map 분석 시스템")
    st.caption("DRAM(1z-nm) 가상 라인 · 불량 패턴에서 개선안까지 근거를 이어 붙인다")

    try:
        summary = _summary(SOURCE)
    except FileNotFoundError as exc:
        layout.missing_artifact(exc, what="개요 화면")
        return

    start, end = summary["period"]
    defect = sum(n for label, n in summary["label_counts"].items() if label != "none")
    layout.metric_grid([
        ("웨이퍼", f"{summary['n_wafers']:,}장", f"{summary['n_lots']}개 lot"),
        ("평균 수율", f"{summary['mean_yield']:.2f}%", "die 기준"),
        ("불량 패턴", f"{defect / summary['n_wafers']:.1%}",
         f"{defect:,}장 · WM-811K 실측 비율 재현"),
        ("기간", f"{(end - start).days}일", f"{start:%Y-%m-%d} ~ {end:%m-%d}"),
    ])

    st.markdown(
        '<div class="wm-why"><b>이 시스템이 답하는 질문</b><br>'
        '"EDS 검사에서 불량이 났을 때, 웨이퍼 맵에 어떤 패턴이 나타났고, '
        '그 웨이퍼가 거쳐 간 공정·설비·센서 데이터 중 무엇이 불량과 관련이 높은가?"<br>'
        '<span style="color:#4b5563">AI가 원인을 확정하지 않는다. '
        '원인 후보와 근거를 제시하고, 공정 전문가가 최종 검증한다.</span></div>',
        unsafe_allow_html=True,
    )

    _pipeline()
    st.divider()

    with layout.responsive_split(("데이터", "불량의 네 가지 경로"), ratio=(1.0, 1.0)) as (left, right):
        with left:
            st.subheader("라벨 분포")
            st.caption(
                "WM-811K 실측 비율을 그대로 재현했다. none이 85%, Near-full은 0.08%뿐 — "
                "**이 극심한 불균형 자체가 모델링의 난이도**다. (가로축 로그 눈금)"
            )
            _label_chart(summary["label_counts"])

        with right:
            st.subheader("같은 패턴, 다른 원인")
            st.caption(
                "웨이퍼 맵만 봐서는 구분되지 않지만 **조치가 완전히 다르다.** "
                "시뮬레이터가 네 경로를 모두 만들고, 분석이 갈라내는지 채점한다."
            )
            counts = _mechanisms(SOURCE)
            labels = {
                "process": ("공정 기인", "챔버 파라미터가 지속적으로 이탈"),
                "test": ("검사 기인", "프로브 카드 마모 — 칩은 멀쩡하다"),
                "spike": ("순간 스파이크", "2~3초만 튄다. 평균은 정상"),
                "drift": ("드리프트", "처리 중 서서히 밀린다"),
                "interaction": ("교호작용", "각각은 규격 안, 조합이 문제"),
            }
            for key, (name, desc) in labels.items():
                n = int(counts.get(key, 0))
                if not n:
                    continue
                st.markdown(
                    f'<div class="wm-card" style="padding:.6rem .85rem">'
                    f'<div class="wm-row"><b>{name}</b> · {n:,}장</div>'
                    f'<div class="wm-row" style="color:{theme.MUTED};font-size:.82rem">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("단계별 측정 결과")
    st.markdown(
        '<div class="wm-note">모든 수치는 시뮬레이터가 심어 둔 <b>정답지와 대조한 채점</b>이며, '
        '여러 시드의 <b>평균 ± 표준편차</b>다. 합성 데이터의 채점은 시드 하나에 크게 흔들려서, '
        '단일 실행 수치는 인용하지 않는다. '
        '<span style="color:#92400e">"시드 의존" 배지가 붙은 지표는 표준편차를 함께 읽을 것.</span></div>',
        unsafe_allow_html=True,
    )

    n_cols = 1 if layout.is_mobile() else 2
    items = list(milestones.MILESTONES)
    for start_idx in range(0, len(items), n_cols):
        for col, item in zip(st.columns(n_cols), items[start_idx:start_idx + n_cols]):
            with col:
                _milestone_card(item)

    st.divider()
    st.markdown(
        '<div class="wm-why"><b>어디부터 보면 되나</b><br>'
        '왼쪽 메뉴에서 <b>개선방안·기대효과</b>로 가면 이 파이프라인의 결론을 볼 수 있다. '
        '조치안·근거 차트·ROI 계산기가 한 화면에 있다.<br>'
        '<span style="color:#4b5563">코드를 공부하려면 <code>docs/05_learning_guide.md</code>의 '
        '읽기 순서를, 분석의 한계는 <code>docs/04_results.md</code>를 보라.</span></div>',
        unsafe_allow_html=True,
    )


OVERVIEW_CSS = """
<style>
  .wm-flow { display:flex; align-items:stretch; gap:.4rem; margin:.6rem 0 1rem 0; }
  .wm-flow-step {
    flex:1; border:1px solid #dfe3ea; border-radius:10px; padding:.7rem .6rem;
    background:#fff; text-align:center; min-width:0;
  }
  .wm-flow-icon { font-size:1.35rem; line-height:1.6; }
  .wm-flow-title { font-weight:700; font-size:.9rem; color:#111827; }
  .wm-flow-sub { font-size:.74rem; color:#6b7280; margin-top:.15rem; line-height:1.35; }
  .wm-flow-arrow {
    display:flex; align-items:center; color:#9ca3af; font-size:1.1rem; padding:0 .1rem;
  }
  @media (max-width: 640px) {
    .wm-flow-step { padding:.55rem; }
    .wm-flow-sub { font-size:.8rem; }
    .wm-flow-arrow { justify-content:center; padding:.1rem 0; }
  }
</style>
"""


render()
