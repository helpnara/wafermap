"""화면 7 — 모델 검증 (설계서 §4.5).

이 화면의 원칙: **한계를 접어 두지 않는다.**

설계서가 이 화면만 "한계·가정 서술 섹션 항상 펼침, 축소 불가"로 정한 이유가 있다.
성능 숫자는 눈에 잘 띄고 한계는 접힌 곳에 넣기 쉽다. 그러면 보는 사람은 숫자만
가져가고, 나중에 재현이 안 될 때 **분석 전체의 신뢰를 잃는다.**
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wafermap.ui import artifacts, layout, milestones, theme

SOURCE = "synthetic"

#: 아직 풀지 못한 한계 — `docs/09_roadmap.md`와 같은 내용을 화면에도 둔다
OPEN_LIMITS: tuple[tuple[str, str, str], ...] = (
    (
        "시드 의존성",
        "채점이 난수 시드 하나에 크게 흔들린다. M3 원인 챔버 Top-1은 시드 8개에서 "
        "67% ± 17%(최소 33%, 최대 83%)였다. 기본 시드 한 번의 결과였던 '100%'를 정정했다.",
        "단일 실행 수치를 인용하지 말 것. `scripts/run_seed_study.py`로 분포를 재라.",
    ),
    (
        "합성 데이터 상한",
        "섭동을 직접 심었으므로 실측보다 신호가 깨끗하다. 실제 FDC에는 결측·센서 드리프트·"
        "레시피 변경이 섞이고, WM-811K 라벨은 사람이 붙여 경계가 모호한 맵이 많다.",
        "이 점수들은 **상한선**이다. M2는 실측으로 다시 잴 수 있고 준비까지 끝냈다 "
        "(`scripts/compare_sources.py`). 측정은 로컬에서 한다 — 원본이 약 2GB다.",
    ),
    (
        "실측 재측정이 되는 것과 안 되는 것",
        "WM-811K에는 **FDC 센서 데이터가 없다.** 맵과 라벨뿐이라 '어느 챔버가 원인이었나'는 "
        "존재하지 않는 정보다. 실측 모드는 실측 맵에 합성 공정 이력을 라벨로 짝지어 붙인다.",
        "M2만 진짜 재측정이다. M3~M5는 정답지가 여전히 심어 둔 값이라 "
        "**강건성 관찰**이지 실측 성능이 아니다.",
    ),
    (
        "맵 크기 지름길 — 합성으로는 못 보는 부류",
        "합성은 맵이 전부 34×67이라, 맵 크기에만 반응하는 피처가 있어도 값이 상수가 되어 "
        "모델이 무시한다. 즉 **합성으로는 원리적으로 볼 수 없다.** 실제로 3건을 찾아 "
        "고쳤다(피처 121 → 99종). `radon_mean_*` 20개는 값이 `1/맵대각선` 하나였다.",
        "이산화 잔여분 5개는 남아 있다. 실측 학습 후 이 피처들의 중요도가 튀면 "
        "맵 크기를 학습한 것으로 의심해야 한다 — `docs/06_local_validation.md` §5.",
    ),
    (
        "상관 ≠ 인과",
        "모든 개선안이 상관 기반이다. \"이 구간으로 옮기면 좋아진다\"가 아니라 "
        "\"이 구간에 있던 웨이퍼는 불량이 적었다\"이다.",
        "확증은 DOE(실험계획)로 해야 한다. 이 분석은 **DOE 대상을 좁히는 도구**다.",
    ),
    (
        "카운터의 시각 교락",
        "`pad_life`·`touchdown_count`처럼 단조 증가하는 카운터는 사실상 **시각의 대리 변수**다. "
        "불량이 특정 기간에 몰리면 무관한 불량에서도 원인처럼 보인다. "
        "Scratch의 Top-1 오답이 이 때문이다.",
        "미해결. 시간 층화 또는 카운터 파라미터 분리 취급이 필요하다.",
    ),
    (
        "검사 기인 '여부'는 판정 불가",
        "어느 프로브 카드인지는 찾지만(2/2), 검사 기인이 **있는지 없는지**는 못 가린다. "
        "검사 축 오즈비가 진짜 원인이 없을 때 오히려 컸다(4.20 vs 2.23).",
        "확정은 다른 카드로 재측정해야 한다. 분석은 그 대상을 좁힐 뿐이다.",
    ),
    (
        "검정력 부족",
        "6,000장(240 lot)에서는 다중검정 보정 후 유의 판정이 0~1개뿐이다. "
        "순위는 맞지만 통계적 확증은 못 한다.",
        "표본을 늘리거나 실데이터로 재측정해야 한다.",
    ),
    (
        "SPC 위양성 판정 여지",
        "전체 121일 중 89일(74%)이 이상 구간이라 '위양성 0건'을 성능으로 볼 수 없다.",
        "정상 기간 비중을 늘린 시나리오로 재측정 필요.",
    ),
    (
        "스파이크 검출의 해상도 가정",
        "60초를 60점으로 본다고 가정했다. 실측 FDC 로그가 이보다 성기면 "
        "`time_above`·`n_excursions`가 무의미해진다.",
        "실측 로그 주기를 확인한 뒤 재설계해야 한다.",
    ),
)


@st.cache_data(show_spinner=False)
def _rootcause(source: str) -> dict | None:
    try:
        return artifacts.load_rootcause(source)
    except FileNotFoundError:
        return None


def _metric_table() -> None:
    rows = []
    for item in milestones.MILESTONES:
        for metric in item.metrics:
            rows.append({
                "단계": item.key,
                "지표": metric.label,
                "값": metric.value,
                "측정 조건": metric.detail or "단일 실행",
                "시드 안정": "✅" if metric.robust else "⚠️ 흔들림",
            })
    table = pd.DataFrame(rows)
    # 높이를 행 수에 맞춘다: 한계 목록과 마찬가지로 "스크롤해야 보이는 수치"를
    # 만들지 않기 위해서다. 기본 높이는 10행에서 잘려 뒤쪽 단계가 가려진다.
    st.dataframe(table, hide_index=True, width="stretch",
                 height=(len(table) + 1) * 35 + 3)


def _pattern_table(data: dict) -> None:
    rows = []
    for pattern, item in data["patterns"].items():
        truth = item["truth"]
        process = next(
            (a for a in item["axes"] if a["axis"] == "process"), None
        )
        found = process["chamber_id"] if process else "—"
        hit = found in truth["equip"]
        rows.append({
            "패턴": pattern,
            "불량": item["n_case"],
            "정답 스텝": f"{truth['process_step']} {truth['process_step_name']}",
            "공정 축 1위": found,
            "적중": "✅" if hit else "❌",
            "모델 AUC": item["model"].get("auc"),
        })
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config={"모델 AUC": st.column_config.NumberColumn(format="%.3f")},
    )
    hits = sum(1 for r in rows if r["적중"] == "✅")
    st.caption(
        f"이 시드에서 {hits}/{len(rows)} 적중. "
        "**이 한 번의 결과를 성능이라고 부르면 안 된다** — 위의 시드 분포를 함께 보라."
    )


def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("모델 검증")

    st.markdown(
        '<div class="wm-why"><b>무엇을 검증하는가</b><br>'
        '이 프로젝트는 시뮬레이터가 <b>원인을 심어 두고</b> 분석이 그것을 찾아내는지 '
        '채점한다. 정답지(<code>ground_truth</code>)는 분석 모델에 절대 넣지 않고 '
        '채점에만 쓴다.<br>'
        '<span style="color:#4b5563">실데이터에는 정답지가 없다. 그래서 합성 데이터로 '
        '<b>방법이 작동하는지</b>를 먼저 확인하는 것이다.</span></div>',
        unsafe_allow_html=True,
    )

    st.subheader("단계별 측정 결과")
    st.caption(
        "여러 시드의 평균 ± 표준편차다. 합성 데이터의 채점은 시드 하나에 크게 흔들려서, "
        "단일 실행 수치는 인용하지 않는다."
    )
    _metric_table()

    st.markdown(
        '<div class="wm-note"><b>왜 시드 분포로 말하는가</b> ★<br>'
        '처음에는 기본 시드 한 번의 결과를 성능으로 적었다. M5.5-② 작업 중 성능이 '
        '떨어진 줄 알고 추적하다 <b>그 시드가 운이 좋았을 뿐</b>임을 발견했다.<br>'
        '포트폴리오에 "원인 설비를 100% 찾아냅니다"라고 적어 두면, 다른 시드로 돌려 본 '
        '사람은 67%를 보고 <b>재현이 안 된다</b>고 판단한다. 그 순간 잃는 것은 그 수치 '
        '하나가 아니라 분석 전체의 신뢰다.</div>',
        unsafe_allow_html=True,
    )

    data = _rootcause(SOURCE)
    if data:
        st.divider()
        st.subheader("이 데이터셋에서의 패턴별 결과")
        _pattern_table(data)

    # ── 한계 — 항상 펼쳐 둔다 (설계서 §4.5) ────────────────────────────
    st.divider()
    st.subheader("아직 풀지 못한 한계")
    st.markdown(
        '<div class="wm-note">이 목록은 <b>접히지 않는다.</b> 성능 숫자는 눈에 잘 띄고 '
        '한계는 접힌 곳에 넣기 쉽다. 그러면 보는 사람은 숫자만 가져간다. '
        '수치를 인용할 때 이 표를 함께 봐야 한다.</div>',
        unsafe_allow_html=True,
    )

    n_cols = 1 if layout.is_mobile() else 2
    for start in range(0, len(OPEN_LIMITS), n_cols):
        block = OPEN_LIMITS[start:start + n_cols]
        for col, (title, body, action) in zip(st.columns(n_cols), block):
            with col:
                st.markdown(
                    f'<div class="wm-card">'
                    f'<h4>⚠️ {title}</h4>'
                    f'<div class="wm-row" style="color:#374151">{theme.html(body)}</div>'
                    f'<div class="wm-row" style="color:{theme.GOOD};margin-top:.4rem">'
                    f'→ {theme.html(action)}</div></div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown(
        '<div class="wm-why"><b>그래서 이 프로젝트를 어떻게 읽어야 하나</b><br>'
        '숫자 자체보다 <b>어떤 함정을 발견하고 어떻게 고쳤는지</b>가 이 프로젝트의 내용이다. '
        '지름길 학습, 관리도 오적용, 사이클 타임 불일치, 표본 단위 부풀림, 계측값 함정, '
        '자기모순 데이터, 시드 의존성 — 모두 처음엔 좋아 보이는 결과 뒤에 숨어 있었다.<br>'
        '<span style="color:#4b5563">자세한 경위는 <code>docs/04_results.md</code>에 '
        '발견 순서대로 기록돼 있다.</span></div>',
        unsafe_allow_html=True,
    )


render()
