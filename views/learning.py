"""화면 8 — 도움말·학습 (설계서 §5).

이 프로젝트의 코드는 대부분 AI가 썼다. 그러면 **읽는 사람이 이해하지 못한 채 쌓인다.**
이 화면은 그걸 막기 위한 것이다.

    · 무엇을 어떤 순서로 읽어야 하는가  — 읽기 순서
    · 이 코드는 무엇을 왜 그렇게 했나   — 학습노트 8건
    · 정말 이해했는가                   — 확인 문제

읽은 노트는 URL에 남는다(`?done=spc,commonality`). 링크를 저장해 두면
다른 기기에서 이어서 볼 수 있다 — 모바일로 틈틈이 공부하기 위한 장치다.
"""

from __future__ import annotations

import streamlit as st

from wafermap.learning import glossary, notes
from wafermap.ui import layout, milestones, theme

DIFFICULTY_COLORS = {"초급": "#0f766e", "중급": "#1f6feb", "고급": "#7c3aed"}


def _read_progress() -> set[str]:
    """URL에서 읽은 노트 목록을 꺼낸다.

    왜 URL인가 (설계서 §4A.6): 브라우저 저장소를 쓰려면 외부 컴포넌트가 필요하다.
        URL이면 의존성이 없고, **링크를 저장하면 기기 사이를 옮겨 다닐 수 있다.**
        모바일로 보다가 데스크탑에서 이어 보는 것이 목적이다.
    """
    raw = st.query_params.get("done", "")
    return {s for s in raw.split(",") if s}


def _write_progress(done: set[str]) -> None:
    if done:
        st.query_params["done"] = ",".join(sorted(done))
    else:
        st.query_params.pop("done", None)


def _note_body(note: notes.Note) -> None:
    """노트 본문 — 확인 문제만 따로 떼어 접이식으로 만든다."""
    for name, text in note._sections.items():
        if name == "확인 문제":
            continue
        st.markdown(f"## {name}")
        st.markdown(text)

    quiz = note.quiz
    if quiz:
        st.markdown("## 확인 문제")
        st.caption(
            "답을 **먼저 떠올린 뒤** 펼쳐 보라. 읽기만 하면 아는 것 같지만 "
            "설명하려 하면 막히는 지점이 드러난다."
        )
        for i, (question, answer) in enumerate(quiz, 1):
            st.markdown(f"**{i}.** {question}")
            if answer:
                with st.expander("답 보기"):
                    st.markdown(answer)


def _note_view(note: notes.Note, done: set[str]) -> None:
    color = DIFFICULTY_COLORS.get(note.difficulty, theme.MUTED)
    st.subheader(note.title)
    st.markdown(
        theme.badge(note.milestone, theme.GOOD)
        + theme.badge(note.difficulty, color)
        + theme.badge(f"약 {note.minutes}분", theme.MUTED),
        unsafe_allow_html=True,
    )

    if note.prerequisites:
        st.caption(f"먼저 읽으면 좋은 노트: {' · '.join(note.prerequisites)}")
    if note.concepts:
        st.caption(f"다루는 개념: {' · '.join(note.concepts)}")

    is_done = note.slug in done
    if st.checkbox("읽었음", value=is_done, key=f"done_{note.slug}") != is_done:
        done.symmetric_difference_update({note.slug})
        _write_progress(done)
        st.rerun()

    tabs = st.tabs(["노트", "소스 코드", "이 노트의 용어"])
    with tabs[0]:
        _note_body(note)

    with tabs[1]:
        sources = notes.source_code(note)
        if not sources:
            st.info("소스 파일을 찾을 수 없습니다.")
        for path, code in sources.items():
            st.caption(f"`{path}` · {len(code.splitlines()):,}줄")
            st.code(code, language="python", line_numbers=True)

    with tabs[2]:
        found = glossary.find_in_text(note.body)
        if not found:
            st.info("이 노트에서 사전에 있는 용어를 찾지 못했습니다.")
        for term in found:
            with st.expander(f"{term.term} — {term.short}"):
                st.markdown(term.plain)
                if term.why_matters:
                    st.markdown(
                        f'<div class="wm-why"><b>왜 중요한가</b><br>'
                        f'{term.why_matters.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )


def _roadmap(done: set[str]) -> None:
    """읽기 순서 — 선수 노트가 앞에 오도록 정렬된 목록."""
    order = notes.reading_order()
    total_minutes = sum(n.minutes for n in order)
    layout.metric_grid([
        ("학습노트", f"{len(order)}건", "모듈별 코드 해설"),
        ("읽은 노트", f"{len(done & {n.slug for n in order})}건", "URL에 저장된다"),
        ("총 예상 시간", f"약 {total_minutes}분", "확인 문제 포함"),
        ("용어사전", f"{glossary.stats()['합계']}개", "모르는 말은 여기서"),
    ])

    st.markdown("##### 읽는 순서")
    st.caption(
        "선수 노트가 앞에 오도록 정렬했다. 순서를 건너뛰면 앞에서 설명한 개념을 "
        "모른 채 다음 노트를 만나게 된다."
    )
    for i, note in enumerate(order, 1):
        mark = "✅" if note.slug in done else f"{i}."
        color = DIFFICULTY_COLORS.get(note.difficulty, theme.MUTED)
        st.markdown(
            f'<div class="wm-card" style="padding:.55rem .85rem">'
            f'<div class="wm-row"><b>{mark} {note.title}</b></div>'
            f'<div class="wm-row" style="font-size:.8rem">'
            f'{theme.badge(note.milestone, theme.GOOD)}'
            f'{theme.badge(note.difficulty, color)}'
            f'<span style="color:{theme.MUTED}">{note.minutes}분 · '
            f'<code>{note.module}</code></span></div></div>',
            unsafe_allow_html=True,
        )


def _milestone_tree() -> None:
    """마일스톤별로 무엇을 만들었고 어떤 노트가 있는가."""
    grouped = notes.by_milestone()
    for item in milestones.MILESTONES:
        related = grouped.get(item.key, [])
        with st.expander(f"{item.key} · {item.title} — “{item.question}”"):
            st.caption(item.method)
            for metric in item.metrics:
                warn = "" if metric.robust else " ⚠️ 시드 의존"
                st.markdown(f"- **{metric.label}** {metric.value}{warn}")
            if item.reproduce:
                st.caption(f"재현: `{item.reproduce}`")
            if related:
                st.markdown("**학습노트** " + " · ".join(n.title for n in related))
            else:
                st.caption("이 단계의 학습노트는 아직 없다.")


def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("도움말 · 학습")

    st.markdown(
        '<div class="wm-why"><b>왜 이 화면이 있는가</b><br>'
        '이 프로젝트의 코드는 대부분 AI가 썼다. 그러면 <b>읽는 사람이 이해하지 못한 채 '
        '코드만 쌓인다.</b> 학습노트는 모듈마다 "무엇을 / 어떻게 / <b>왜 이렇게</b>"를 '
        '설명하고, 확인 문제로 정말 이해했는지 점검한다.<br>'
        '<span style="color:#4b5563">노트가 인용한 코드는 <b>테스트로 강제되어</b> '
        '실제 소스와 항상 일치한다. 옛 코드를 설명하는 노트는 생길 수 없다.</span></div>',
        unsafe_allow_html=True,
    )

    done = _read_progress()
    catalog = {n.slug: n for n in notes.reading_order()}

    tabs = st.tabs(["읽기 순서", "노트 보기", "마일스톤"])
    with tabs[0]:
        _roadmap(done)
        if done:
            st.caption(
                "진도는 주소창의 `?done=` 에 저장된다. "
                "**이 페이지 링크를 저장해 두면 다른 기기에서 이어서 볼 수 있다.**"
            )
    with tabs[1]:
        slug = st.selectbox(
            "노트", list(catalog),
            format_func=lambda s: (
                f"{'✅ ' if s in done else ''}{catalog[s].milestone} · {catalog[s].title}"
            ),
        )
        st.divider()
        _note_view(catalog[slug], done)
    with tabs[2]:
        _milestone_tree()


render()
