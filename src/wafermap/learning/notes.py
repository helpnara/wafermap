"""학습노트 읽기 — 화면이 마크다운 파일을 구조화해 쓸 수 있게 한다 (M6-5).

왜 별도 모듈인가: 노트는 `docs/learning_notes/*.md`가 원본이다. 화면이 그 파일을
    직접 파싱하면 규칙이 화면 코드에 흩어진다. 그리고 **테스트가 이 규칙을 검증**하고
    있으므로(`tests/test_learning_notes.py`) 같은 규칙을 한곳에서 쓰는 편이 안전하다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from wafermap.config import LEARNING_NOTES_DIR, PROJECT_ROOT

#: 노트가 반드시 갖춰야 하는 섹션 (설계서 §5.3)
REQUIRED_SECTIONS = ("이 코드는 무엇을 하나", "핵심 로직", "왜 이렇게 만들었나", "확인 문제")


@dataclass(frozen=True)
class Note:
    """학습노트 1건.

    Attributes:
        slug: 파일명(확장자 제외)
        title: 첫 번째 `#` 제목
        module: 설명 대상 소스 파일
        also_covers: 함께 인용하는 보조 모듈
        milestone: 어느 마일스톤에서 만든 코드인가
        difficulty: 난이도
        concepts: 다루는 개념 목록
        prerequisites: 먼저 읽어야 할 노트 slug
        minutes: 예상 소요 시간
        body: 제목 아래 본문 (frontmatter 제외)
    """

    slug: str
    title: str
    module: str
    milestone: str
    body: str
    also_covers: tuple[str, ...] = ()
    difficulty: str = ""
    concepts: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    minutes: int = 0
    _sections: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def modules(self) -> tuple[str, ...]:
        return (self.module, *self.also_covers)

    def section(self, name: str) -> str:
        """`## 이름` 섹션의 본문을 돌려준다 (없으면 빈 문자열).

        **접두 일치**로 찾는다. 노트마다 제목에 부제를 붙이기 때문이다 —
        예: `## 왜 이렇게 만들었나 — 설계 선택과 대안`. 정확히 일치를 요구하면
        멀쩡한 섹션을 못 찾는다.
        """
        if name in self._sections:
            return self._sections[name]
        for heading, body in self._sections.items():
            if heading.startswith(name):
                return body
        return ""

    @property
    def quiz(self) -> list[tuple[str, str]]:
        """확인 문제 목록 — (문제, 정답).

        노트는 `**1.** 질문` 다음에 `<details>` 접이식 정답을 둔다. 화면에서는
        `st.expander`로 바꿔야 하므로 여기서 쪼갠다.
        """
        text = self.section("확인 문제")
        if not text:
            return []
        out: list[tuple[str, str]] = []
        blocks = re.split(r"^\*\*\d+\.\*\*", text, flags=re.M)[1:]
        for block in blocks:
            answer = ""
            match = re.search(r"<details>(.*?)</details>", block, re.S)
            if match:
                # `<summary>정답</summary>` 를 **한 덩어리로** 지운다.
                # 여는 태그와 닫는 태그를 따로 지우면(`</?summary>[^<]*`) 닫는 태그
                # 뒤의 `[^<]*` 가 답 본문 전체를 삼켜 버린다 — 실제로 그랬다.
                answer = re.sub(
                    r"<summary>.*?</summary>", "", match.group(1), flags=re.S
                ).strip()
                block = block[: match.start()]
            out.append((block.strip(), answer))
        return out


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def _split_list(value: str) -> tuple[str, ...]:
    """`[a, b, c]` 또는 `a, b` 형태를 튜플로."""
    cleaned = value.strip().strip("[]")
    return tuple(part.strip() for part in cleaned.split(",") if part.strip())


def _split_sections(body: str) -> dict[str, str]:
    """`## 제목` 단위로 쪼갠다."""
    sections: dict[str, str] = {}
    current, buffer = None, []
    for line in body.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current, buffer = heading.group(1), []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


def load(slug: str) -> Note:
    """노트 하나를 읽는다.

    Raises:
        FileNotFoundError: 그런 노트가 없을 때
    """
    path = LEARNING_NOTES_DIR / f"{slug}.md"
    if not path.exists():
        raise FileNotFoundError(f"학습노트가 없습니다: {path}")

    meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    title_match = re.search(r"^#\s+(.+?)\s*$", body, re.M)
    title = title_match.group(1) if title_match else slug
    if title_match:
        body = body[title_match.end():].lstrip()

    return Note(
        slug=slug,
        title=title,
        module=meta.get("module", ""),
        also_covers=_split_list(meta.get("also_covers", "")),
        milestone=meta.get("milestone", ""),
        difficulty=meta.get("difficulty", ""),
        concepts=_split_list(meta.get("concepts", "")),
        prerequisites=_split_list(meta.get("prerequisites", "")),
        minutes=int(meta.get("estimated_minutes", "0") or 0),
        body=body,
        _sections=_split_sections(body),
    )


def all_notes() -> tuple[Note, ...]:
    """모든 노트를 마일스톤 → 제목 순으로."""
    notes = [load(p.stem) for p in sorted(LEARNING_NOTES_DIR.glob("*.md"))]
    return tuple(sorted(notes, key=lambda n: (n.milestone, n.title)))


def by_milestone() -> dict[str, list[Note]]:
    """마일스톤별로 묶는다 — 학습 순서를 트리로 보여주기 위해."""
    grouped: dict[str, list[Note]] = {}
    for note in all_notes():
        grouped.setdefault(note.milestone, []).append(note)
    return grouped


def reading_order() -> list[Note]:
    """선수 노트가 앞에 오도록 정렬한다.

    왜 필요한가: 노트마다 `prerequisites`가 있다. 순서를 무시하고 읽으면
        앞 노트에서 설명한 개념을 모른 채 뒤 노트를 만나게 된다.
    """
    notes = {n.slug: n for n in all_notes()}
    ordered: list[Note] = []
    seen: set[str] = set()

    def visit(slug: str, stack: tuple[str, ...] = ()) -> None:
        if slug in seen or slug not in notes or slug in stack:
            return  # 순환 참조는 조용히 건너뛴다 (테스트가 따로 막는다)
        for prerequisite in notes[slug].prerequisites:
            visit(prerequisite, stack + (slug,))
        seen.add(slug)
        ordered.append(notes[slug])

    for slug in notes:
        visit(slug)
    return ordered


def source_code(note: Note) -> dict[str, str]:
    """노트가 다루는 소스 파일의 실제 내용.

    왜 화면에서 원본을 보여주나: 노트는 코드의 **일부만** 인용한다. 전체를 같이
        볼 수 있어야 "인용된 부분이 어디에 있는지"가 이해된다.
    """
    out: dict[str, str] = {}
    for relative in note.modules:
        path = PROJECT_ROOT / relative
        if path.exists():
            out[relative] = path.read_text(encoding="utf-8")
    return out
