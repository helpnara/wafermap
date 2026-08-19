"""용어사전 검증.

사전은 '읽는 자료'라 버그가 나도 프로그램이 죽지 않는다. 대신 조용히 잘못된 것을
가르친다. 그래서 아래를 자동으로 검사한다.

  · 관련어 링크가 실제로 존재하는가 (깨진 링크 = 클릭했더니 없는 용어)
  · 별칭이 다른 용어와 충돌하지 않는가 (검색이 엉뚱한 것을 찾아 준다)
  · 프로젝트 핵심 어휘가 빠지지 않았는가
  · 비전문가용 설명이 실제로 쉬운가 (길이·구조로 근사 검사)
"""

from __future__ import annotations

import pytest

from wafermap.learning import glossary as G

#: 이 프로젝트를 설명할 때 반드시 나오는 어휘 — 하나라도 빠지면 안 된다
REQUIRED_TERMS = (
    # 반도체 기본
    "웨이퍼", "die", "lot", "수율", "EDS", "wafer map", "Bin",
    # 공정·설비
    "FDC", "챔버", "CMP", "식각", "포토", "PM", "레시피", "파라미터",
    # 불량 패턴 9종
    "Center", "Donut", "Edge-Ring", "Edge-Loc", "Loc", "Scratch",
    "Random", "Near-full", "none",
    # 통계
    "SPC", "커미널리티 분석", "p-value", "오즈비", "다중검정", "교락",
    # 머신러닝
    "피처", "LightGBM", "CNN", "macro-F1", "클래스 불균형",
    "데이터 누출", "SHAP", "Grad-CAM",
    # 데이터
    "WM-811K", "합성 데이터", "정답지",
)

#: 산업적 의미를 반드시 설명해야 하는 용어 (학습이 목적이므로)
NEEDS_WHY = (
    "수율", "EDS", "wafer map", "FDC", "챔버", "PM",
    "불량 패턴", "SPC", "커미널리티 분석", "다중검정", "교락",
    "macro-F1", "클래스 불균형", "데이터 누출", "SHAP", "지름길 학습",
    "WM-811K", "합성 데이터", "정답지",
)


# ── 구조 무결성 ──────────────────────────────────────────────────────────


def test_all_categories_are_declared():
    """모든 용어의 분류가 CATEGORIES에 선언되어 있어야 한다."""
    valid = set(G.categories())
    for term in G.all_terms():
        assert term.category in valid, f"{term.term}: 알 수 없는 분류 '{term.category}'"


def test_no_duplicate_terms():
    names = [t.term for t in G.all_terms()]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"중복된 용어: {duplicates}"


def test_related_links_resolve():
    """관련어가 모두 사전에 존재해야 한다 (깨진 링크 방지)."""
    known = {t.term for t in G.all_terms()}
    broken: list[str] = []
    for term in G.all_terms():
        for rel in term.related:
            if rel not in known:
                broken.append(f"{term.term} → {rel}")
    assert not broken, "존재하지 않는 용어를 참조합니다:\n  " + "\n  ".join(broken)


def test_aliases_do_not_collide():
    """서로 다른 용어가 같은 별칭을 쓰면 검색이 엉뚱한 것을 찾는다."""
    seen: dict[str, str] = {}
    collisions: list[str] = []

    for term in G.all_terms():
        for alias in (term.term, *term.aliases):
            key = G._normalize(alias)
            if key in seen and seen[key] != term.term:
                collisions.append(f"'{alias}' → {seen[key]} vs {term.term}")
            seen[key] = term.term

    assert not collisions, "별칭 충돌:\n  " + "\n  ".join(collisions)


def test_no_self_reference():
    """자기 자신을 관련어로 넣으면 순환 링크가 된다."""
    for term in G.all_terms():
        assert term.term not in term.related, f"{term.term}이 자기 자신을 참조함"


# ── 내용 품질 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", REQUIRED_TERMS)
def test_required_term_exists(name: str):
    """프로젝트 핵심 어휘가 사전에 있어야 한다."""
    assert G.get(name) is not None, f"핵심 용어 '{name}'이 사전에 없습니다"


@pytest.mark.parametrize("name", NEEDS_WHY)
def test_important_terms_explain_why_it_matters(name: str):
    """중요 용어는 '왜 중요한가'를 반드시 설명해야 한다.

    왜: 이 프로젝트는 학습이 목적이다. 정의만 있으면 "그래서 이걸 왜 배우지?"에
        답하지 못한다. 산업적 의미와 연결되어야 기억에 남는다.
    """
    term = G.get(name)
    assert term is not None, f"'{name}' 용어가 없습니다"
    assert term.why_matters.strip(), f"'{name}': why_matters가 비어 있습니다"
    assert len(term.why_matters) > 100, (
        f"'{name}': why_matters가 너무 짧습니다 ({len(term.why_matters)}자)"
    )


def test_every_term_has_short_and_plain():
    """모든 용어는 한 줄 정의와 쉬운 설명을 갖춰야 한다."""
    for term in G.all_terms():
        assert term.short.strip(), f"{term.term}: short가 비어 있음"
        assert term.plain.strip(), f"{term.term}: plain이 비어 있음"
        assert len(term.short) < 120, f"{term.term}: short가 한 줄을 넘음"
        assert len(term.plain) > 40, f"{term.term}: plain이 너무 짧음"


def test_plain_explanation_is_longer_than_short():
    """쉬운 설명이 한 줄 정의보다 짧으면 설명 역할을 못 한다."""
    for term in G.all_terms():
        assert len(term.plain) > len(term.short), f"{term.term}: plain이 short보다 짧음"


def test_all_nine_defect_patterns_are_covered():
    """웨이퍼 맵 불량 패턴 9종이 모두 사전에 있어야 한다."""
    from wafermap.config import PATTERN_LABELS

    for label in PATTERN_LABELS:
        assert G.get(label) is not None, f"불량 패턴 '{label}'이 사전에 없습니다"
        assert G.get(label).category == "불량패턴"


def test_used_in_paths_exist():
    """used_in이 가리키는 파일이 실제로 존재해야 한다."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    missing: list[str] = []
    for term in G.all_terms():
        for path in term.used_in:
            if not (root / path).exists():
                missing.append(f"{term.term} → {path}")
    assert not missing, "존재하지 않는 경로를 참조합니다:\n  " + "\n  ".join(missing)


# ── 검색 동작 ────────────────────────────────────────────────────────────


def test_search_finds_by_alias():
    """영문 표기나 약어로도 찾아져야 한다."""
    assert G.get("Electrical Die Sorting").term == "EDS"
    assert G.get("commonality analysis").term == "커미널리티 분석"
    assert G.get("data leakage").term == "데이터 누출"


def test_search_is_insensitive_to_spacing_and_case():
    """'Edge-Ring', 'edge ring', 'EDGERING'이 모두 같아야 한다."""
    target = G.get("Edge-Ring")
    for variant in ("edge-ring", "edge ring", "EDGERING", "Edge Ring"):
        assert G.get(variant) is target, f"'{variant}'로 찾지 못함"


def test_search_returns_ordered_results():
    """이름 일치가 본문 일치보다 앞에 와야 한다."""
    results = G.search("교락")
    assert results, "검색 결과가 없음"
    assert results[0].term == "교락", f"완전일치가 1위가 아님: {results[0].term}"


def test_search_empty_query_returns_nothing():
    assert G.search("") == ()
    assert G.search("   ") == ()


def test_unknown_term_returns_none():
    assert G.get("존재하지않는용어xyz") is None


def test_related_terms_returns_objects():
    related = G.related_terms("커미널리티 분석")
    assert related
    assert all(isinstance(t, G.Term) for t in related)


def test_find_in_text_detects_terms():
    """본문에서 용어를 찾아내야 한다 (자동 링크용)."""
    text = "이번 분석에서는 커미널리티 분석으로 챔버를 특정한 뒤 SHAP으로 원인을 규명했다."
    found = {t.term for t in G.find_in_text(text)}
    assert "커미널리티 분석" in found
    assert "챔버" in found
    assert "SHAP" in found


def test_find_in_text_ignores_absent_terms():
    found = {t.term for t in G.find_in_text("오늘 점심은 김치찌개였다.")}
    assert "커미널리티 분석" not in found
    assert "Grad-CAM" not in found


def test_by_category_rejects_unknown():
    with pytest.raises(KeyError):
        G.by_category("없는분류")


def test_stats_sums_to_total():
    s = G.stats()
    total = s.pop("합계")
    assert sum(s.values()) == total == len(G.all_terms())


# ── 학습 자료와의 연결 ───────────────────────────────────────────────────


def test_glossary_does_not_import_data_layer():
    """용어사전은 데이터·모델 계층에 의존하면 안 된다 (설계서 §4A.7).

    왜: 모바일에서 용어사전을 열 때 데이터 로딩을 기다리면 안 된다.
        의존성이 생기면 수 초가 걸리고 '틈틈이 공부'가 불가능해진다.
    """
    import inspect

    source = inspect.getsource(G)
    for forbidden in ("from wafermap.data", "from wafermap.models", "import pandas", "import numpy"):
        assert forbidden not in source, (
            f"용어사전이 무거운 의존성을 가짐: {forbidden}"
        )
