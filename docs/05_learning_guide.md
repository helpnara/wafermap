# 학습 가이드 — 이 프로젝트로 공부하는 법

> 이 프로젝트는 **동작하는 앱**이자 **학습 교재**다. 이 문서는 무엇을 어떤 순서로
> 읽어야 하는지 안내한다.

---

## 세 가지 학습 자료

| 자료 | 무엇을 배우나 | 어디에 |
|---|---|---|
| **용어사전** | 반도체·통계·ML 용어 98개의 뜻과 **왜 중요한지** | `📖 용어사전` 메뉴 / `scripts/glossary.py` |
| **해석 가이드** | 분석 결과를 **읽고 판단하는 법** | `docs/08_interpretation_guide.md` |
| **학습노트** | 코드가 **왜 그렇게 짜였는지** | `🎓 도움말·학습` 메뉴 / `docs/learning_notes/` |

세 자료의 역할이 다르다.

```
용어사전   "커미널리티 분석이 뭐지?"          → 개념
해석 가이드 "OR 45.9가 나왔는데 뭘 해야 하지?" → 판단
학습노트   "이 코드는 왜 이렇게 짰지?"         → 구현
```

---

## 추천 학습 순서

### 1단계 — 도메인 감 잡기 (30분)

반도체를 전혀 모른다면 여기서 시작한다.

```bash
python scripts/glossary.py --category 제품     # 웨이퍼, die, lot, 수율
python scripts/glossary.py --category 테스트   # EDS, wafer map, Bin
python scripts/glossary.py 수율                # why_matters를 꼭 읽을 것
```

**`수율`과 `wafer map` 두 항목의 "왜 중요한가"만 읽어도** 이 프로젝트가
무엇을 하려는지 이해할 수 있다.

### 2단계 — 전체 그림 (20분)

`docs/08_interpretation_guide.md` 의 **"전체 흐름 — 왜 이 순서인가"** 절만 읽는다.

```
웨이퍼 맵 → 패턴 분류 → SPC → 커미널리티 → SHAP → 개선안
```

각 단계가 다음 단계의 범위를 좁힌다는 것만 이해하면 된다.

### 3단계 — 데이터가 어떻게 만들어졌나 (M1)

| 순서 | 자료 |
|---|---|
| 1 | `learning_notes/synth_wafer.md` — 웨이퍼 맵 생성 |
| 2 | `learning_notes/fdc_simulator.md` — 공정 데이터 생성 |

**가장 중요한 개념**: 왜 불량을 웨이퍼별로 무작위 배정하면 안 되는가
(→ 분석할 것이 없어진다)

### 4단계 — 모델링 (M2)

| 순서 | 자료 |
|---|---|
| 1 | `learning_notes/geometry.md` — 이미지를 숫자로 바꾸기 |
| 2 | `learning_notes/pattern_lgbm.md` — 1,000:1 불균형과 싸우기 |
| 3 | `learning_notes/pattern_cnn.md` — 딥러닝은 왜 졌나 |

**가장 중요한 개념**: 정확도 대신 macro-F1을 쓰는 이유, 데이터 누출, 지름길 학습

### 5단계 — 결과 읽기

`docs/04_results.md` 를 `docs/08_interpretation_guide.md` 와 나란히 놓고 본다.
숫자를 보면서 "이걸 어떻게 읽는 거였지?"를 바로 찾아볼 수 있다.

---

## 모바일에서 공부하기

`📖 용어사전`과 `🎓 도움말·학습` 은 **데이터를 전혀 로드하지 않도록** 만들어져 있다.
그래서 휴대폰에서도 3초 안에 뜬다.

- 학습노트는 3~5분 단위 카드로 나뉘어 있어 이동 중에 한 카드씩 볼 수 있다
- 진도는 URL에 기록되므로 홈 화면에 바로가기로 저장하면 이어서 볼 수 있다
- 용어사전은 검색이 한글·영문·약어를 모두 지원한다 (`엣지링`, `edge ring`, `EDS`)

---

## 스스로 확인하기

각 학습노트 끝에 **확인 문제**가 있다. 접힌 정답을 보기 전에 스스로 답해 본다.

특히 아래 질문들에 답할 수 있으면 핵심을 이해한 것이다.

1. 왜 정확도가 아니라 macro-F1을 쓰는가?
2. 데이터 증강을 fold 분할 전에 하면 왜 안 되는가?
3. 챔버 37개를 동시에 검정하면 무엇이 문제인가?
4. SHAP 값이 크다는 것이 "그것이 원인이다"를 뜻하는가?
5. 이 프로젝트의 macro-F1 0.974를 포트폴리오에 어떻게 써야 하는가?

---

## 코드를 직접 만져 보기

```bash
# 데이터 다시 만들기 (난이도를 바꿔 보면 결과가 달라진다)
python scripts/build_dataset.py --n-wafers 3000

# 피처 다시 뽑기
python scripts/build_features.py --n-jobs -1

# 모델 다시 학습 (증강을 켜 보면?)
python scripts/train_pattern_model.py --augment-target 200

# 테스트로 내가 뭘 깨뜨렸는지 확인
python -m pytest tests/ -q
```

**테스트가 안전망이다.** 코드를 고치다 뭔가 깨지면 테스트가 알려 준다.
특히 아래 테스트들은 "이건 절대 하면 안 된다"를 코드로 못박은 것이다.

| 테스트 | 무엇을 막나 |
|---|---|
| `test_no_absolute_size_features` | 지름길 학습 유발 피처 추가 |
| `test_cv_does_not_leak_augmented_maps_into_validation` | 데이터 누출 |
| `test_quoted_code_matches_source` | 학습노트와 코드의 불일치 |
| `test_related_links_resolve` | 용어사전 깨진 링크 |
