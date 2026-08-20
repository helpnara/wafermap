# EDS Wafer Map 기반 불량 패턴 · 공정 원인 분석 시스템

EDS(Electrical Die Sorting) 검사 결과를 웨이퍼 맵으로 공간화하고,
그 웨이퍼가 거쳐 간 **공정·설비·센서 데이터와 연결**해 불량의 원인 후보를 찾고
개선안과 기대효과까지 도출하는 분석 시스템입니다.

> 제품 시나리오는 **DRAM(1z-nm)** 기준입니다.
> 데이터는 공개 데이터셋(WM-811K)과 물리 기전 기반 합성 FDC를 함께 씁니다.

---

## 이 프로젝트가 답하는 질문

```
불량 발생 → 어떤 패턴인가 → 언제 시작됐나 → 어느 설비인가
         → 그 설비의 무엇이 문제인가 → 무엇을 바꿀 것인가 → 얼마를 버는가
```

| 단계 | 질문 | 방법 | 측정된 성능 |
|---|---|---|---|
| M2 | 어떤 패턴인가 | 기하 피처 + LightGBM | macro-F1 **0.973** |
| M3 | 언제 · 어느 설비인가 | p-chart SPC + 커미널리티 | 원인 챔버 Top-3 **90% ± 12%** |
| M4 | 그 설비의 무엇이 | LightGBM + SHAP | 원인 파라미터 Top-3 **100% ± 0%** |
| M5 | 무엇을 바꿀 것인가 | 권고 구간 + 반사실 시뮬레이션 | 실행 가능성 제약 반영 |
| M5.5 | 공정이 맞나 · 무엇이 튀었나 | 검사 설비 축 + 시계열 피처 | 검사 설비 Top-1 2/2 |

성능 수치는 시뮬레이터가 심어 둔 **정답지와 대조한 채점** 결과이며,
**여러 시드의 평균 ± 표준편차**입니다. 합성 데이터의 채점은 시드 하나에 크게
흔들리므로 단일 실행 수치는 인용하지 않습니다 (`scripts/run_seed_study.py`).

근거와 한계는 [`docs/04_results.md`](docs/04_results.md)에 전부 기록돼 있습니다.

---

## 빠른 시작

```bash
git clone https://github.com/helpnara/wafermap.git
cd wafermap

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 데이터 → 피처 → 모델 → 개선안 (합계 약 2분)
python scripts/build_dataset.py
python scripts/build_features.py
python scripts/train_pattern_model.py
python scripts/build_recommendations.py
python scripts/build_rootcause.py

streamlit run app.py
```

> 상세 설치·VS Code 연동·문제 해결은 **[`docs/07_local_setup.md`](docs/07_local_setup.md)** 를 보세요.

---

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/00_design.md`](docs/00_design.md) | 설계서 — 데이터 전략, 화면 설계, 반응형/접근성 |
| [`docs/01_domain_eds.md`](docs/01_domain_eds.md) | **EDS 도메인 입문** — 다른 산업에서 온 사람을 위한 배경 |
| [`docs/02_data_dictionary.md`](docs/02_data_dictionary.md) | 데이터 사전 (코드에서 생성) |
| [`docs/03_simulator_spec.md`](docs/03_simulator_spec.md) | 시뮬레이터 명세 — 무엇을 어떻게 심는가 |
| [`docs/04_results.md`](docs/04_results.md) | 분석 결과와 **발견한 문제들** (포트폴리오 본문) |
| [`docs/05_learning_guide.md`](docs/05_learning_guide.md) | 코드 학습 로드맵 |
| [`docs/06_local_validation.md`](docs/06_local_validation.md) | 실측(WM-811K) 전환 절차와 함정 |
| [`docs/07_local_setup.md`](docs/07_local_setup.md) | 로컬 환경 구축 · VS Code · GitHub |
| [`docs/08_interpretation_guide.md`](docs/08_interpretation_guide.md) | 분석 결과 해석 가이드 |
| [`docs/09_roadmap.md`](docs/09_roadmap.md) | **남은 작업 목록** |
| [`docs/learning_notes/`](docs/learning_notes/) | 모듈별 코드 해설 (8건) |

---

## 이 프로젝트의 원칙

**AI가 원인을 확정하지 않습니다.** 원인 후보와 근거를 제시하고, 공정 전문가가 최종 검증합니다.
모든 분석은 상관 기반이며, 확증은 DOE(실험계획)로 해야 합니다.
이 시스템의 역할은 **DOE 대상을 좁히는 것**입니다.

그래서 코드 곳곳에서 다음을 강제합니다.

- 모델 AUC가 낮으면 SHAP 해석을 신뢰하지 말라고 **화면에 띄운다**
- SHAP 순위 옆에 **모델과 무관한 분포 증거**(Cliff's δ, KS 검정)를 나란히 놓는다
- 계측값(파티클 수, 두께 산포)은 조작 손잡이가 아니므로 **조치 대상에서 뺀다**
- 기대효과는 챔버 기준과 **라인 기준을 병기**한다
- ROI 가정치는 전부 **화면에서 바꿀 수 있고**, 민감도를 함께 보여준다
