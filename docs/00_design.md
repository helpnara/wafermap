# EDS Wafer Map 기반 이상 공정 탐지 및 원인 분석 시스템 — 설계서

> **문서 버전** v1.0 (2026-08-17) · **상태** 설계 확정 대기
> **목적** 반도체 EDS(Electrical Die Sorting) 테스트 결과인 wafer map으로부터 이상 공정을 특정하고,
> 해당 공정의 설비/FDC 데이터와 연결하여 불량 원인을 규명한 뒤 개선방안과 기대효과를 도출하는
> Streamlit 기반 분석 애플리케이션.

---

## 1. 배경과 문제 정의

### 1.1 EDS 테스트란

EDS(Electrical Die Sorting)는 FAB 공정이 완료된 웨이퍼를 조립(Assembly) 전에 웨이퍼 상태로
전기적 검사하는 단계다. 국내 메모리 업계 용어이며, 산업 일반 용어로는 Wafer Test / Probe Test /
CP(Chip Probe)에 해당한다.

전형적인 EDS 흐름:

| 단계 | 내용 | 산출물 |
|---|---|---|
| EDS #1 | DC Parametric — open/short, leakage 등 기본 전기특성 | Bin code |
| EDS #2 | Function Test — cell fail 검출 | Fail bit map |
| Repair | Laser/e-Fuse repair — redundancy cell로 대체 | Repair log |
| EDS #3 | Post-repair verify | Bin code |
| EDS #4 | Burn-in / Stress / Speed grading | 최종 Bin sort |

각 die는 최종적으로 Bin code를 부여받는다(Bin1 = Good, Bin2~N = fail mode별 코드).
**die별 pass/fail을 웨이퍼 좌표에 배열한 것이 wafer map(bin map)** 이며, 본 프로젝트의 출발점이다.

### 1.2 현업의 문제

Wafer map의 **공간적 패턴은 원인 공정의 지문(fingerprint)** 이다. 웨이퍼는 원판이고 대부분의
단위 공정은 회전 대칭(spin coating, CMP 연마, 플라즈마 식각)이거나 반경 방향 프로파일을 갖기
때문에, 불량이 중심/링/엣지/선형 등 **물리적으로 해석 가능한 형상**으로 나타난다.

그러나 현업에서는:
- 패턴 판정이 엔지니어 육안에 의존 → 판정자 간 편차, 대량 물량 시 누락
- 패턴을 봐도 **어느 스텝의 어느 설비**인지 특정하는 데 시간 소요 (수백 스텝 × 수십 설비)
- 원인 파라미터 규명이 경험칙에 의존, 정량적 개선 목표치 설정이 어려움

### 1.3 본 프로젝트가 푸는 것

```
[1] Wafer Map 구성      → EDS bin map 재구성 및 시각화
[2] 패턴 자동 분류      → 9종 불량 패턴 ML 분류 (이상 유형 판정)
[3] 이상 발생 시점 탐지 → SPC 관리도 기반 이상 구간 검출 (언제부터?)
[4] 이상 공정/설비 특정 → 커미널리티 분석 (어느 스텝, 어느 챔버?)
[5] 원인 인자 규명      → FDC 파라미터 수준 SHAP 기여도 (무엇이 문제?)
[6] 개선방안 + 기대효과 → 최적 운전구간 도출, 수율 개선분·ROI 추정 (어떻게 조치?)
```

---

## 2. 데이터 전략

### 2.1 핵심 제약

**공개 데이터 중 wafer map과 공정 설비 센서(FDC)가 실제로 연결된 데이터셋은 존재하지 않는다.**

| 데이터셋 | 내용 | 한계 |
|---|---|---|
| WM-811K (MIR Lab) | 실제 fab의 wafer map 811,457장, 그중 172,950장 패턴 라벨(9종) | 공정/설비 정보 없음 |
| UCI SECOM | 반도체 공정 센서 590개 × 1,567 lot, pass/fail | 센서 의미 익명화, wafer map 없음 |
| UCI SECOM 계열 | — | 두 데이터 간 조인 키 없음 |

### 2.2 채택 전략 — 실측 맵 + 물리 기반 합성 FDC

```
┌────────────────────────┐        ┌──────────────────────────────┐
│ WM-811K (실측)          │        │ FDC Simulator (합성)          │
│ · die-level bin map     │◀──조인──▶│ · 8개 공정 스텝, 설비/챔버    │
│ · 9종 패턴 라벨          │ lot/    │ · 파라미터 시계열·요약통계     │
│ · 811k wafer            │ wafer   │ · 패턴별 인과 섭동 주입        │
└────────────────────────┘  key    │ · ground truth 동시 생성       │
                                    └──────────────────────────────┘
```

**설계 원칙**

1. **맵은 실측, 공정은 합성** — 앱 전면에 이 사실을 명시(사이드바 배지 + 데이터 개요 페이지).
   포트폴리오에서 데이터 출처를 호도하지 않는 것이 신뢰성의 핵심.
2. **합성은 임의가 아니라 물리 기반** — 각 불량 패턴에 대해 반도체 공정 물리로 설명되는
   원인 스텝·파라미터를 문헌/현업 지식에 근거해 매핑하고(§2.4), 그 파라미터에만 섭동을 주입.
3. **난이도를 인위적으로 유지** — 교락 요인, 정상군 드리프트(위양성), 설명 불가 불량(위음성),
   챔버 간 baseline 편차, 시간에 따른 설비 노후화를 함께 주입해 분석이 자명해지지 않게 한다.
4. **Ground truth를 별도 보관** — `ground_truth.parquet`에 실제 주입한 원인 스텝/설비/파라미터를
   기록. 이를 **원인 규명 모델의 정답지**로 사용해 "원인 분석 정확도"라는 정량 지표를 제시한다.
   → 대부분의 유사 포트폴리오가 못 하는 차별점.

### 2.3 개발 환경 제약과 대응

현 개발 컨테이너는 네트워크 정책상 pypi/npm만 허용되고 Kaggle/UCI/HuggingFace는 차단되어
**WM-811K 원본을 컨테이너에서 내려받을 수 없다.** 대응:

| 대응 | 내용 |
|---|---|
| `scripts/download_wm811k.py` | 사용자 로컬에서 Kaggle API로 `LSWMD.pkl` 다운로드 (~2GB) |
| `src/wafermap/data/synth_wafer.py` | WM-811K와 **동일 스키마**의 합성 wafer map 생성기. 개발·CI·클라우드 데모용 |
| 데이터 소스 토글 | 앱 사이드바에서 `실측(WM-811K)` / `데모(합성)` 전환. 코드 경로는 동일 |

이 구조 덕분에 원본 없이 전체 파이프라인을 개발·테스트하고, 사용자가 로컬에서 `LSWMD.pkl`을
`data/raw/`에 두는 순간 그대로 실데이터로 동작한다.

### 2.4 불량 패턴 ↔ 원인 공정 매핑 (도메인 코어)

시뮬레이터의 인과 규칙이자, 앱이 제시하는 진단 근거의 기반.

| 패턴 | 형상 | 물리적 기전 | 원인 스텝 | 핵심 FDC 파라미터 |
|---|---|---|---|---|
| **Center** | 중심부 집중 | 반경방향 제거율/증착율 프로파일이 중심에서 이탈 | CMP (P050) | `down_force`↑, `pad_life`↑, `center_zone_pressure`↑ |
| **Donut** | 중심 제외 링 | 중간 반경대 온도/플라즈마 밀도 불균일 | CVD (P030) | `susceptor_temp_mid_delta`, `precursor_flow` 변동 |
| **Edge-Ring** | 최외곽 링 | 엣지 배제영역 처리 이상, 챔버 edge ring 마모로 엣지 식각률 저하 | Etch (P020) | `edge_ring_rf_hours`↑, `chamber_pressure` drift, `edge_gas_ratio` |
| **Edge-Loc** | 가장자리 국부 | 웨이퍼 핸들링 접촉, 척 엣지 파티클/온도 국부 이상 | Cleaning (P060) / 핸들링 | `chuck_edge_temp_dev`, `particle_count`↑ |
| **Loc** | 국부 클러스터 | 파티클 낙하, 국부 디포커스 | Photo (P010) | `focus_offset` 이탈, `defect_adder_count`↑ |
| **Scratch** | 선형 긁힘 | CMP 슬러리 이물/패드 스크래치, 핸들링 긁힘 | CMP (P050) | `slurry_flow`↓, `conditioning_time`↓, `particle_count`↑ |
| **Random** | 산발 | 랜덤 파티클, 미세 오염 | 다중/미상 | 특정 인자 없음 (노이즈 baseline 상승) |
| **Near-full** | 대부분 fail | 설비 major fault, recipe 오적용, mask 오류 | Implant (P040) 등 | `dose` 30%+ 이탈, `recipe_id` 오적용 |
| **none** | 정상 | — | — | — |

> 이 매핑은 시뮬레이터가 "심는" 정답이자, 커미널리티/SHAP 분석이 **독립적으로 다시 찾아내야 하는**
> 대상이다. 앱의 규칙 기반 힌트에는 이 표를 노출하되, 모델 결과는 데이터에서만 도출한다.

### 2.5 가상 공정 플로우 (합성 FDC)

| Step | 공정 | 설비 (챔버) | 주요 파라미터 |
|---|---|---|---|
| P010 | Photo / Litho | SCAN-01~03 | focus_offset, exposure_dose, overlay_x/y, bake_temp, ebr_width |
| P020 | Etch | ETCH-A/B/C (ch1~4) | rf_power, chamber_pressure, cf4_flow, o2_flow, electrode_temp, endpoint_time, edge_ring_rf_hours |
| P030 | Thin Film / CVD | CVD-01/02 | dep_temp, precursor_flow, chamber_pressure, thickness_mean/sigma, susceptor_temp_mid_delta |
| P040 | Implant | IMP-01/02 | dose, energy, beam_current, tilt_angle |
| P050 | CMP | CMP-01/02 (head1~4) | down_force, platen_speed, slurry_flow, pad_life, conditioning_time, removal_rate, center_zone_pressure |
| P060 | Cleaning | CLN-01/02 | chem_conc, bath_temp, di_resistivity, particle_count, chuck_edge_temp_dev |
| P070 | Diffusion / Anneal | DIFF-01 | furnace_temp_z1~z3, ramp_rate, o2_flow |
| P080 | Inline Metrology | MET-01 | cd_mean/sigma, thickness, overlay_residual |

- Lot = 25 wafer, wafer는 slot 1~25. Lot 단위로 스텝별 설비/챔버가 배정된다(현업과 동일).
- Slot 위치 효과(엣지 슬롯 열 이력 차이)를 약하게 부여 → 슬롯 상관 분석 소재.
- 설비별 baseline 편차 + 시간 드리프트(PM 주기 톱니파) 주입 → 커미널리티/트렌드 분석 소재.

### 2.6 데이터 스키마

```
data/processed/
├─ wafer_master.parquet      # 웨이퍼 1행
│    wafer_id, lot_id, slot_no, product, tech_node,
│    fab_in_time, eds_time, die_total, die_pass, yield_pct,
│    pattern_label, map_path, data_source(real|synthetic)
├─ die_map.npz               # wafer_id → 2D int8 array (0=no die, 1=pass, 2=fail)
├─ fdc_summary.parquet       # (wafer_id, step_id) × 파라미터 wide
│    wafer_id, step_id, step_name, equip_id, chamber_id, recipe_id,
│    <param>_mean, <param>_std, <param>_min, <param>_max, run_time
├─ fdc_trace.parquet         # 주요 파라미터만 시계열 (long)
│    wafer_id, step_id, param, t_sec, value
└─ ground_truth.parquet      # 합성 정답지 (검증 전용, 분석 모델 입력 금지)
     wafer_id, true_root_step, true_root_equip, true_root_params(list),
     severity, is_confounded, is_unexplained
```

---

## 3. 분석·모델 설계

### M1. Wafer Map 패턴 분류

**입력** die-level bin map (가변 크기) → 64×64 정규화 그리드 (die-aware 리샘플링, 웨이퍼 외곽 마스크 보존)

**피처 (LightGBM용, 약 90차원)**

| 그룹 | 피처 |
|---|---|
| 기본 | fail_ratio, die_total, 웨이퍼 반경, 결측 die 비율 |
| 반경 프로파일 | 10개 동심 링별 fail rate, 링간 기울기, 최외곽/내부 비 (Edge-Ring 판별 핵심) |
| 각도 프로파일 | 12개 섹터별 fail rate, 각도 분산, 최대 섹터 비 |
| 밀도 | 13-zone density (Wu et al. 2015 표준 피처) |
| 기하 모멘트 | Hu moment 7종, 중심 편심도, 관성 반경 |
| 연결성분 | cluster 수, 최대 cluster 면적/이심률/solidity/장축비 (Loc·Scratch 판별) |
| Radon | Radon 변환 행 평균의 cubic interp 20점 + 표준편차 20점 (Scratch·Donut 판별, 원논문 방식) |

**모델**

- 주 모델: **LightGBM multiclass** (9 class). 극심한 불균형(none 147k vs Near-full 149) 대응 →
  class_weight balanced + macro-F1 최적화 + stratified 5-fold CV
- 비교군: **경량 CNN** (4 conv block, 64×64×1 입력, ~200K params, PyTorch CPU 학습 가능 규모) + Grad-CAM
- 평가: macro-F1(주지표), per-class recall, confusion matrix, PR curve, 추론 지연시간
- 해석: LGBM SHAP(어떤 기하 피처가 판정을 만들었나) vs CNN Grad-CAM(어디를 보았나) 병렬 제시

> **불균형 처리 방침**: 오버샘플링은 리크 위험이 있어 fold 내부에서만 적용. Near-full처럼 149장뿐인
> 클래스는 augmentation(회전·미러 — 웨이퍼는 회전 대칭이므로 물리적으로 타당)으로 보강.

### M2. 이상 발생 시점 탐지 (SPC)

- 시간축: `eds_time` 기준 일/시프트 단위 집계
- 관제 지표: 전체 수율, 패턴별 발생률, 설비별 수율
- 기법: **EWMA 관리도**(λ=0.2) + **CUSUM**(작은 시프트 민감) 병행, 관리한계 3σ
- 산출: 이상 구간(alarm 시작~해제) 자동 검출 → 이 구간의 wafer 집합을 **이상군(case)**,
  직전 안정 구간을 **정상군(control)** 으로 정의 → M3의 입력

### M3. 이상 공정·설비 특정 (커미널리티 분석)

현업 불량 분석의 표준 절차를 그대로 구현.

1. 이상군 vs 정상군에 대해 **스텝 × 설비/챔버** 교차표 생성
2. Fisher exact test (기대빈도 5 미만 셀 대응) 또는 chi-square
3. 효과크기: **odds ratio** + 95% CI, `lift = 이상군 통과율 / 정상군 통과율`
4. 다중검정 보정: **Benjamini-Hochberg FDR** (수백 개 설비를 동시 검정하므로 필수)
5. 랭킹 출력 예시:

   > `ETCH-B / ch3` — 이상군 통과 68/80(85.0%) vs 정상군 41/400(10.3%) · OR 45.9 (CI 23.1–91.2) · p_adj = 3.1e-27

6. 교락 통제: 상위 후보에 대해 다른 스텝 설비를 층화한 Cochran–Mantel–Haenszel 검정으로
   "정말 이 설비인가, 같이 흘러간 다른 설비 때문인가" 구분

### M4. 원인 인자 규명 (FDC 파라미터 수준)

- **타깃**: (a) 해당 패턴 발생 여부(binary), (b) wafer yield(regression) — 두 관점 병행
- **입력**: M3에서 특정된 스텝의 FDC 요약 통계 + 설비/챔버/recipe 범주형
- **모델**: LightGBM + **SHAP**
  - Global: mean|SHAP| 기여도 랭킹
  - Local: 개별 wafer의 force plot → "이 웨이퍼는 edge_ring_rf_hours=412h가 불량확률을 +0.38 올렸다"
  - Dependence: 파라미터 값 ↔ 불량확률 관계 곡선 (임계점 시각화)
- **통계 보조**: 이상/정상군 파라미터 분포 비교 (KS test, PSI, Cliff's delta)
- **인과 주의**: 상관 ≠ 인과 경고를 UI에 명시. 설비를 고정한 **층화 분석**으로 교락 최소화.
  (DoWhy 등 본격 인과추론은 범위 외 — 데이터 생성 구조상 과잉)
- **검증**: `ground_truth.parquet`와 대조하여
  - Root step 적중률 (Top-1 / Top-3)
  - Root parameter 적중률 (Top-5 SHAP 내 정답 파라미터 포함률)
  → **"원인 분석 정확도 87%" 같은 정량 성과**를 포트폴리오 지표로 제시

### M5. 개선방안 및 기대효과

1. **최적 운전 구간 도출**: SHAP dependence / partial dependence에서 불량확률이 최소인 구간을
   탐색 → 파라미터별 권고 spec 제안 (예: `CMP down_force 3.20–3.55 psi`, 현 spec 3.0–4.0 대비 tightening)
2. **조치안 템플릿** (패턴별 자동 생성 + 편집 가능)
   - 설비 PM 주기 단축 (예: edge ring 교체 400h → 300h)
   - 파라미터 spec tightening 및 SPC 인터락 설정
   - 챔버 매칭(chamber-to-chamber) 튜닝
   - 계측 샘플링 강화 구간 지정
3. **기대효과 시뮬레이션**: 제안 spec 내로 파라미터를 clip한 반사실(counterfactual) 입력을
   학습 모델에 통과시켜 예측 수율 변화 산출. **가정(모델 외삽 한계, 인과 가정)을 명시적으로 표기.**
4. **ROI 추정**: wafer 1장당 가치·월 투입량을 사용자가 조정 가능한 입력으로 두고 연간 절감액 환산

### M6. 신규 웨이퍼 실시간 스코어링

맵 업로드(또는 데이터셋 내 선택) → 패턴 예측 → 원인 후보 Top-3 스텝/설비 + 조치 가이드 즉시 출력.
현업 적용 시나리오를 보여주는 데모 페이지.

---

## 4. Streamlit 애플리케이션 설계

### 4.1 페이지 구성

| 페이지 | 목적 | 핵심 컴포넌트 |
|---|---|---|
| `app.py` — 개요 | 프로젝트 요약, KPI 카드, 데이터 출처 명시 | 수율 추이, 패턴 분포, 모델 성능 요약 |
| `1_데이터_개요` | 데이터 구조·품질 | lot/wafer/die 계층, 결측·이상치, 스키마 사전 |
| `2_웨이퍼맵_탐색` | 맵 시각화 | 개별 맵 뷰어, 패턴별 갤러리, 반경/섹터 오버레이, 로트 단위 25장 뷰 |
| `3_패턴_분류_모델` | 모델링 결과 | 피처 설명, LGBM vs CNN 비교, 혼동행렬, SHAP/Grad-CAM |
| `4_이상공정_탐지` | 이상 구간 검출 | EWMA/CUSUM 관리도, alarm 목록, 이상군 정의 |
| `5_원인_분석` | 핵심 페이지 | 커미널리티 랭킹 → 설비 선택 → FDC 분포 비교 → SHAP → trace 뷰어 |
| `6_개선방안_기대효과` | 결론 | 권고 spec, 조치안 카드, 반사실 수율 시뮬레이터, ROI 계산기 |
| `7_모델_검증` | 신뢰성 | ground truth 대비 원인규명 정확도, 실험 로그, 한계 명시 |

### 4.2 UX 원칙

- **언어**: 한국어 UI. 공정 용어는 현업 표기 유지(EDS, FDC, Bin, Lot, CMP …)
- **서사 흐름**: 페이지 번호 순서가 곧 분석 스토리. 각 페이지 하단에 "다음 단계" 안내 버튼
- **드릴다운 일관성**: 사이드바 전역 필터(기간·제품·스텝)를 `st.session_state`로 페이지 간 공유
- **데이터 출처 배지**: 모든 페이지 사이드바에 `맵: 실측 WM-811K / 공정: 합성` 배지 상시 노출
- **성능**: `@st.cache_data`(데이터·피처), `@st.cache_resource`(모델). 무거운 학습은 사전 수행 후
  아티팩트 로드만 수행 — 앱 내 학습은 금지(클라우드 메모리 한계)

### 4.3 배포

| 항목 | 로컬 | Streamlit Community Cloud |
|---|---|---|
| 데이터 | WM-811K 전체(172,950 라벨) | 샘플 (~15,000 wafer, parquet 압축, <100MB) |
| 모델 | 전체 학습 가능 | 사전학습 아티팩트만 로드 (LGBM <10MB, CNN <5MB) |
| CNN | PyTorch CPU 학습 | `torch` CPU wheel, 추론만 |
| 메모리 | 제한 없음 | ~1GB 내 동작 검증 필수 |

---

## 5. 리포지토리 구조

```
wafermap/
├─ README.md                         # 프로젝트 소개, 결과 요약, 실행법
├─ requirements.txt / requirements-dev.txt
├─ .streamlit/config.toml
├─ .gitignore                        # data/raw, data/interim 제외
├─ app.py
├─ pages/
│   ├─ 1_데이터_개요.py
│   ├─ 2_웨이퍼맵_탐색.py
│   ├─ 3_패턴_분류_모델.py
│   ├─ 4_이상공정_탐지.py
│   ├─ 5_원인_분석.py
│   ├─ 6_개선방안_기대효과.py
│   └─ 7_모델_검증.py
├─ src/wafermap/
│   ├─ config.py                     # 경로·상수·공정 플로우 정의
│   ├─ data/
│   │   ├─ wm811k.py                 # 원본 pkl 로더/정규화
│   │   ├─ synth_wafer.py            # 합성 wafer map 생성기 (개발/데모용)
│   │   ├─ fdc_simulator.py          # 공정·설비·FDC 생성 + 인과 섭동 주입
│   │   ├─ schema.py                 # 스키마 정의·검증
│   │   └─ loader.py                 # 캐시 로더 (real/synthetic 토글)
│   ├─ features/
│   │   ├─ geometry.py               # 반경/각도/모멘트/13-zone
│   │   ├─ radon_feat.py
│   │   ├─ connectivity.py
│   │   └─ build.py                  # 피처 파이프라인
│   ├─ models/
│   │   ├─ pattern_lgbm.py
│   │   ├─ pattern_cnn.py
│   │   ├─ cause_model.py
│   │   └─ registry.py               # 아티팩트 저장/로드
│   ├─ analysis/
│   │   ├─ spc.py                    # EWMA / CUSUM
│   │   ├─ commonality.py            # Fisher / OR / BH-FDR / CMH
│   │   ├─ attribution.py            # SHAP 래퍼, 분포 비교 검정
│   │   ├─ recommend.py              # 최적 구간·조치안·반사실 시뮬레이션
│   │   └─ validate.py               # ground truth 대비 원인규명 정확도
│   └─ viz/
│       ├─ wafer_plot.py             # 맵 렌더링(plotly/matplotlib)
│       └─ charts.py
├─ scripts/
│   ├─ download_wm811k.py
│   ├─ build_dataset.py              # 원본/합성 → processed 파케이 일괄 생성
│   ├─ train_pattern_model.py
│   ├─ train_cause_model.py
│   └─ make_cloud_sample.py          # 클라우드 배포용 샘플 추출
├─ data/{raw,interim,processed,sample}/
├─ models/                           # 사전학습 아티팩트 (소형만 커밋)
├─ docs/
│   ├─ 00_design.md                  # 본 문서
│   ├─ 01_domain_eds.md              # EDS·패턴-공정 매핑 도메인 정리
│   ├─ 02_data_dictionary.md
│   ├─ 03_simulator_spec.md          # 합성 데이터 생성 가정 전체 공개
│   └─ 04_results.md                 # 분석 결론·개선안 (포트폴리오 본문)
└─ tests/                            # 피처·시뮬레이터·통계 함수 단위 테스트
```

---

## 6. 기술 스택

| 영역 | 선택 | 사유 |
|---|---|---|
| UI | Streamlit ≥1.36 (multipage) | 요구사항 |
| 데이터 | pandas, numpy, pyarrow(parquet) | 표준 |
| 이미지/기하 | scikit-image (Radon, label, moments), scipy | 피처 추출 |
| ML | scikit-learn, **LightGBM** | 표형 데이터 성능·속도·해석성 |
| DL | **PyTorch (CPU)** | 경량 CNN + Grad-CAM |
| 해석 | **SHAP** | 원인 인자 기여도 |
| 통계 | scipy.stats, statsmodels (BH-FDR, CMH) | 커미널리티 검정 |
| 시각화 | plotly (인터랙티브), matplotlib (정적) | 맵 hover·드릴다운 |
| 품질 | pytest, ruff | 테스트·린트 |

---

## 7. 개발 마일스톤

| # | 산출물 | 완료 기준 |
|---|---|---|
| **M0** | 설계 문서 | 본 문서 확정 (현 단계) |
| **M1** | 데이터 계층 | 합성 wafer 생성기 + FDC 시뮬레이터 + WM-811K 로더, `build_dataset.py` 실행 시 processed 일괄 생성, 스키마 테스트 통과 |
| **M2** | 피처·패턴 모델 | 90차원 피처 파이프라인, LGBM macro-F1 리포트, CNN 비교, 아티팩트 저장 |
| **M3** | 이상탐지·커미널리티 | EWMA/CUSUM alarm 검출, Fisher+BH-FDR 랭킹, CMH 층화 |
| **M4** | 원인 모델·SHAP | cause model 학습, SHAP 산출, ground truth 대비 적중률 리포트 |
| **M5** | 개선안·기대효과 | 권고 spec 도출, 반사실 수율 시뮬레이션, ROI 계산기 |
| **M6** | Streamlit 통합 | 8개 페이지 동작, 전역 필터, 캐시 최적화 |
| **M7** | 검증·문서·배포 | 테스트 통과, README·결과 문서, 클라우드 샘플·메모리 검증 |

---

## 8. 성공 기준 (포트폴리오 관점)

| 지표 | 목표 |
|---|---|
| 패턴 분류 macro-F1 | ≥ 0.80 (WM-811K 라벨 기준, 문헌 수준 대비 경쟁력) |
| Near-full 등 희소 클래스 recall | ≥ 0.70 |
| 원인 스텝 Top-1 적중률 | ≥ 0.80 (ground truth 대비) |
| 원인 파라미터 Top-5 포함률 | ≥ 0.85 |
| 앱 응답 | 페이지 전환 < 2초 (캐시 워밍 후) |
| 서사 완결성 | 데이터 → 탐지 → 원인 → **개선안·기대효과**까지 단절 없이 연결 |

---

## 9. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| WM-811K 라벨 노이즈(오라벨 알려짐) | 성능 상한 제약 | 문서에 명시, 혼동행렬로 오라벨 사례 제시 (한계 인식도 역량) |
| 극심한 클래스 불균형 | 희소 클래스 붕괴 | class weight + 회전/미러 augmentation + macro-F1 주지표 |
| **합성 FDC의 신뢰성 의심** | 포트폴리오 설득력 저하 | ① 출처 상시 명시 ② `03_simulator_spec.md`에 전 가정 공개 ③ ground truth 검증 지표 제시 ④ "방법론 검증용 디지털 트윈"으로 명확히 포지셔닝 |
| 시뮬레이터가 너무 쉬워 분석이 자명 | 분석 역량 미증명 | 교락·노이즈·위양성/위음성·설비 baseline 편차 의도적 주입, 난이도 파라미터화 |
| Streamlit Cloud 1GB 메모리 | 배포 실패 | 샘플 데이터셋 + 사전학습 아티팩트, 앱 내 학습 금지, 메모리 프로파일링 |
| 개발 컨테이너 네트워크 차단 | 실데이터 검증 불가 | 합성 경로로 전체 개발, 사용자 로컬에서 실데이터 검증 단계 별도 |
| 상관을 인과로 오독 | 잘못된 개선안 | 층화 분석, 인과 가정 UI 명시, 기대효과에 신뢰구간·가정 병기 |

---

## 10. 남은 확인 사항

1. **WM-811K 실데이터 적용 시점** — 합성 경로로 전체 개발 완료 후, 로컬에서 `LSWMD.pkl`을
   `data/raw/`에 두고 실행하는 방식으로 진행하면 되는지.
2. **제품 시나리오** — 가상 제품을 DRAM(1z nm)으로 설정할지, 로직/파운드리로 할지.
   (패턴-공정 매핑 자체는 큰 차이 없으나 용어·스텝 명칭에 영향)
3. **데이터 규모** — 라벨이 있는 172,950장만 사용(표준 관행)할지, 미라벨 638k장으로
   준지도/이상탐지까지 확장할지.
4. **ROI 가정치** — wafer 1장당 가치, 월 투입 매수 기본값. (미정 시 공개 자료 기반 가정치를
   사용하고 앱에서 조정 가능하게 처리)
