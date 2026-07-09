# BISTRO-LSTM

**한국 거시지표 예측 벤치마크 → 한국은행 GDP Nowcasting 협업**

CPI YoY 예측 벤치마크(task-specific 소형 모델 vs Foundation Model vs LLM in-context)에서 출발해, 2026-06부터 **한국은행 GDP Nowcasting 시스템 고도화 협업**이 본 트랙이 된 프로젝트. CPI 챕터에서 얻은 lesson(특히 **regime-conditional value** — 모델의 가치는 국면에 따라 다르다)이 GDP 협업의 핵심 제안(regime-gated 앙상블)으로 이어졌다.

| 트랙 | 상태 | 위치 |
|---|---|---|
| **GDP Nowcasting 협업 (한국은행)** | **진행 중** — v1 보고(7/2), v2 후보 공유(7/9) | [`gdp/`](gdp/), [`docs/`](docs/) |
| CPI 벤치마크 (22단계 ablation, rolling-2025) | 완료 — lesson 아카이브 | 아래 "완료된 챕터" |

---

## 🔥 현재 진행 — GDP Nowcasting 협업 (한국은행, 2026-06~)

한은 GDP Nowcasting 시스템을 동일 검증체계에서 **재현**하고, 국면전환(regime-gated) 앙상블로 **개선**하는 협업. 분석 코드는 [`gdp/`](gdp/) 참조. (데이터·아티팩트는 기밀이라 미포함)

**성과 (속보치 flash, 전망주차 w[-19,-1] 평균 RMSE, 2018Q1–2025Q4):**

| 모형 | 전체 32분기 | 상태 |
|---|---|---|
| **regime-gated v2 (3-arm, 반등 국면 추가)** | **0.722** (strict 0.738) | 유망 후보 — 7/9 한은 공유 |
| regime-gated v1 (2-arm: shock/calm) | 0.755 | 7/2 중간보고 |
| DFM+XGBoost (한은 기존 최고) | 0.765 | 재현 완료 |
| DFM (한은 기준선) | 0.865 | 재현 완료 |
| Transformer·Foundation·AttnLSTM (단독/보정) | 0.94–1.45 | 기각 — 소표본에서 열위 |

**타임라인과 현재 위치:**

- **7/2 중간보고** — 재현 성공(소수점 수준) + v1(shock→DFM+XGB / calm→DFM+RF, 실시간 충격탐지기) 0.755 + 딥러닝 한계 정직 보고 → 최신기술은 설명가능성·국면진단 역할 제안
- **7/8~9 v2 발견** — 반등 국면(직전 분기 flash<0 & 심리 저점통과) 추가, 반등기엔 **보정 OFF(DFM 단독)**. 핵심 통찰: *반등 국면은 학습 표본에 드물어 트리 보정이 잡음이 됨*. NN(AttnLSTM) 반등 arm은 기각, LLM arm은 보류(무오염 검증 전, 기준선이 "반등기 DFM 단독"으로 상향)
- **다음 단계** — ① DM 검정(v2 vs v1/DFM+XGB, 한은 보고 전 필수) ② 반등 정의 고정 후 재검증 ③ 충격탐지기 고도화(월별 지표 조기 감지) ④ 설명가능성 레이어

상세: [`docs/GDP_모델_평가표_2026-07-02.md`](docs/GDP_모델_평가표_2026-07-02.md) · [`docs/regime-gated_구조_2026-07-02.md`](docs/regime-gated_구조_2026-07-02.md) · v2 요약 슬라이드 [`docs/regime-gated_v2_3arm_1p_2026-07-09.pptx`](docs/regime-gated_v2_3arm_1p_2026-07-09.pptx)

---

# 완료된 챕터 — CPI 벤치마크 (2026-04~06, lesson 아카이브)

> 아래는 CPI YoY 예측 벤치마크의 결과 기록이다. 22단계 ablation으로 prompt/모델 공간을 탐색했고,
> 여기서 확립된 lesson들(N=8 multi-seed, baseline ladder, vintage=cutoff 통제, Minimal Prompt Principle,
> **regime-conditional value**)이 GDP 협업의 방법론적 기반이 됐다.

## Key Results

### 1) 2023 + 2024 OOS (12-step, CPI YoY)

| Model | Params | 2023 RMSE | 2024 RMSE | Avg RMSE | Type |
|-------|--------|-----------|-----------|----------|------|
| **Transformer (Ours)** | **170K** | **0.556** | **0.556** | **0.556** | Task-specific |
| LSTM (Ours) | 327K | 0.690 | 0.717 | 0.703 | Task-specific |
| Chronos-2 | 120M | 0.536 | 1.327 | 0.932 | Zero-shot |
| BISTRO (Moirai) | 91M | 1.161 | 0.811 | 0.986 | Pre-trained |
| TimesFM | 200M | 1.370 | — | — | Zero-shot |
| AR(1) Baseline | — | 1.133 | — | — | Statistical |

### 2) Rolling 2025 OOS (1-step, 8 origins 2025-04..2025-11, post-LLM-cutoff)

서로 다른 평가 프로토콜 — 매월 직전까지의 데이터만으로 1개월 앞 CPI YoY를 예측. LLM 학습 컷오프 이후 구간이라 contamination이 없다.

| Model | RMSE | MAE | Type |
|-------|------|-----|------|
| **HCX-32B (forced CoT + CSI)** | **0.250** | — | LLM in-context |
| Moirai-1.0-R-small (+CSI, ctx36) | 0.254 | 0.220 | Zero-shot FM |
| Trend12 baseline | 0.258 | — | Statistical |
| Chronos-Bolt-tiny (ctxALL) | 0.281 | 0.238 | Zero-shot FM |
| In-domain LSTM | 0.484 | — | Task-specific |
| In-domain TFM | 0.773 | — | Task-specific |

> **핵심 관찰**: 평온한 2025 1-step 구간에서는 **단순 Trend12가 매우 강력**하고 LLM/FM의 우위는 근소하다. task-specific 소형 모델은 오히려 열위. → 값어치는 "언제나"가 아니라 **shock 구간에서만** 나온다는 가설로 이어짐 (아래 Regime-Gated Escalation).

---

## Architecture

```
AttentionLSTMForecaster (~327K params)
├── Variable Embedding       — per-variable linear projection
├── Variable Fusion Attention — cross-variable MHA (4 heads)
├── Stacked LSTM Encoder     — 2 layers, hidden=128
├── Temporal Attention Decoder — learnable forecast queries
└── Output Head              — Gaussian NLL (mu + log_sigma)
```

---

## Economic Narrative Analysis

어텐션 분석을 넘어, 변수가 CPI에 **어떻게, 얼마나, 어떤 경로로** 영향을 미치는지 경제적 서사로 풀어내는 분석 모듈.

| 방법 | 답하는 질문 | 원리 |
|------|-----------|------|
| **Counterfactual** | 얼마나, 어느 방향으로? | 변수 ±1σ 충격 → CPI pp 변화 |
| **Jacobian Lag** | 몇 개월 후 반영? | ∂CPI/∂x per timestep (IRF) |
| **Pathway Decomposition** | 어떤 경로로? | 직접 vs 채널별 간접 효과 분해 |
| **Narrative Generation** | 종합 서사 | 위 3가지 결합 → 자동 경제 해석 |

```
CN_Interbank3M → Korean CPI 영향 분석
Tier: T3 | 채널: 통화/금융

총 영향도: +1σ(약 1.19) 충격 시 CPI 하락 0.3486pp (12개월 평균)
피크 시차: 약 12개월의 중기 시차
비대칭성: 상승 충격이 하락 충격보다 46% 더 강함

경로 분해:
  직접 효과:     -0.5105pp (84%)
  환율 경유:     -0.0524pp (9%)
  통화/금융 경유: -0.0234pp (4%)
  원자재 경유:   -0.0113pp (2%)
```

---

## LLM In-Context Forecasting (HCX-32B / Claude)

LLM에게 거시 패널을 TSV로 주고 1-step CPI를 직접 예측하게 하는 in-context forecaster. rolling-2025 OOS에서 다수 ablation을 수행.

| 스크립트 | 실험 |
|---------|------|
| `rolling_2025.py` / `rolling_2025_full.py` | HCX 기본 rolling (cov_base 18 vs cov_csi) |
| `rolling_2025_claude.py` / `rolling_2025_blinded_claude.py` | Claude(Opus/Sonnet) 예측 + blinded contamination 테스트 |
| `rolling_2025_blinded.py` | 변수명/시점 마스킹(blinded) — contamination 검증 |
| `rolling_2025_fewshot.py` | few-shot 예시 + forced CoT |
| `rolling_2025_format.py` | 입력 포맷 ablation (헤더/단위/sentinel) |
| `rolling_2025_csi_forced.py` | CSI + forced-attention 프롬프트 |
| `rolling_2025_aug.py` | Google Trends 5개 추가 (cov_aug) |
| `rolling_2025_extra_ablations.py` / `rolling_2025_paper_ablations.py` | 문헌 기반 ablation |
| `hcx_ablation.py` | (cov/univar) × (think on/off) × (2023/2024) seed ablation |
| `posthoc_aggregation.py` / `posthoc_ensemble_perseed.py` | seed 집계(median/trimmed) + HCX×Trend12 앙상블 분해 |

---

## Foundation Models (Zero-shot)

| 스크립트 | 모델 | 입력 |
|---------|------|------|
| `rolling_2025_chronos.py` | Chronos-Bolt (tiny~base) | univariate CPI |
| `rolling_2025_moirai.py` | Moirai-1.0-R (small~large) | univariate + BoK_CSI past covariate |
| `run_foundation_models.py` | Chronos / Sundial / TimesFM | 통합 러너 |

> 최적 변형: Chronos-Bolt-tiny(ctxALL) 0.281, Moirai-1.0-R-small+CSI(ctx36) 0.254. CSI를 leading indicator로 쓴 Moirai가 동급 최강 FM.

---

## Regime-Gated Escalation

평온 구간엔 Trend12, **shock 구간에만 LLM으로 escalate**하자는 가설을 정량 검증.

### 1) Regime Detector Backtest (`regime_detector_backtest.py`)

Ground truth = `|Trend12 1-step 오차| > 0.5pp` (단순 baseline이 실제로 실패한 달). 모든 디텍터는 vintage-safe (시점 t 점수는 t까지 데이터만 사용).

| Detector | AUC | TPR@opt | FPR@opt |
|----------|-----|---------|---------|
| **D. Multi-signal (z-sum)** | **0.704** | 0.589 | 0.226 |
| A. Rolling σ 12m | 0.695 | 0.421 | 0.095 |
| C. MAD around MA12 | 0.580 | 0.378 | 0.199 |
| B. CUSUM 2-sided | 0.557 | 0.326 | 0.107 |

### 2) HCX Value by Regime (`hcx_value_quantification.py`)

D Multi-signal(threshold 0.362)로 shock 여부를 나눠 HCX vs Trend12 비교:

| Regime | HCX RMSE | Trend12 RMSE | Winner |
|--------|----------|--------------|--------|
| Non-flagged (calm) | 0.293 | **0.248** | Trend12 (HCX +0.038 손해) |
| **Shock-flagged** | **0.441** | 0.568 | **HCX (−0.098 이득)** |

> 결론: LLM 예측은 평온 구간엔 오히려 손해, **shock 구간에서만 가치**. → regime detector를 게이트로 두는 escalation 전략이 합리적.

---

## Multi-Agent Experiment Loop (`agents/`)

HCX in-context 예측을 자동으로 개선하는 Researcher → Engineer → Analyzer 루프.

```
Researcher (Claude Opus)  — 다음 실험 후보(Candidate) 가설 생성
        ↓
Engineer   (HCX runner)   — 후보 설정으로 rolling 예측 실행 → Metric
        ↓
Analyzer   (Claude Opus)  — 결과 해석, Lesson 추출 → cognition.json 누적
        ↺  (N rounds, experiments.db에 기록)
```

```bash
python -m agents.orchestrator --rounds 5
python -m agents.orchestrator --rounds 1 --dry-run     # 가설만 제안, HCX 호출 skip
```

- `schemas.py` — Candidate / Metric / Lesson 스키마
- `store.py` — `data/experiments.db` (SQLite) 영속화
- `data/cognition.json` — 누적된 lessons

---

## 폴더 구조

```
core/         공유 라이브러리 — lstm_core, lstm_model, lstm_trainer, transformer_model,
              preprocessing_util, inference_util
training/     LSTM 학습·평가 — train_and_evaluate, train_optimal, lstm_runner, ablation_study
hcx/          HCX in-context 예측 실험 (hcx_forecast*, hcx_ablation, hcx_no_think, ...)
rolling2025/  Rolling 2025 OOS 평가 스위트 (rolling_2025*, run_foundation_models, stress_*)
analysis/     분석·사후처리 — comparison, causal_narrative, feature_importance,
              posthoc_*, regime_detector_backtest, ensemble_decomposition
datatools/    데이터 수집·패널 구축 — data_collector, fetch_ecos_csi, build_augmented_panel
report/       PDF 리포트 생성 — export_pdf, export_report_v1, make_onboarding_pdf
agents/       멀티에이전트 실험 루프
gdp/          한국은행 GDP Nowcasting 협업 (분석 코드 아카이브)
data/         패널 CSV·실험 결과 (npz/json/db)
app.py        Streamlit 대시보드 (루트 유지)
```

> 모든 스크립트는 **저장소 루트에서 실행**합니다. 내부 경로는 소스 위치 기준으로
> 루트를 찾도록 되어 있어 `python training/train_and_evaluate.py` 처럼 바로 실행 가능.

## Quick Start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 학습 + 평가
python training/train_and_evaluate.py

# 2-Stage 파이프라인 (29var screening → top-K refinement)
python training/lstm_runner.py --top-k 10

# Economic Narrative 분석
python analysis/causal_narrative.py

# Rolling 2025 OOS — foundation models
python rolling2025/rolling_2025_chronos.py
python rolling2025/rolling_2025_moirai.py

# Regime-gated escalation
python analysis/regime_detector_backtest.py
python hcx/hcx_value_quantification.py

# 멀티에이전트 개선 루프
python -m agents.orchestrator --rounds 5

# 대시보드 (9개 탭)
streamlit run app.py
```

---

## Dashboard

9개 탭으로 구성된 Streamlit 대시보드:

| Tab | 내용 |
|-----|------|
| Key Findings | 전체 모델 비교 요약 |
| LSTM vs Transformer | 10-seed 실험, 통계 검정 |
| 2024 Forecast | 2024 OOS 비교 |
| Training Process | Walk-forward CV, 학습 곡선 |
| Forecast Results | 2023 OOS 예측 차트 |
| Variable Importance | Cross-variate attention matrix |
| Temporal Patterns | Decoder temporal attention |
| BISTRO Comparison | LSTM vs BISTRO 상세 비교 |
| **Economic Narrative** | **CF + Jacobian + 경로분해 + 자동 서사** |

---

## XAI Methods (6 types)

| Method | File | What it measures |
|--------|------|-----------------|
| Cross-variable Attention | `feature_importance.py` | Variable i → j attention (N×N) |
| Integrated Gradients | `feature_importance.py` | Input → output end-to-end attribution |
| Permutation Importance | `feature_importance.py` | ΔRMSE when variable shuffled |
| Ablation Study | `ablation_study.py` | ΔRMSE when variable removed |
| **Counterfactual + Jacobian** | `causal_narrative.py` | **Direction, magnitude, lag, pathway** |
| **Economic Narrative** | `causal_narrative.py` | **Auto-generated economic story** |

---

## Data

- **Target**: Korean CPI YoY (%)  *(GDP 전환 예정 — Roadmap 참조)*
- **Covariates**: 18 optimal macroeconomic variables (from 29 candidates), + BoK CSI / Google Trends 옵션
- **Period**: 2003-01 ~ 2025-12 (276 months)
- **Train**: 2003-2022 | **OOS Test**: 2023 + 2024 (12-step), 2025-04..11 (rolling 1-step)
- **Source**: FRED API, BIS, Bank of Korea (ECOS)

---

## CPI → GDP 전환 (완료)

2026-06 한국은행 요청으로 타깃이 GDP로 전환됐고, 당초의 "CPI 하네스 이식" 계획 대신
**한은 시스템(DFM 기반)을 재현·확장하는 협업 방식**으로 실현됐다 — 상단 "현재 진행" 섹션 참조.
예상대로 분기 GDP 소표본에서 task-specific NN·파운데이션 모델은 열위였고, DFM을 기반으로
국면 규칙을 얹는 구도가 이겼다. CPI 챕터의 regime-conditional lesson이 그대로 적용된 사례.

---

## Related Project

- [BISTRO-XAI](../bistro-xai/) — BISTRO Transformer (Moirai 91M) 기반 XAI 파이프라인
