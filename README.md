# BISTRO-LSTM

**Task-Specific vs Foundation vs LLM In-Context: Korean Macro Forecasting Benchmark**

170K 파라미터의 task-specific Transformer가 91~200M 파라미터의 Foundation Model 및 LLM in-context forecaster(HCX-32B, Claude)와 한국 거시지표 예측에서 경쟁한 결과를 정리한 프로젝트. 초기엔 CPI YoY 예측에서 출발했고, 이후 **rolling-2025 OOS 평가**, **regime-gated escalation**, **자기개선 멀티에이전트 루프**로 확장되었다.

> **방향 전환 (2026-06)**: 발주처(한국은행) 요청으로 타깃이 **CPI → GDP 예측**으로 전환됨. 현 코드베이스는 CPI 기준이며, GDP 전환 계획은 [Roadmap](#roadmap--cpi--gdp-전환) 참조.

---

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

## Quick Start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 학습 + 평가
python train_and_evaluate.py

# 2-Stage 파이프라인 (29var screening → top-K refinement)
python lstm_runner.py --top-k 10

# Economic Narrative 분석
python causal_narrative.py

# Rolling 2025 OOS — foundation models
python rolling_2025_chronos.py
python rolling_2025_moirai.py

# Regime-gated escalation
python regime_detector_backtest.py
python hcx_value_quantification.py

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

## Project Structure

```
bistro-lstm/
├── lstm_core.py            — Domain classes, presets, tier labels, loaders
├── lstm_model.py           — AttentionLSTMForecaster (PyTorch)
├── lstm_trainer.py         — Walk-forward CV + Optuna tuning
├── lstm_runner.py          — 2-Stage pipeline + narrative integration
├── transformer_model.py    — Task-specific Transformer (170K)
├── causal_narrative.py     — Economic Narrative Analysis (CF, Jacobian, Pathway)
├── feature_importance.py   — Integrated Gradients + Permutation importance
├── ablation_study.py       — Variable removal/addition experiments
├── train_and_evaluate.py   — Main training entry point
├── comparison.py           — BISTRO vs LSTM comparison
├── app.py                  — Streamlit dashboard (9 tabs)
├── export_pdf.py           — PDF report generator
├── data_collector.py       — FRED/BIS data collection + panel build
├── build_augmented_panel.py— optimal18 + Google Trends 병합
├── fetch_ecos_csi.py       — BoK CSI(소비자심리지수) fetch
├── preprocessing_util.py   — Monthly panel preprocessing (z-score, splits)
├── inference_util.py       — AR(1) baseline
│
├── rolling_2025*.py        — Rolling 2025 OOS: HCX/Claude in-context + ablations
├── rolling_2025_chronos.py — Chronos zero-shot
├── rolling_2025_moirai.py  — Moirai zero-shot (+CSI covariate)
├── run_foundation_models.py— Chronos/Sundial/TimesFM 통합 러너
├── hcx_*.py                — HCX-32B forecaster 변형 (clean/univar/no-think)
├── posthoc_*.py            — seed 집계 + 앙상블 분해
├── regime_detector_backtest.py  — 4 statistical shock detectors
├── hcx_value_quantification.py  — regime별 HCX 가치 정량화
│
├── agents/                 — Researcher→Engineer→Analyzer 자기개선 루프
│   ├── orchestrator.py  researcher.py  engineer.py  analyzer.py
│   ├── schemas.py  store.py  manual_run.py
│
├── data/
│   ├── macro_panel*.csv             — 거시 패널 (29 / optimal18 / aug)
│   ├── google_trends_kr.csv         — Google Trends 키워드
│   ├── experiments.db               — 멀티에이전트 실험 DB
│   ├── cognition.json               — 누적 lessons
│   ├── rolling_2025_*_results.json  — rolling OOS 결과
│   ├── regime_detector_backtest.json
│   └── hcx_value_by_regime.json
└── requirements.txt
```

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

## Roadmap — CPI → GDP 전환

2026-06 한국은행 요청으로 예측 타깃이 **CPI → GDP 성장률**로 전환. 이식 계획:

| 단계 | 내용 | 재사용 여부 |
|------|------|------------|
| 1. 데이터 레이어 | 분기 GDP + 혼합빈도(mixed-frequency) 공변량 수집 | ❌ 재설계 |
| 2. 공변량 재선정 | GDP 드라이버(투자·소비·순수출·심리) 기준 | ❌ 재실행 |
| 3. DFM 베이스라인 | `statsmodels` DynamicFactorMQ로 nowcasting 기준선 | ➕ 신규 |
| 4. 하네스 이식 | rolling 평가 / FM 비교 / regime-gating을 분기로 | ✅ 대부분 이식 |
| 5. LLM in-context | HCX/Claude 프롬프트를 GDP·분기로 교체 | ✅ 구조 재사용 |

> **핵심 제약**: GDP는 분기 데이터(~90 obs since 2003)라 CPI 월별(276) 대비 소표본. 327K LSTM/Transformer 단독은 불리하며, 중앙은행 표준인 **DFM / MIDAS / bridge equation**을 베이스라인으로 깔고 task-specific 모델을 챌린저로 두는 구도가 현실적.

---

## Related Project

- [BISTRO-XAI](../bistro-xai/) — BISTRO Transformer (Moirai 91M) 기반 XAI 파이프라인
