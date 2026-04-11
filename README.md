# BISTRO-LSTM

**Task-Specific vs Foundation Model: Korean CPI Forecasting Benchmark**

170K 파라미터의 task-specific Transformer가 91~200M 파라미터의 Foundation Model들과 한국 CPI YoY 예측에서 경쟁한 결과를 정리한 프로젝트.

## Key Results (2023 + 2024 OOS)

| Model | Params | 2023 RMSE | 2024 RMSE | Avg RMSE | Type |
|-------|--------|-----------|-----------|----------|------|
| **Transformer (Ours)** | **170K** | **0.556** | **0.556** | **0.556** | Task-specific |
| LSTM (Ours) | 327K | 0.690 | 0.717 | 0.703 | Task-specific |
| Chronos-2 | 120M | 0.536 | 1.327 | 0.932 | Zero-shot |
| BISTRO (Moirai) | 91M | 1.161 | 0.811 | 0.986 | Pre-trained |
| TimesFM | 200M | 1.370 | — | — | Zero-shot |
| AR(1) Baseline | — | 1.133 | — | — | Statistical |

## Architecture

```
AttentionLSTMForecaster (~327K params)
├── Variable Embedding       — per-variable linear projection
├── Variable Fusion Attention — cross-variable MHA (4 heads)
├── Stacked LSTM Encoder     — 2 layers, hidden=128
├── Temporal Attention Decoder — learnable forecast queries
└── Output Head              — Gaussian NLL (mu + log_sigma)
```

## Economic Narrative Analysis (NEW)

어텐션 분석을 넘어, 변수가 CPI에 **어떻게, 얼마나, 어떤 경로로** 영향을 미치는지 경제적 서사로 풀어내는 분석 모듈.

### 4가지 분석 방법

| 방법 | 답하는 질문 | 원리 |
|------|-----------|------|
| **Counterfactual** | 얼마나, 어느 방향으로? | 변수 ±1σ 충격 → CPI pp 변화 |
| **Jacobian Lag** | 몇 개월 후 반영? | ∂CPI/∂x per timestep (IRF) |
| **Pathway Decomposition** | 어떤 경로로? | 직접 vs 채널별 간접 효과 분해 |
| **Narrative Generation** | 종합 서사 | 위 3가지 결합 → 자동 경제 해석 |

### 서사 출력 예시

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

## Quick Start

```bash
# 환경 설정
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 학습 + 평가
python train_and_evaluate.py

# 2-Stage 파이프라인 (29var screening → top-K refinement)
python lstm_runner.py --top-k 10

# Economic Narrative 분석
python causal_narrative.py

# Ablation Study
python ablation_study.py

# 대시보드 (11개 탭)
streamlit run app.py
```

## Dashboard

11개 탭으로 구성된 Streamlit 대시보드:

| Tab | 내용 |
|-----|------|
| Key Findings | 전체 모델 비교 요약 |
| LSTM vs Transformer | 10-seed 실험, 통계 검정 |
| 2024 Forecast | 2024 OOS 비교 |
| Training Process | Walk-forward CV, 학습 곡선 |
| Forecast Results | 2023 OOS 예측 차트 |
| Variable Importance | Cross-variate attention matrix |
| Temporal Patterns | Decoder temporal attention |
| Feature Selection | Stage 1 변수 랭킹 |
| Ablation & Incremental | Leave-one-out + 순차 추가 |
| BISTRO Comparison | LSTM vs BISTRO 상세 비교 |
| **Economic Narrative** | **CF + Jacobian + 경로분해 + 자동 서사** |

## Project Structure

```
bistro-lstm/
├── lstm_core.py            — Domain classes, presets, tier labels, loaders
├── lstm_model.py           — AttentionLSTMForecaster (PyTorch)
├── lstm_trainer.py         — Walk-forward CV + Optuna tuning
├── lstm_runner.py          — 2-Stage pipeline + narrative integration
├── causal_narrative.py     — Economic Narrative Analysis (CF, Jacobian, Pathway)
├── feature_importance.py   — Integrated Gradients + Permutation importance
├── ablation_study.py       — Variable removal/addition experiments
├── train_and_evaluate.py   — Main training entry point
├── comparison.py           — BISTRO vs LSTM comparison
├── transformer_model.py    — Task-specific Transformer (170K)
├── app.py                  — Streamlit dashboard (11 tabs)
├── export_pdf.py           — PDF report generator
├── data_collector.py       — FRED API data collection
├── preprocessing_util.py   — Monthly panel preprocessing (z-score, splits)
├── inference_util.py       — AR(1) baseline
├── run_foundation_models.py — Foundation model (Chronos, TimesFM) runners
├── data/
│   ├── macro_panel.csv              — 29 macroeconomic variables (2003-2025)
│   ├── macro_panel_optimal18.csv    — 18 selected variables
│   ├── lstm_inference_results.npz   — LSTM forecast + attention
│   ├── lstm_narrative_results.npz   — Economic narrative analysis
│   ├── lstm_stage1_screening.npz    — Stage 1 variable ranking
│   ├── lstm_ablation_results.npz    — Ablation study results
│   ├── lstm_model_best.pt           — Model checkpoint
│   └── ...                          — Foundation model results
└── requirements.txt
```

## XAI Methods (6 types)

| Method | File | What it measures |
|--------|------|-----------------|
| Cross-variable Attention | `feature_importance.py` | Variable i → j attention (N×N) |
| Integrated Gradients | `feature_importance.py` | Input → output end-to-end attribution |
| Permutation Importance | `feature_importance.py` | ΔRMSE when variable shuffled |
| Ablation Study | `ablation_study.py` | ΔRMSE when variable removed |
| **Counterfactual + Jacobian** | `causal_narrative.py` | **Direction, magnitude, lag, pathway** |
| **Economic Narrative** | `causal_narrative.py` | **Auto-generated economic story** |

## Data

- **Target**: Korean CPI YoY (%)
- **Covariates**: 18 optimal macroeconomic variables (from 29 candidates)
- **Period**: 2003-01 ~ 2025-12 (276 months)
- **Train**: 2003-2022 | **OOS Test**: 2023 + 2024
- **Source**: FRED API, BIS, Bank of Korea

## Related Project

- [BISTRO-XAI](../bistro-xai/) — BISTRO Transformer (Moirai 91M) 기반 XAI 파이프라인
