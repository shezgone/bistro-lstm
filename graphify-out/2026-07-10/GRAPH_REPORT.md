# Graph Report - bistro-lstm  (2026-07-10)

## Corpus Check
- 105 files · ~76,424 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1449 nodes · 1998 edges · 119 communities (97 shown, 22 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 58 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f2972ebf`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Agent Experiment Framework
- Economic Regime Eras
- Dashboard & Results Loading
- Claude Rolling Results
- Baseline Forecasts 2025
- Regime Detector Signals
- Blinded Claude Results
- Paper Ablation Results
- CSI Forced Results
- Extra Ablation Results
- Transformer Architecture
- Google Trends Aug Results
- Blinded HCX Results
- CSI Covariate Results
- Few-shot Results
- HCX Rolling Results
- LSTM Training Pipeline
- Causal Narrative Analysis
- Project Overview Concepts
- Torch CSI Benchmarks
- Format Ablation Results
- phase_b_soft_gate.py
- ZScoreNormalizer
- HCX Value by Regime
- Training & Evaluation
- Ablation Study Runner
- Torch Model Results
- Stress Test Results
- Post-hoc Aggregation Results
- Post-hoc Ensemble Results
- 2025 Actuals
- Consensus Aggregation
- Claude Results Metadata
- HCX-32B Ablation Results
- Bias-corrected Mean
- Ensemble 30/70
- Ensemble 50/50
- Ensemble 70/30
- Max Seed Aggregation
- Mean Aggregation
- Median Aggregation
- Min Seed Aggregation
- Trend12 Baseline
- Trimmed Mean Aggregation
- Chronos Results
- Moirai Results
- Cognition Lessons
- XAI Importance Concepts
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- train_and_evaluate.py
- ZScoreNormalizer
- AttentionLSTMForecaster
- causal_narrative.py
- LSTMConfig
- comparison.py
- regime_detector_backtest.py
- generate
- ablation_study.py
- hcx_value_quantification.py
- run_foundation_models.py
- data_collector.py
- rolling_2025_chronos.py
- rolling_2025_moirai.py
- TorchMLP
- phase_b_harness.py
- TorchSeq
- rolling_2025.py
- rolling_2025_format.py
- phase_b_transformer.py
- rolling_2025_blinded.py
- rolling_2025_fewshot.py
- TorchSeq
- posthoc_aggregation.py
- posthoc_ensemble_perseed.py
- make_onboarding_pdf.py
- rolling_2025_blinded_claude.py
- rolling_2025_extra_ablations.py
- make_bok_report.py
- phase_b_llm.py
- phase_b_regime_gated.py
- phase_b_regime_gated_v2.py
- rolling_2025_aug.py
- rolling_2025_claude.py
- rolling_2025_csi_forced.py
- rolling_2025_full.py
- rolling_2025_paper_ablations.py
- stress_2023_2024_blinded.py
- phase_b_ensemble_search.py
- phase_b_foundation.py
- hcx_ablation.py
- phase_b_compare.py
- ensemble_decomposition.py
- hcx_forecast_clean.py
- build_augmented_panel.py
- fetch_ecos_csi.py
- 메일_한은_regime-gated_v2_2026-07-09.md
- hcx_forecast.py
- hcx_forecast_univar.py
- hcx_no_think.py

## God Nodes (most connected - your core abstractions)
1. `results` - 49 edges
2. `AttentionLSTMForecaster` - 36 edges
3. `Candidate` - 31 edges
4. `ZScoreNormalizer` - 29 edges
5. `LSTMConfig` - 26 edges
6. `train_model()` - 21 edges
7. `Metric` - 17 edges
8. `run_training()` - 17 edges
9. `results` - 17 edges
10. `results` - 17 edges

## Surprising Connections (you probably didn't know these)
- `run_narrative()` --calls--> `run_full_analysis()`  [INFERRED]
  training/lstm_runner.py → analysis/causal_narrative.py
- `run_narrative()` --calls--> `save_narrative_results()`  [INFERRED]
  training/lstm_runner.py → analysis/causal_narrative.py
- `TorchSeq` --uses--> `AttentionLSTMForecaster`  [INFERRED]
  gdp/scripts/phase_b_seq.py → core/lstm_model.py
- `train_detect()` --calls--> `AttentionLSTMForecaster`  [EXTRACTED]
  gdp/scripts/phase_b_tf_detector.py → core/lstm_model.py
- `BISTRO-LSTM` --references--> `fpdf2>=2.7.0`  [EXTRACTED]
  README.md → requirements.txt

## Import Cycles
- None detected.

## Communities (119 total, 22 thin omitted)

### Community 0 - "Agent Experiment Framework"
Cohesion: 0.06
Nodes (69): analyze(), _extract_json(), _format_history_block(), _format_lessons_block(), _format_round_block(), Lesson, Analyzer agent — Claude Opus distills transferable lessons from the latest round, _build_messages() (+61 more)

### Community 1 - "Economic Regime Eras"
Cohesion: 0.05
Nodes (58): first_flag, flagged, period_len, rate, first_flag, flagged, period_len, rate (+50 more)

### Community 2 - "Dashboard & Results Loading"
Cohesion: 0.16
Nodes (12): ar1_forecast(), PeriodIndex, Series, Utility: AR(1) baseline forecast. bistro-xai의 inference_util.py와 동일., Simple AR(1) forecast: y_t = c + phi * y_{t-1}      Parameters     ----------, main(), device, BISTRO-LSTM: Main Entry Point ============================== 학습 → 평가 → 결과 저장 파이프 (+4 more)

### Community 3 - "Claude Rolling Results"
Cohesion: 0.04
Nodes (49): results, claude-opus-4-5|2025-05|cov, claude-opus-4-5|2025-05|univar, claude-opus-4-5|2025-06|cov, claude-opus-4-5|2025-06|univar, claude-opus-4-5|2025-07|cov, claude-opus-4-5|2025-07|univar, claude-opus-4-5|2025-08|cov (+41 more)

### Community 4 - "Baseline Forecasts 2025"
Cohesion: 0.04
Nodes (49): AR(1), BoK linear, MA12, RW, Trend12, AR(1), BoK linear, MA12 (+41 more)

### Community 5 - "Regime Detector Signals"
Cohesion: 0.05
Nodes (41): auc, fpr_at_opt, j_at_opt, n_shock, n_stable, n_total, optimal_threshold, tpr_at_opt (+33 more)

### Community 6 - "Blinded Claude Results"
Cohesion: 0.07
Nodes (39): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+31 more)

### Community 7 - "Paper Ablation Results"
Cohesion: 0.08
Nodes (34): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+26 more)

### Community 8 - "CSI Forced Results"
Cohesion: 0.06
Nodes (33): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+25 more)

### Community 9 - "Extra Ablation Results"
Cohesion: 0.09
Nodes (33): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+25 more)

### Community 10 - "Transformer Architecture"
Cohesion: 0.17
Nodes (11): ① Base 예측 (재료, 3종), ② 국면별 최적 앙상블 (2종), ③ 충격 탐지기 (vintage-safe), ④ 게이트 (스위치), regime-gated 앙상블 — 구조 설명, 구성요소, 다음 단계 후보, 성과 (flash w[−19,−1] 평균 RMSE) (+3 more)

### Community 11 - "Google Trends Aug Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 12 - "Blinded HCX Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 13 - "CSI Covariate Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 14 - "Few-shot Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 15 - "HCX Rolling Results"
Cohesion: 0.06
Nodes (31): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+23 more)

### Community 16 - "LSTM Training Pipeline"
Cohesion: 0.29
Nodes (6): 1. 기간 정의, 2-1. ★ regime-gated 앙상블 (DFM+XGBoost를 넘는 우리 모델), 2. 평가 결과 (flash, w[−19,−1] 평균 RMSE), 3. 판정, 4. 재현 방법 / 산출물, GDP Nowcasting 모델 평가표

### Community 17 - "Causal Narrative Analysis"
Cohesion: 0.33
Nodes (4): panel_feat(), q_first_vintage(), Transformer(어텐션) 기반 충격 탐지기 실험. - 입력: DFM 보정 월별 패널 (최근 L개월 × 변수)  [q의 첫 빈티지 CSV], train_detect()

### Community 18 - "Project Overview Concepts"
Cohesion: 0.08
Nodes (24): 1) 2023 + 2024 OOS (12-step, CPI YoY), 1) Regime Detector Backtest (`regime_detector_backtest.py`), 2) HCX Value by Regime (`hcx_value_quantification.py`), 2) Rolling 2025 OOS (1-step, 8 origins 2025-04..2025-11, post-LLM-cutoff), Architecture, BISTRO-LSTM, CPI → GDP 전환 (완료), Dashboard (+16 more)

### Community 19 - "Torch CSI Benchmarks"
Cohesion: 0.08
Nodes (24): LSTM-base (18), mae, n_params, name, rmse, rows, LSTM-csi (18+CSI), mae (+16 more)

### Community 20 - "Format Ablation Results"
Cohesion: 0.08
Nodes (23): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+15 more)

### Community 21 - "phase_b_soft_gate.py"
Cohesion: 0.17
Nodes (7): g_soft(), p_ae(), p_rule(), soft gate 타당성 프로토타입: (1) 규칙 신호 연속화, (2) 오토인코더 이상탐지 점수 게이트.  ŷ = DFM + 0.5·(1−g)·, 분기 q 시작 직전 3개월 평균 오차의 z. 기준을 최근 win개월로 국한(국소 z) —     전체 이력 기준이면 학습기간(~2017) 대비, vol 기반 충격 가중 (연속) — z = (vol−median)/MAD, 확장창., trough_passed()

### Community 22 - "ZScoreNormalizer"
Cohesion: 0.31
Nodes (4): DataFrame, ndarray, Per-variable z-score 정규화.     학습 세트 기준으로 mean/std 계산, 테스트 세트에 동일 적용., ZScoreNormalizer

### Community 24 - "HCX Value by Regime"
Cohesion: 0.11
Nodes (17): main_per_target, main_summary, non-flagged, shock-flagged, hcx_rmse, mean_delta_err, n, p (+9 more)

### Community 25 - "Training & Evaluation"
Cohesion: 0.25
Nodes (4): create_model_patched(), Phase B 공정 재실험: 우리 신경망(torch MLP)을 그들 backtest에 model_spec으로 주입. → XGBoost와 동일 (, 소형 MLP (우리 신경망 접근). 입력은 파이프라인의 imputer+scaler 통과 후., TorchMLP

### Community 28 - "Ablation Study Runner"
Cohesion: 0.20
Nodes (10): ensemble_with_dfm(), load_baseline(), load_grid(), load_rtf(), Phase B 공통 하네스: 우리 모델 예측을 그들과 동일 잣대로 채점. 평가 그리드/타깃/DFM baseline은 제공 아티팩트에서 로드. f, 평가 그리드: (tq, vintage, week_idx, flash) unique + 참조 xgboost 예측., DFM/LSTM baseline 예측을 그리드에 매핑., flash w[wmin,wmax] 평균 RMSE (모형별). 주차별 RMSE(분기 평균)의 평균. (+2 more)

### Community 29 - "Torch Model Results"
Cohesion: 0.15
Nodes (12): LSTM, mae, n_params, name, rmse, rows, Transformer, mae (+4 more)

### Community 33 - "Stress Test Results"
Cohesion: 0.17
Nodes (11): actuals, 2023, 2024, forecasts, 2023, 2024, n_seeds, rationales (+3 more)

### Community 36 - "Post-hoc Aggregation Results"
Cohesion: 0.22
Nodes (9): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+1 more)

### Community 37 - "Post-hoc Ensemble Results"
Cohesion: 0.20
Nodes (9): ar1_rmse, ensemble_30_per_seed_rmses, ensemble_50_per_seed_rmses, ensemble_70_per_seed_rmses, hcx_per_seed_rmses, median_ensemble_rmse, optimal_rmses, optimal_weights_per_seed (+1 more)

### Community 38 - "2025 Actuals"
Cohesion: 0.22
Nodes (9): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+1 more)

### Community 53 - "Consensus Aggregation"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Consensus (3-closest-to-median)

### Community 54 - "Claude Results Metadata"
Cohesion: 0.40
Nodes (4): ctx_len, n_seeds, targets, temperature

### Community 56 - "HCX-32B Ablation Results"
Cohesion: 0.50
Nodes (3): n_per_cell, rows, temperature

### Community 57 - "Bias-corrected Mean"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Bias-corrected mean (TTA-style)

### Community 58 - "Ensemble 30/70"
Cohesion: 0.33
Nodes (5): mae, preds, rmse, methods, Ensemble HCX 30% + Trend12 70%

### Community 59 - "Ensemble 50/50"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Ensemble HCX 50% + Trend12 50%

### Community 60 - "Ensemble 70/30"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Ensemble HCX 70% + Trend12 30%

### Community 61 - "Max Seed Aggregation"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Max seed (cherry-pick pessimistic)

### Community 62 - "Mean Aggregation"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Mean (current default, baseline)

### Community 63 - "Median Aggregation"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Median (Self-Consistency aggreg.)

### Community 64 - "Min Seed Aggregation"
Cohesion: 0.50
Nodes (4): Min seed (cherry-pick optimistic), mae, preds, rmse

### Community 65 - "Trend12 Baseline"
Cohesion: 0.50
Nodes (4): Trend12 baseline alone, mae, preds, rmse

### Community 66 - "Trimmed Mean Aggregation"
Cohesion: 0.50
Nodes (4): Trimmed mean (k=1, drop min+max), mae, preds, rmse

### Community 78 - "XAI Importance Concepts"
Cohesion: 0.50
Nodes (3): hooks, PostToolUse, PreToolUse

### Community 84 - "Community 84"
Cohesion: 0.10
Nodes (16): Tensor, 최근 forward의 attention weights 반환. (B, N, N), Temporal Attention Decoder.     Learnable forecast queries가 LSTM hidden states에, Per-variable linear projection.     각 변수를 독립적인 learned embedding으로 변환.      Inpu, 최근 forward의 temporal attention weights. (B, pred_len, seq_len), Forward pass.          Parameters         ----------         x : (batch, seq_len, 예측 + 불확실성 추정.          Returns         -------         dict with keys:, Cross-variable attention per timestep.     각 시점에서 변수 간 상호작용을 학습.      Input:  (b (+8 more)

### Community 85 - "Community 85"
Cohesion: 0.39
Nodes (7): load_panel(), main(), make_window(), q_last_month(), Phase B — 우리 AttentionLSTMForecaster를 DFM 보정 월별 패널에 이식. 각 (target quarter q, vin, end_month에서 끝나는 L개월 윈도우 (features only)., train_predict()

### Community 87 - "Community 87"
Cohesion: 0.33
Nodes (3): panel_feat(), q_first_vintage(), Transformer(어텐션) 기반 충격 탐지기 실험. - 입력: DFM 보정 월별 패널 (최근 L개월 × 변수)  [q의 첫 빈티지 CSV]

### Community 89 - "Community 89"
Cohesion: 0.33
Nodes (3): regime-gated 앙상블: 실시간(vintage-safe) 충격 탐지기로 DFM+XGB(shock)/DFM+RF(calm) 전환. 탐지기:, q 직전 K분기 실현 GDP 변동성 (q 이전 정보만)., vol_before()

### Community 90 - "Community 90"
Cohesion: 0.40
Nodes (4): GDP Nowcasting 협업 — 분석 코드, 스크립트, 실행 전제, 핵심 결과 (flash w[-19,-1] 평균 RMSE)

### Community 91 - "Community 91"
Cohesion: 0.50
Nodes (3): inv_rmse_w(), DFM+XGBoost(0.765)를 넘는 앙상블 탐색. 이미 가진 base 예측을 결합(재학습 없음). base: dfm, xgboost, rf, rmse_of()

### Community 93 - "Community 93"
Cohesion: 0.67
Nodes (3): one(), Phase B 종합 비교: 우리 모델(단독/앙상블) vs DFM 0.865 / DFM+XGBoost 0.765. 동일 잣대·국면별., score_sub()

### Community 94 - "train_and_evaluate.py"
Cohesion: 0.14
Nodes (20): EarlyStopping, evaluate(), optuna_objective(), device, ndarray, LSTM Trainer — Walk-Forward CV + Optuna Tuning =================================, 단일 fold 학습.      Returns     -------     dict with best_model_state, train_histo, Walk-forward cross-validation.      Parameters     ----------     splits : prepa (+12 more)

### Community 95 - "ZScoreNormalizer"
Cohesion: 0.17
Nodes (15): BISTRO-LSTM Model — Attention-Augmented Stacked LSTM ===========================, set_seed(), create_sequences(), prepare_walk_forward_splits(), Preprocessing Utility for BISTRO-LSTM ===================================== 월별 매, 슬라이딩 윈도우로 학습용 시퀀스 생성.      Parameters     ----------     data : (T, n_vars) 정규화된, Walk-forward CV용 데이터 분할.      Parameters     ----------     df : 전체 학습 데이터 (Peri, split_train_test() (+7 more)

### Community 96 - "AttentionLSTMForecaster"
Cohesion: 0.27
Nodes (12): compute_all_importance(), compute_gradient_importance(), compute_permutation_importance(), extract_attention_importance(), device, ndarray, Tensor, Feature Importance Methods for LSTM ==================================== 1. Vari (+4 more)

### Community 97 - "causal_narrative.py"
Cohesion: 0.13
Nodes (22): counterfactual_analysis(), generate_narrative(), get_active_channels(), get_variable_channel(), jacobian_lag_analysis(), main(), pathway_decomposition(), DataFrame (+14 more)

### Community 99 - "LSTMConfig"
Cohesion: 0.17
Nodes (14): LSTMConfig, LSTM 모델 설정.      Parameters     ----------     variates    : 변수 이름 리스트 (순서 = 입력, main(), DataFrame, device, BISTRO-LSTM Runner — 2-Stage Inference Pipeline ================================, Stage 2: 선택된 변수로 최종 학습 + 예측., Counterfactual 분석: 각 공변량을 ±1σ perturbation.      Returns     -------     dict wi (+6 more)

### Community 100 - "comparison.py"
Cohesion: 0.07
Nodes (33): compare_forecasts(), compare_variable_rankings(), compute_metrics(), load_bistro_ablation(), load_bistro_results(), load_lstm_ablation(), load_lstm_results(), main() (+25 more)

### Community 101 - "regime_detector_backtest.py"
Cohesion: 0.24
Nodes (15): compute_trend12_errors(), coverage_in_periods(), cusum_two_sided(), evaluate_detector(), mad_around_ma(), main(), multi_signal(), ndarray (+7 more)

### Community 103 - "generate"
Cohesion: 0.21
Nodes (9): generate(), load_all(), make_2023_chart(), make_2024_chart(), make_seed_chart(), FPDF, CPI Forecasting Benchmark - PDF Report v1 ======================================, LSTM vs Transformer seed 비교 차트. (+1 more)

### Community 106 - "ablation_study.py"
Cohesion: 0.19
Nodes (18): load_inference_results(), load_stage1_screening(), BISTRO-LSTM Core — Domain Classes ================================== bistro-xai의, Stage 1 전체 변수 스크리닝 결과 로딩., results_available(), _results_path(), load_macro_panel(), macro_panel.csv 로딩.      Parameters     ----------     csv_path : CSV 파일 경로 (+10 more)

### Community 107 - "hcx_value_quantification.py"
Cohesion: 0.31
Nodes (12): annotate_baselines(), load_2022_h1_round0(), load_2023_2024_blinded(), load_2025_forced_cot(), main(), multi_signal(), DataFrame, ndarray (+4 more)

### Community 108 - "run_foundation_models.py"
Cohesion: 0.26
Nodes (12): load_data(), main(), DataFrame, ndarray, Foundation Model Inference for Korean CPI Forecasting ==========================, Sundial zero-shot inference via HuggingFace transformers., CPI 데이터 로딩 (monthly)., Chronos-2 zero-shot inference. (+4 more)

### Community 110 - "data_collector.py"
Cohesion: 0.27
Nodes (11): build_daily_panel(), build_panel(), build_tournament_daily_panel(), download_all_fred(), download_fred(), load_bis_data(), DataFrame, Macro Variable Data Collector for BISTRO-XAI Feature Selection ================= (+3 more)

### Community 111 - "rolling_2025_chronos.py"
Cohesion: 0.26
Nodes (11): evaluate(), load_series(), main(), origins(), pick_device(), predict_one(), ndarray, Period (+3 more)

### Community 112 - "rolling_2025_moirai.py"
Cohesion: 0.29
Nodes (11): build_predictor(), evaluate(), forecast_one(), load_panel(), main(), origins(), pick_device(), DataFrame (+3 more)

### Community 113 - "TorchMLP"
Cohesion: 0.20
Nodes (6): create_model_patched(), BaseEstimator, RegressorMixin, Phase B 공정 재실험: 우리 신경망(torch MLP)을 그들 backtest에 model_spec으로 주입. → XGBoost와 동일 (, 소형 MLP (우리 신경망 접근). 입력은 파이프라인의 imputer+scaler 통과 후., TorchMLP

### Community 114 - "phase_b_harness.py"
Cohesion: 0.20
Nodes (10): ensemble_with_dfm(), load_baseline(), load_grid(), load_rtf(), Phase B 공통 하네스: 우리 모델 예측을 그들과 동일 잣대로 채점. 평가 그리드/타깃/DFM baseline은 제공 아티팩트에서 로드. f, 평가 그리드: (tq, vintage, week_idx, flash) unique + 참조 xgboost 예측., DFM/LSTM baseline 예측을 그리드에 매핑., flash w[wmin,wmax] 평균 RMSE (모형별). 주차별 RMSE(분기 평균)의 평균. (+2 more)

### Community 115 - "TorchSeq"
Cohesion: 0.27
Nodes (5): cm(), BaseEstimator, RegressorMixin, Phase B — 공정 조건 시퀀스 AttnLSTM. max_lag=12 랙 피처를 (13개월×변수) 시퀀스로 reshape. 그들 backte, TorchSeq

### Community 116 - "rolling_2025.py"
Cohesion: 0.31
Nodes (8): baseline_forecasts(), build_messages(), hcx_call(), main(), Period, Rolling 1-step-ahead forecast for 2025-05 to 2025-12 (post-LLM-cutoff OOS).  Mod, Build prompt for 1-step-ahead forecast given data through origin month., Compute deterministic baseline forecasts for 1-step-ahead from origin.

### Community 117 - "rolling_2025_format.py"
Cohesion: 0.33
Nodes (8): build_messages(), hcx_call(), main(), panel_to_tsv_with_units(), DataFrame, Period, HCX rolling 2025 format ablation: minimal header changes vs current TSV.  Tests, TSV with 'Date' header + units in column names. No other changes.

### Community 118 - "phase_b_transformer.py"
Cohesion: 0.39
Nodes (7): load_panel(), main(), make_window(), q_last_month(), Phase B — 우리 AttentionLSTMForecaster를 DFM 보정 월별 패널에 이식. 각 (target quarter q, vin, end_month에서 끝나는 L개월 윈도우 (features only)., train_predict()

### Community 119 - "rolling_2025_blinded.py"
Cohesion: 0.39
Nodes (7): build_messages(), hcx_call(), main(), panel_blinded_tsv(), Period, HCX rolling 2025 BLINDED evaluation — strip identifying info to test contaminati, Convert to blinded TSV: var_N column names, t=N row labels.

### Community 120 - "rolling_2025_fewshot.py"
Cohesion: 0.39
Nodes (7): build_messages(), hcx_call(), main(), panel_to_tsv(), DataFrame, Period, HCX rolling 2025 OOS with few-shot examples + forced CoT.  For each origin t: -

### Community 121 - "TorchSeq"
Cohesion: 0.36
Nodes (3): cm(), Phase B — 공정 조건 시퀀스 AttnLSTM. max_lag=12 랙 피처를 (13개월×변수) 시퀀스로 reshape. 그들 backte, TorchSeq

### Community 125 - "rolling_2025_blinded_claude.py"
Cohesion: 0.43
Nodes (6): build_messages(), claude_call(), main(), panel_blinded_tsv(), Period, Claude (Opus 4.7 + Sonnet 4.5) BLINDED rolling 2025 — contamination evidence.  K

### Community 126 - "rolling_2025_extra_ablations.py"
Cohesion: 0.48
Nodes (6): build_messages(), hcx_call(), main(), panel_tsv(), Period, HCX rolling 2025 — extra ablations (literature-grounded).  Modes: 1. wrong_cpi:

### Community 128 - "phase_b_llm.py"
Cohesion: 0.47
Nodes (4): build_prompt(), call_llm(), load_vintage(), phase_b_llm: Claude LLM(Fable 5 / Opus 4.8) zero-shot GDP nowcast — 2025Q1·Q2 종전

### Community 129 - "phase_b_regime_gated.py"
Cohesion: 0.33
Nodes (3): regime-gated 앙상블: 실시간(vintage-safe) 충격 탐지기로 DFM+XGB(shock)/DFM+RF(calm) 전환. 탐지기:, q 직전 K분기 실현 GDP 변동성 (q 이전 정보만)., vol_before()

### Community 131 - "rolling_2025_aug.py"
Cohesion: 0.47
Nodes (5): build_messages(), hcx_call(), main(), Period, HCX rolling 2025 OOS: cov_base (18) vs cov_aug (18 + 5 Google Trends).  Direct a

### Community 132 - "rolling_2025_claude.py"
Cohesion: 0.47
Nodes (5): build_messages(), claude_call(), main(), Period, Rolling 1-step-ahead forecast for 2025-05 to 2025-12 with Claude models.  Identi

### Community 133 - "rolling_2025_csi_forced.py"
Cohesion: 0.47
Nodes (5): build_messages(), hcx_call(), main(), Period, HCX rolling 2025 with CSI + FORCED attention prompt.  Re-uses cov_base results f

### Community 134 - "rolling_2025_full.py"
Cohesion: 0.47
Nodes (5): build_messages(), hcx_call(), main(), Period, HCX rolling 2025 OOS: cov_base (18) vs cov_csi (18 + 1 BoK CSI).  Drop Google Tr

### Community 135 - "rolling_2025_paper_ablations.py"
Cohesion: 0.47
Nodes (5): build_messages(), hcx_call(), main(), Period, HCX ablations grounded in literature.  Modes (system msg variants only — Minimal

### Community 136 - "stress_2023_2024_blinded.py"
Cohesion: 0.53
Nodes (5): build_messages(), hcx_call(), main(), panel_blinded_tsv(), HCX blinded 12-step forecast for 2023 + 2024 (shock + transition periods).  Test

### Community 137 - "phase_b_ensemble_search.py"
Cohesion: 0.50
Nodes (3): inv_rmse_w(), DFM+XGBoost(0.765)를 넘는 앙상블 탐색. 이미 가진 base 예측을 결합(재학습 없음). base: dfm, xgboost, rf, rmse_of()

### Community 138 - "phase_b_foundation.py"
Cohesion: 0.60
Nodes (4): chronos_forecast(), main(), moirai_forecast(), Phase B — Foundation(Chronos/Moirai) zero-shot, 분기 flash 외삽 baseline. 한계: 외삽형이라

### Community 139 - "hcx_ablation.py"
Cohesion: 0.60
Nodes (4): build_messages(), call_once(), main(), HCX-32B-Think n-seed ablation: (cov/univar) x (think on/off) x (2023/2024).

### Community 140 - "phase_b_compare.py"
Cohesion: 0.67
Nodes (3): one(), Phase B 종합 비교: 우리 모델(단독/앙상블) vs DFM 0.865 / DFM+XGBoost 0.765. 동일 잣대·국면별., score_sub()

## Knowledge Gaps
- **574 isolated node(s):** `실행 전제`, `스크립트`, `핵심 결과 (flash w[-19,-1] 평균 RMSE)`, `🔥 현재 진행 — GDP Nowcasting 협업 (한국은행, 2026-06~)`, `1) 2023 + 2024 OOS (12-step, CPI YoY)` (+569 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AttentionLSTMForecaster` connect `causal_narrative.py` to `AttentionLSTMForecaster`, `Dashboard & Results Loading`, `LSTMConfig`, `ablation_study.py`, `Causal Narrative Analysis`, `TorchSeq`, `Community 84`, `phase_b_transformer.py`, `train_and_evaluate.py`, `ZScoreNormalizer`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Why does `AttentionTransformerForecaster` connect `Community 84` to `ZScoreNormalizer`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Why does `coverage` connect `Economic Regime Eras` to `Regime Detector Signals`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `AttentionLSTMForecaster` (e.g. with `EarlyStopping` and `TorchSeq`) actually correct?**
  _`AttentionLSTMForecaster` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `Candidate` (e.g. with `Candidate` and `DataFrame`) actually correct?**
  _`Candidate` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `soft gate 타당성 프로토타입: (1) 규칙 신호 연속화, (2) 오토인코더 이상탐지 점수 게이트.  ŷ = DFM + 0.5·(1−g)·`, `vol 기반 충격 가중 (연속) — z = (vol−median)/MAD, 확장창.`, `분기 q 시작 직전 3개월 평균 오차의 z. 기준을 최근 win개월로 국한(국소 z) —     전체 이력 기준이면 학습기간(~2017) 대비` to the rest of the system?**
  _747 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Agent Experiment Framework` be split into smaller, more focused modules?**
  _Cohesion score 0.059876543209876544 - nodes in this community are weakly interconnected._