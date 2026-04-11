"""
CPI Forecasting Benchmark - PDF Report v1
==========================================
대시보드 핵심 내용을 PDF로 생성.

Usage:
    .venv/bin/python3 export_report_v1.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR = os.path.join(os.path.dirname(__file__), "report_images")
os.makedirs(IMG_DIR, exist_ok=True)


def load_all():
    """모든 결과 로딩."""
    d = {}

    d["lstm"] = dict(np.load(os.path.join(DATA_DIR, "lstm_inference_results.npz"), allow_pickle=True))
    d["lt"] = dict(np.load(os.path.join(DATA_DIR, "lstm_vs_transformer_ours.npz"), allow_pickle=True))
    d["tfm23"] = dict(np.load(os.path.join(DATA_DIR, "transformer_ours_2023.npz"), allow_pickle=True))
    d["f24"] = dict(np.load(os.path.join(DATA_DIR, "forecast_2024_comparison.npz"), allow_pickle=True))

    bistro_path = os.path.join("..", "bistro-xai", "data", "forecast_optimal18.npz")
    if os.path.exists(bistro_path):
        d["bistro"] = dict(np.load(bistro_path, allow_pickle=True))

    bistro_base = os.path.join("..", "bistro-xai", "data", "real_inference_results.npz")
    if os.path.exists(bistro_base):
        d["bistro_base"] = dict(np.load(bistro_base, allow_pickle=True))

    for name in ["chronos_result", "chronos2_result", "chronos2_bolt_result", "foundation_model_results"]:
        path = os.path.join(DATA_DIR, f"{name}.npz")
        if os.path.exists(path):
            d[name] = dict(np.load(path, allow_pickle=True))

    # CPI panel
    panel = pd.read_csv(os.path.join(DATA_DIR, "macro_panel_optimal18.csv"), index_col=0, parse_dates=True)
    d["panel"] = panel

    return d


def make_2023_chart(d):
    """2023 OOS 차트."""
    panel = d["panel"]
    cpi = panel["CPI_KR_YoY"].resample("MS").last().dropna()
    cpi_show = cpi.loc["2021-01":"2025-12"]
    cpi_before = cpi_show.loc[:"2022-12"]
    cpi_fc = cpi_show.loc["2023-01":"2023-12"]
    cpi_after = cpi_show.loc["2024-01":]

    fc_dates = pd.to_datetime([f"2023-{m:02d}-01" for m in range(1, 13)])
    actual = d["lstm"]["forecast_actual"]

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(x=cpi_before.index, y=cpi_before.values,
                             mode="lines", line=dict(color="#000000", width=3), name="Actual CPI"))

    _fc_x = [cpi_before.index[-1]] + list(cpi_fc.index)
    _fc_y = [cpi_before.values[-1]] + list(cpi_fc.values)
    fig.add_trace(go.Scatter(x=_fc_x, y=_fc_y, mode="lines+markers",
                             line=dict(color="#000000", width=4),
                             marker=dict(size=10, symbol="square", color="#D62728",
                                         line=dict(width=2, color="#000000")),
                             name="Actual (forecast)"))

    if len(cpi_after) > 0:
        _af_x = [cpi_fc.index[-1]] + list(cpi_after.index)
        _af_y = [cpi_fc.values[-1]] + list(cpi_after.values)
        fig.add_trace(go.Scatter(x=_af_x, y=_af_y, mode="lines",
                                 line=dict(color="#000000", width=3), showlegend=False))

    # LSTM CI
    lstm_lo = d["lstm"]["forecast_ci_lo"]
    lstm_hi = d["lstm"]["forecast_ci_hi"]
    fig.add_trace(go.Scatter(x=list(fc_dates)+list(fc_dates[::-1]),
                             y=list(lstm_hi)+list(lstm_lo[::-1]),
                             fill="toself", fillcolor="rgba(33,150,243,0.15)",
                             line=dict(width=0), name="LSTM 90% CI"))

    # LSTM
    fig.add_trace(go.Scatter(x=fc_dates, y=d["lstm"]["forecast_med"],
                             mode="lines+markers", line=dict(color="#2196F3", width=2.5),
                             marker=dict(size=7), name="LSTM (327K)"))

    # Transformer
    fig.add_trace(go.Scatter(x=fc_dates, y=d["tfm23"]["forecast_med"],
                             mode="lines+markers", line=dict(color="#E91E63", width=3),
                             marker=dict(size=9, symbol="star"), name="TFM Ours (170K)"))

    # BISTRO
    if "bistro" in d:
        fig.add_trace(go.Scatter(x=fc_dates, y=d["bistro"]["forecast_med"],
                                 mode="lines+markers", line=dict(color="#FF9800", width=3, dash="dot"),
                                 marker=dict(size=8, symbol="diamond"), name="BISTRO (91M)"))

    # Chronos-2
    if "chronos2_result" in d:
        fig.add_trace(go.Scatter(x=fc_dates, y=d["chronos2_result"]["forecast_med"],
                                 mode="lines+markers", line=dict(color="#4CAF50", width=3),
                                 marker=dict(size=9, symbol="hexagram"), name="Chronos-2 (120M)"))

    # AR(1)
    fig.add_trace(go.Scatter(x=fc_dates, y=d["lstm"]["forecast_ar1"],
                             mode="lines+markers", line=dict(color="#888888", width=2, dash="dash"),
                             marker=dict(size=7), name="AR(1)"))

    fig.add_vline(x=fc_dates[0].timestamp()*1000, line_dash="dash", line_color="rgba(100,100,100,0.6)")

    fig.update_layout(width=1100, height=480, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, font=dict(size=11)),
                      margin=dict(t=10, b=40, l=50, r=20),
                      yaxis_title="CPI YoY (%)")

    path = os.path.join(IMG_DIR, "chart_2023.png")
    fig.write_image(path, scale=2)
    print(f"  Saved {path}")
    return path


def make_2024_chart(d):
    """2024 OOS 차트."""
    panel = d["panel"]
    cpi = panel["CPI_KR_YoY"].resample("MS").last().dropna()
    cpi_show = cpi.loc["2021-01":"2025-12"]
    cpi_before = cpi_show.loc[:"2023-12"]
    cpi_fc = cpi_show.loc["2024-01":"2024-12"]
    cpi_after = cpi_show.loc["2025-01":]

    fc_dates = pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 13)])
    f24 = d["f24"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=cpi_before.index, y=cpi_before.values,
                             mode="lines", line=dict(color="#000000", width=3), name="Actual CPI"))
    _fc_x = [cpi_before.index[-1]] + list(cpi_fc.index)
    _fc_y = [cpi_before.values[-1]] + list(cpi_fc.values)
    fig.add_trace(go.Scatter(x=_fc_x, y=_fc_y, mode="lines+markers",
                             line=dict(color="#000000", width=4),
                             marker=dict(size=10, symbol="square", color="#D62728",
                                         line=dict(width=2, color="#000000")),
                             name="Actual (2024)"))
    if len(cpi_after) > 0:
        _af_x = [cpi_fc.index[-1]] + list(cpi_after.index)
        _af_y = [cpi_fc.values[-1]] + list(cpi_after.values)
        fig.add_trace(go.Scatter(x=_af_x, y=_af_y, mode="lines",
                                 line=dict(color="#000000", width=3), showlegend=False))

    # Models
    if "lstm_med" in f24:
        fig.add_trace(go.Scatter(x=fc_dates, y=f24["lstm_med"],
                                 mode="lines+markers", line=dict(color="#2196F3", width=2.5),
                                 marker=dict(size=7), name=f"LSTM ({float(f24['lstm_rmse']):.4f})"))
    if "tfm_med" in f24:
        fig.add_trace(go.Scatter(x=fc_dates, y=f24["tfm_med"],
                                 mode="lines+markers", line=dict(color="#E91E63", width=3),
                                 marker=dict(size=9, symbol="star"), name=f"TFM Ours ({float(f24['tfm_rmse']):.4f})"))
    if "bistro_med" in f24:
        fig.add_trace(go.Scatter(x=fc_dates, y=f24["bistro_med"],
                                 mode="lines+markers", line=dict(color="#FF9800", width=3, dash="dot"),
                                 marker=dict(size=8, symbol="diamond"), name=f"BISTRO ({float(f24['bistro_rmse']):.4f})"))
    if "chronos2_med" in f24:
        fig.add_trace(go.Scatter(x=fc_dates, y=f24["chronos2_med"],
                                 mode="lines+markers", line=dict(color="#4CAF50", width=3),
                                 marker=dict(size=9, symbol="hexagram"), name=f"Chronos-2 ({float(f24['chronos2_rmse']):.4f})"))

    fig.add_vline(x=fc_dates[0].timestamp()*1000, line_dash="dash", line_color="rgba(100,100,100,0.6)")
    fig.update_layout(width=1100, height=480, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, font=dict(size=11)),
                      margin=dict(t=10, b=40, l=50, r=20),
                      yaxis_title="CPI YoY (%)")

    path = os.path.join(IMG_DIR, "chart_2024.png")
    fig.write_image(path, scale=2)
    print(f"  Saved {path}")
    return path


def make_seed_chart(d):
    """LSTM vs Transformer seed 비교 차트."""
    lt = d["lt"]
    seeds = [42, 7, 123, 0, 99, 2024, 314, 55, 77, 1]
    labels = [f"s={s}" for s in seeds]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=lt["tfm_rmses_23"], name="Transformer",
                         marker_color="#E91E63", text=[f"{r:.3f}" for r in lt["tfm_rmses_23"]],
                         textposition="outside", textfont=dict(size=9)))
    fig.add_trace(go.Bar(x=labels, y=lt["lstm_rmses_23"], name="LSTM",
                         marker_color="#2196F3", text=[f"{r:.3f}" for r in lt["lstm_rmses_23"]],
                         textposition="outside", textfont=dict(size=9)))
    fig.add_hline(y=1.161, line_dash="dash", line_color="#FF9800",
                  annotation_text="BISTRO (1.161)", annotation_position="top right")
    fig.update_layout(barmode="group", width=1100, height=400, template="plotly_white",
                      yaxis_title="RMSE (pp)", yaxis=dict(range=[0, 1.7]),
                      margin=dict(t=10, b=40, l=50, r=20))

    path = os.path.join(IMG_DIR, "chart_seeds.png")
    fig.write_image(path, scale=2)
    print(f"  Saved {path}")
    return path


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, "Task-Specific vs Foundation Model: CPI Forecasting Benchmark", align="C", ln=True)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(240, 240, 240)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, align="C", fill=True)
        self.ln()

    def table_row(self, cols, widths, bold=False):
        self.set_font("Helvetica", "B" if bold else "", 8)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, align="C")
        self.ln()


def generate():
    print("Loading data...")
    d = load_all()
    actual_23 = d["lstm"]["forecast_actual"]
    lt = d["lt"]

    # Compute RMSEs
    lstm_23 = np.sqrt(np.mean((d["lstm"]["forecast_med"] - actual_23)**2))
    tfm_23 = float(d["tfm23"]["rmse"])
    bistro_23 = np.sqrt(np.mean((d["bistro"]["forecast_med"] - actual_23)**2)) if "bistro" in d else None
    c2_23 = float(d["chronos2_result"]["rmse"]) if "chronos2_result" in d else None
    ar1_23 = np.sqrt(np.mean((d["lstm"]["forecast_ar1"] - actual_23)**2))

    tfm_24 = float(d["f24"]["tfm_rmse"]) if "tfm_rmse" in d["f24"] else None
    lstm_24 = float(d["f24"]["lstm_rmse"]) if "lstm_rmse" in d["f24"] else None
    bistro_24 = float(d["f24"]["bistro_rmse"]) if "bistro_rmse" in d["f24"] else None
    c2_24 = float(d["f24"]["chronos2_rmse"]) if "chronos2_rmse" in d["f24"] else None

    ct5_23 = float(d["chronos_result"]["rmse"]) if "chronos_result" in d else None
    tfm_g_23 = float(d["foundation_model_results"]["timesfm_rmse"]) if "foundation_model_results" in d else None
    bolt_23 = float(d["chronos2_bolt_result"]["rmse"]) if "chronos2_bolt_result" in d else None

    # Generate charts
    print("Generating charts...")
    img_2023 = make_2023_chart(d)
    img_2024 = make_2024_chart(d)
    img_seeds = make_seed_chart(d)

    # PDF
    print("Building PDF...")
    pdf = Report()
    pdf.alias_nb_pages()

    # ── Page 1: Title + Key Findings ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "CPI Forecasting Benchmark", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Task-Specific vs Foundation Model  |  Korean CPI YoY  |  2023-2024 OOS", ln=True, align="C")
    pdf.cell(0, 6, "Report Date: 2026-04-03  |  v1", ln=True, align="C")
    pdf.ln(10)

    pdf.section_title("Executive Summary")
    pdf.body_text(
        "170K parameter task-specific Transformer achieves RMSE 0.5562pp (2023) and 0.5564pp (2024) "
        "on Korean CPI YoY forecasting, outperforming Foundation Models with 91M-200M parameters. "
        "Covariate-informed models (avg 0.796pp) dominate univariate models (avg 1.400pp). "
        "Chronos-2 (120M) achieves best single-year RMSE (0.5362, 2023) but collapses in 2024 (1.3270). "
        "Task-specific Transformer is the most consistent performer across both years."
    )

    # Ranking table
    pdf.section_title("Final Ranking (2023 + 2024 Average RMSE)")
    w = [10, 45, 22, 22, 20, 27, 27]
    pdf.table_header(["#", "Model", "2023", "2024", "Params", "Type", "Covariates"], w)
    rows = [
        ("1", "Transformer (Ours)", f"{tfm_23:.4f}", f"{tfm_24:.4f}" if tfm_24 else "-", "170K", "Task-spec", "18var"),
        ("2", "Chronos-2 (+cov)", f"{c2_23:.4f}" if c2_23 else "-", f"{c2_24:.4f}" if c2_24 else "-", "120M", "Zero-shot", "18var"),
        ("3", "LSTM (Ours)", f"{lstm_23:.4f}", f"{lstm_24:.4f}" if lstm_24 else "-", "327K", "Task-spec", "18var"),
        ("4", "BISTRO", f"{bistro_23:.4f}" if bistro_23 else "-", f"{bistro_24:.4f}" if bistro_24 else "-", "91M", "Pre-train", "18var"),
        ("5", "Chronos-T5", f"{ct5_23:.4f}" if ct5_23 else "-", "-", "200M", "Zero-shot", "univar"),
        ("6", "TimesFM", f"{tfm_g_23:.4f}" if tfm_g_23 else "-", "-", "200M", "Zero-shot", "univar"),
        ("7", "Chronos-Bolt", f"{bolt_23:.4f}" if bolt_23 else "-", "-", "205M", "Zero-shot", "univar"),
        ("8", "AR(1)", f"{ar1_23:.4f}", "-", "-", "Stat", "-"),
    ]
    for i, row in enumerate(rows):
        pdf.table_row(row, w, bold=(i == 0))

    # ── Page 2: 2023 Forecast Chart ──
    pdf.add_page()
    pdf.section_title("2023 Out-of-Sample Forecast")
    pdf.image(img_2023, x=10, w=190)
    pdf.ln(5)
    pdf.body_text(
        "Context: 2020-01 ~ 2022-12 (36 months). All covariate models track the 2023 disinflation "
        f"(4.98% -> 3.18%). Transformer (Ours) RMSE {tfm_23:.4f}pp achieves best consistency. "
        f"Chronos-2 RMSE {c2_23:.4f}pp is slightly better but from a 120M zero-shot model. "
        "Univariate models (Chronos-T5, TimesFM, Bolt) fail to capture the trend, clustering around 4.6-4.9%."
    )

    # Monthly table
    pdf.section_title("2023 Monthly Detail")
    dates_23 = [f"2023-{m:02d}" for m in range(1, 13)]
    mw = [18, 16, 16, 16, 16, 16]
    pdf.table_header(["Date", "Actual", "TFM", "LSTM", "BISTRO", "C2"], mw)
    for i in range(12):
        pdf.table_row([
            dates_23[i],
            f"{actual_23[i]:.2f}",
            f"{d['tfm23']['forecast_med'][i]:.2f}",
            f"{d['lstm']['forecast_med'][i]:.2f}",
            f"{d['bistro']['forecast_med'][i]:.2f}" if "bistro" in d else "-",
            f"{d['chronos2_result']['forecast_med'][i]:.2f}" if "chronos2_result" in d else "-",
        ], mw)

    # ── Page 3: 2024 Forecast ──
    pdf.add_page()
    pdf.section_title("2024 Out-of-Sample Forecast")
    pdf.image(img_2024, x=10, w=190)
    pdf.ln(5)
    pdf.body_text(
        "Context: 2021-01 ~ 2023-12 (36 months, includes 2023 actuals). "
        f"Transformer (Ours) RMSE {tfm_24:.4f}pp maintains consistency from 2023. "
        f"Chronos-2 collapses to RMSE {c2_24:.4f}pp - zero-shot model inconsistency. "
        f"BISTRO improves to {bistro_24:.4f}pp in the post-inflation stabilization period."
    )

    # 2023 vs 2024 comparison
    pdf.section_title("2023 vs 2024 Consistency")
    cw = [45, 25, 25, 30]
    pdf.table_header(["Model", "2023 RMSE", "2024 RMSE", "Consistency"], cw)
    pdf.table_row(["Transformer (170K)", f"{tfm_23:.4f}", f"{tfm_24:.4f}", "Stable"], cw, bold=True)
    pdf.table_row(["LSTM (327K)", f"{lstm_23:.4f}", f"{lstm_24:.4f}" if lstm_24 else "-", "Stable"], cw)
    pdf.table_row(["BISTRO (91M)", f"{bistro_23:.4f}" if bistro_23 else "-", f"{bistro_24:.4f}" if bistro_24 else "-", "Improved"], cw)
    pdf.table_row(["Chronos-2 (120M)", f"{c2_23:.4f}" if c2_23 else "-", f"{c2_24:.4f}" if c2_24 else "-", "Collapsed"], cw)

    # ── Page 4: LSTM vs Transformer ──
    pdf.add_page()
    pdf.section_title("LSTM vs Transformer (Same Conditions)")
    pdf.body_text(
        "Same architecture (Variable Fusion + Temporal Decoder), only encoder replaced. "
        "Same data (18 covariates), seq_len (36), training method. "
        "Transformer wins 10/10 seeds in 2023, 27% lower mean RMSE, 22% lower in 2024."
    )
    pdf.image(img_seeds, x=10, w=190)
    pdf.ln(5)

    # Stats table
    pdf.section_title("Statistical Comparison")
    sw = [45, 30, 30, 25]
    pdf.table_header(["Metric", "LSTM", "Transformer", "Winner"], sw)
    pdf.table_row(["Parameters", "326,850", "170,690", "TFM"], sw)
    pdf.table_row(["2023 Mean RMSE", f"{np.mean(lt['lstm_rmses_23']):.4f}", f"{np.mean(lt['tfm_rmses_23']):.4f}", "TFM"], sw, bold=True)
    pdf.table_row(["2023 Best RMSE", f"{np.min(lt['lstm_rmses_23']):.4f}", f"{np.min(lt['tfm_rmses_23']):.4f}", "TFM"], sw)
    pdf.table_row(["2023 Std", f"{np.std(lt['lstm_rmses_23']):.4f}", f"{np.std(lt['tfm_rmses_23']):.4f}", "TFM"], sw)
    pdf.table_row(["2024 RMSE", f"{float(lt['lstm_rmse_24']):.4f}", f"{float(lt['tfm_rmse_24']):.4f}", "TFM"], sw, bold=True)

    # ── Page 5: Key Conclusions ──
    pdf.add_page()
    pdf.section_title("Key Conclusions")

    conclusions = [
        ("1. Task-specific Transformer is the most consistent model",
         f"170K params achieves RMSE 0.5562 (2023) and 0.5564 (2024). "
         "Foundation Models with 91M-200M parameters cannot match this consistency."),

        ("2. Covariates are decisive",
         "Covariate models average RMSE 0.796pp vs univariate models 1.400pp. "
         "Macroeconomic forecasting requires related variable information."),

        ("3. Chronos-2 is best single-year but inconsistent",
         "Zero-shot with covariates achieves 0.5362 in 2023 but collapses to 1.3270 in 2024. "
         "Pre-trained knowledge helps but doesn't guarantee year-over-year stability."),

        ("4. Transformer beats LSTM under identical conditions",
         "10/10 seed wins, 27% lower RMSE in 2023, 22% in 2024. "
         "Self-attention captures global dependencies more effectively than recurrent processing."),

        ("5. Shorter context (36mo) is optimal for task-specific models",
         "seq_len=36 consistently outperforms 60 and 120 months. "
         "However, Foundation Models (BISTRO) benefit from longer context - opposite pattern."),

        ("6. Variable selection matters more than architecture",
         "18 selected covariates vs 29 raw variables: RMSE improved from 1.13 to 0.55. "
         "BISTRO attention for screening + Transformer attention for refinement = hybrid optimal."),

        ("7. Model scaling is unnecessary",
         "170K params is already near the capacity ceiling for 276 monthly observations. "
         "Seed ensemble and rolling-window retraining are more effective than parameter scaling."),
    ]

    for title, text in conclusions:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5, text)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(2)

    # ── Page 6: Architecture ──
    pdf.add_page()
    pdf.section_title("Model Architecture")

    pdf.set_font("Courier", "", 8)
    pdf.multi_cell(0, 4,
        "AttentionTransformerForecaster (170,690 params)\n"
        "+-- Variable Embedding      19 x Linear(1 -> 64)\n"
        "+-- Variable Fusion Attn    4-head MHA, 19 vars per timestep\n"
        "+-- Positional Encoding     Sinusoidal (max_len=512)\n"
        "+-- Transformer Encoder     2 layers, d_model=64, ff=128, causal mask\n"
        "+-- Projection              Linear(64 -> 128)\n"
        "+-- Temporal Attn Decoder   12 learnable queries, 4 heads\n"
        "+-- Output Head             Gaussian NLL (mu + log_sigma)")
    pdf.ln(5)

    pdf.section_title("Training Configuration")
    pdf.set_font("Helvetica", "", 9)
    config_text = (
        "Data: 18 macroeconomic covariates + CPI target (monthly, 2003-2025)\n"
        "Context: 36 months | Prediction: 12 months (direct multi-step)\n"
        "Normalization: Per-variable Z-score (train set only)\n"
        "Optimizer: AdamW (lr=5e-4, weight_decay=1e-4) + CosineAnnealingLR\n"
        "Loss: Gaussian NLL | Gradient clipping: max_norm=1.0\n"
        "Early stopping: patience=20 on validation RMSE\n"
        "Walk-Forward CV: 5-fold expanding window (val: 2018-2022)\n"
        "Uncertainty: 200 samples from N(mu, sigma^2) -> 90% CI"
    )
    pdf.multi_cell(0, 5, config_text)
    pdf.ln(5)

    pdf.section_title("Covariates (18 variables from BISTRO tournament)")
    pdf.set_font("Helvetica", "", 8)
    covariates = (
        "AUD_USD, CN_Interbank3M, US_UnempRate, JP_REER, JP_Interbank3M, JP_CoreCPI, "
        "KC_FSI, KR_MfgProd, Pork, US_NFP, US_TradeTransEmp, THB_USD, "
        "PPI_CopperNickel, CN_PPI, US_Mortgage15Y, UK_10Y_Bond, US_ExportPI, US_DepInstCredit"
    )
    pdf.multi_cell(0, 4, covariates)

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "CPI_Forecasting_Benchmark_v1.pdf")
    pdf.output(output_path)
    print(f"\nReport saved: {output_path}")
    return output_path


if __name__ == "__main__":
    generate()
