"""
PDF Report Generator for BISTRO-LSTM
=====================================
Usage:
    .venv/bin/python3 export_pdf.py
"""

import os
import numpy as np
from fpdf import FPDF

from lstm_core import (
    results_available, load_inference_results, load_ablation_results,
)
from comparison import load_bistro_results


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class LSTMReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "BISTRO-LSTM: LSTM vs Transformer for CPI Forecasting", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report(output_path: str = None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "BISTRO_LSTM_Report.pdf")

    pdf = LSTMReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 15, "BISTRO-LSTM Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "This report compares the performance of an Attention-Augmented Stacked LSTM (~2M parameters) "
        "against the BISTRO Transformer foundation model (91M parameters) for Korean CPI Year-over-Year "
        "forecasting. Both models are evaluated on the same 2023 out-of-sample period using identical "
        "macroeconomic covariates."
    )
    pdf.ln(5)

    # Results
    if results_available("lstm_inference_results.npz"):
        lstm = load_inference_results()
        actual = lstm["forecast_actual"]
        pred = lstm["forecast_med"]
        ar1 = lstm["forecast_ar1"]

        l_rmse = np.sqrt(np.mean((pred - actual) ** 2))
        l_mae = np.mean(np.abs(pred - actual))
        ar1_rmse = np.sqrt(np.mean((ar1 - actual) ** 2))

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "2. Forecast Performance (2023 OOS)", ln=True)
        pdf.set_font("Helvetica", "", 10)

        pdf.cell(60, 7, "Model", border=1, align="C")
        pdf.cell(40, 7, "RMSE (pp)", border=1, align="C")
        pdf.cell(40, 7, "MAE (pp)", border=1, align="C")
        pdf.cell(40, 7, "Parameters", border=1, align="C")
        pdf.ln()

        pdf.cell(60, 7, "LSTM (Ours)", border=1)
        pdf.cell(40, 7, f"{l_rmse:.4f}", border=1, align="C")
        pdf.cell(40, 7, f"{l_mae:.4f}", border=1, align="C")
        pdf.cell(40, 7, "~2M", border=1, align="C")
        pdf.ln()

        bistro = load_bistro_results()
        if bistro is not None:
            b_rmse = np.sqrt(np.mean((bistro["forecast_med"] - actual) ** 2))
            b_mae = np.mean(np.abs(bistro["forecast_med"] - actual))

            pdf.cell(60, 7, "BISTRO (Transformer)", border=1)
            pdf.cell(40, 7, f"{b_rmse:.4f}", border=1, align="C")
            pdf.cell(40, 7, f"{b_mae:.4f}", border=1, align="C")
            pdf.cell(40, 7, "91M", border=1, align="C")
            pdf.ln()

        pdf.cell(60, 7, "AR(1) Baseline", border=1)
        pdf.cell(40, 7, f"{ar1_rmse:.4f}", border=1, align="C")
        pdf.cell(40, 7, "-", border=1, align="C")
        pdf.cell(40, 7, "-", border=1, align="C")
        pdf.ln(10)

        # Monthly forecasts
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "3. Monthly Forecast Detail", ln=True)
        pdf.set_font("Helvetica", "", 9)

        pdf.cell(30, 6, "Date", border=1, align="C")
        pdf.cell(25, 6, "Actual", border=1, align="C")
        pdf.cell(25, 6, "LSTM", border=1, align="C")
        pdf.cell(25, 6, "Error", border=1, align="C")
        pdf.cell(25, 6, "AR(1)", border=1, align="C")
        if bistro is not None:
            pdf.cell(25, 6, "BISTRO", border=1, align="C")
        pdf.ln()

        for i, date in enumerate(lstm["forecast_date"]):
            pdf.cell(30, 6, date, border=1)
            pdf.cell(25, 6, f"{actual[i]:.2f}", border=1, align="C")
            pdf.cell(25, 6, f"{pred[i]:.2f}", border=1, align="C")
            pdf.cell(25, 6, f"{pred[i]-actual[i]:+.2f}", border=1, align="C")
            pdf.cell(25, 6, f"{ar1[i]:.2f}", border=1, align="C")
            if bistro is not None:
                pdf.cell(25, 6, f"{bistro['forecast_med'][i]:.2f}", border=1, align="C")
            pdf.ln()

    # Ablation
    if results_available("lstm_ablation_results.npz"):
        abl = load_ablation_results()
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "4. Ablation Study", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Baseline RMSE: {abl['baseline_rmse']:.4f}pp", ln=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 9)
        pdf.cell(50, 6, "Variable", border=1, align="C")
        pdf.cell(30, 6, "RMSE", border=1, align="C")
        pdf.cell(30, 6, "Delta RMSE", border=1, align="C")
        pdf.cell(30, 6, "Impact", border=1, align="C")
        pdf.ln()

        sorted_idx = np.argsort(-abl["abl_delta_rmse"])
        for idx in sorted_idx:
            var = abl["abl_vars"][idx]
            delta = abl["abl_delta_rmse"][idx]
            impact = "Helps" if delta > 0 else "Hurts"
            pdf.cell(50, 6, var, border=1)
            pdf.cell(30, 6, f"{abl['abl_rmse'][idx]:.4f}", border=1, align="C")
            pdf.cell(30, 6, f"{delta:+.4f}", border=1, align="C")
            pdf.cell(30, 6, impact, border=1, align="C")
            pdf.ln()

    pdf.output(output_path)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
