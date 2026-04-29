"""Merge optimal18 macro panel + Google Trends Korean keywords."""
import pandas as pd

base = pd.read_csv("data/macro_panel_optimal18.csv", index_col=0)
base.index = pd.PeriodIndex(base.index, freq="M")

trends = pd.read_csv("data/google_trends_kr.csv", index_col=0)
trends.index = pd.PeriodIndex(trends.index, freq="M")

aug = base.join(trends, how="left")
print(f"Base shape: {base.shape}")
print(f"Aug shape: {aug.shape}")
print(f"Aug columns: {list(aug.columns)}")
print(f"\nMissing trends rows (early period before 2020):")
print(aug[aug["GT_금리"].isna()].index.tolist()[:5], "...")
print(f"\nLast 6 rows:")
print(aug.tail(6).round(2))

# For pre-2020 rows where trends are NaN, fill with sentinel value (0 or interpolation)
# Since we forecast 2025-05 onwards, context starts 2022-04 — all in trends range → no NaN issue
ctx_start_earliest = pd.Period("2022-05", "M")  # for 2025-04 origin with seq=36
trends_in_ctx = aug.loc[ctx_start_earliest:, ["GT_금리", "GT_물가", "GT_인플레이션", "GT_한국은행", "GT_디플레이션"]]
print(f"\nNaN check in forecast context (2022-05 onwards): {trends_in_ctx.isna().sum().sum()}")

# Fill pre-2020 with 0 (won't affect forecast since out of context)
aug = aug.fillna(0)
aug.to_csv("data/macro_panel_aug.csv")
print(f"\nSaved data/macro_panel_aug.csv")
