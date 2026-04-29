"""Fetch BoK CSI (소비자심리지수) from ECOS API and merge into augmented panel."""
import os, requests
import pandas as pd

API_KEY = os.environ.get("ECOS_API_KEY", "Z3WLWOJ8X7GY0M3AJ0IE")

# 511Y002 = 소비자동향조사(전국, 월), FME = 소비자심리지수, 99988 = 전체
url = (
    f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/200/"
    f"511Y002/M/202001/202512/FME/99988"
)
r = requests.get(url, timeout=30)
data = r.json()["StatisticSearch"]["row"]

records = [{"date": pd.Period(d["TIME"][:4] + "-" + d["TIME"][4:6], "M"),
            "BoK_CSI": float(d["DATA_VALUE"])} for d in data]
csi = pd.DataFrame(records).set_index("date").sort_index()
print(f"CSI fetched: {csi.shape}, range {csi.index[0]}~{csi.index[-1]}")
print(csi.tail(12))

# Merge into augmented panel
aug = pd.read_csv("data/macro_panel_aug.csv", index_col=0)
aug.index = pd.PeriodIndex(aug.index, freq="M")
full = aug.join(csi, how="left")
# Pre-2020 fill (out of forecast context)
full["BoK_CSI"] = full["BoK_CSI"].fillna(100.0)  # neutral
full.to_csv("data/macro_panel_full.csv")
print(f"\nFull panel shape: {full.shape}")
print(f"Columns: {list(full.columns)}")
print(f"\nLast 6 rows (BoK_CSI added):")
print(full.tail(6)[["CPI_KR_YoY", "GT_금리", "GT_물가", "BoK_CSI"]].round(2))
