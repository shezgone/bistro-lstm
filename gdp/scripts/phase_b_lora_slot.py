"""LoRA 부품의 조기 슬롯 장착 — "②(연구)를 ①(운영)에 꽂기" 검증 (2026-08-28, 사용자 아이디어).

질문: FM ICL(zero-shot)의 주차별 평탄 문제를, 조기 6주 슬롯에 'LoRA 적응판'을 넣어 우회하면?
방법: 기존 주차별 결합(early=부품, 이후 XGB)에서 부품만 LoRA 3-seed 앙상블로 교체. 채점만(신규 학습 없음).

결과 (공통 20분기, 2021~2025 — LoRA 커버 창):
  [조기 6주 단독]  GBM 0.583(자격선) | XGB 0.577 | C2f-zs 0.651(이 창에선 자격 미달!)
                   CLo-ens 0.560 | BLo-ens 0.562 | (CLo+BLo)/2 0.546 ← 전부 자격선 통과 + XGB 상회
  [슬롯 결합 — 기준 XGB 단독 0.5178]
    early=(CLo+BLo)/2      0.5080 (-1.9%)  ← 최고
    early=(XGB+CLo+BLo)/3  0.5083
    early=(GBM+CLo+BLo)/3  0.5087
    현행안 (GBM+C2f-zs)/2  0.5259 (이 창에선 기준보다 악화 — zs 부품의 한계)
해석: 2021~2025 창에서 zero-shot 부품은 슬롯 자격을 잃는데(0.651), LoRA 적응판은
  자격을 회복시키고(0.546~0.562) 슬롯 전략 자체를 되살린다 — "평탄 문제의 우회로"로 유효.
유보: ①7개 구성 중 최선 선택(선택편의) ②DM p 0.3~0.6 전부 비유의 ③20분기 창 한정
  (32Q 전체엔 LoRA 예측이 없음 — 2018~2020 확장은 walk-forward 재적응 추가 필요)
  ④운영 시 seed 3개×2모델=6 러너 유지 비용.

실행: gdp-nowcasting-renewal 루트에서 .venv-gdp/bin/python phase_b_lora_slot.py
"""
import sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
import phase_b_harness as H

KEY = ["tq", "vintage", "week_idx"]
def norm(d):
    d = d.copy(); d["vintage"] = pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d

files = sorted(glob.glob("output/csv/all_model_comparison_11_20260123_maxlag00/*/"
                         "all_model_comparison_predictions_with_all_ensembles_*.csv"))
new = pd.concat([pd.read_csv(f, dtype={"tq": str}) for f in files], ignore_index=True)
new["vintage"] = pd.to_datetime(new["vintage"]).dt.strftime("%Y-%m-%d")
def pick(mn, col):
    return new[new.model_name == mn][KEY + ["y_pred"]].rename(columns={"y_pred": col}).drop_duplicates(KEY)

b = pick("xgboost", "xgb").merge(pick("gbm", "gbm"), on=KEY)
b = b.merge(new[new.model_name == "dfm"][KEY + ["flash"]].drop_duplicates(KEY), on=KEY)
srcs = [("_phase_b_our_chronos2f_predictions.csv", "our_chronos2f", "czs"),
        ("_phase_b_our_c2f_lora_predictions.csv", "our_c2f_lora", "c7"),
        ("_phase_b_our_c2f_lora_s11_predictions.csv", "our_c2f_lora_s11", "c11"),
        ("_phase_b_our_c2f_lora_s23_predictions.csv", "our_c2f_lora_s23", "c23"),
        ("_phase_b_our_bistro_lora_predictions.csv", "our_bistro_lora", "b7"),
        ("_phase_b_our_bistro_lora_s11_predictions.csv", "our_bistro_lora_s11", "b11"),
        ("_phase_b_our_bistro_lora_s23_predictions.csv", "our_bistro_lora_s23", "b23")]
for f, mn, c in srcs:
    d = norm(pd.read_csv(f"output/csv/{f}", dtype={"tq": str}))
    b = b.merge(d[d.model_name == mn][KEY + ["y_pred"]].rename(columns={"y_pred": c}), on=KEY, how="inner")
b = b.dropna().reset_index(drop=True)
b = b[(b.week_idx >= -19) & (b.week_idx <= -1)]
b["clo"] = b[["c7", "c11", "c23"]].mean(axis=1)
b["blo"] = b[["b7", "b11", "b23"]].mean(axis=1)
early = (b.week_idx <= -14).values

def sc(v, sub=None):
    d = b if sub is None else b[sub]
    vv = pd.Series(v) if sub is None else pd.Series(v)[sub if isinstance(sub, np.ndarray) else sub.values]
    t = pd.DataFrame({"model_name": "x", "tq": d.tq, "vintage": d.vintage,
                      "week_idx": d.week_idx, "flash": d.flash, "y_pred": vv.values})
    return float(H.score(t).iloc[0])

X, G, C, CL, BL = b.xgb.values, b.gbm.values, b.czs.values, b.clo.values, b.blo.values
print(f"공통 {b.tq.nunique()}분기 — 기준 XGB 단독 {sc(X):.4f}")
for k, v in {"현행안 (GBM+C2f-zs)/2": (G + C) / 2, "(GBM+CLo)/2": (G + CL) / 2,
             "(GBM+BLo)/2": (G + BL) / 2, "(CLo+BLo)/2": (CL + BL) / 2,
             "(GBM+CLo+BLo)/3": (G + CL + BL) / 3}.items():
    print(f"  early={k:24s} → {sc(np.where(early, v, X)):.4f}")
