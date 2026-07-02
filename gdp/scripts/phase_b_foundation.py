"""Phase B — Foundation(Chronos/Moirai) zero-shot, 분기 flash 외삽 baseline.
한계: 외삽형이라 동시 월별 지표(nowcasting의 핵심)를 못 씀 → DFM 대비 열위 예상.
      target quarter별 1개 예측을 해당 분기 grid(week_idx)에 broadcast.
사용: --backend chronos (.venv) | --backend moirai (.venv-moirai)
"""
import sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import phase_b_harness as H

MIN_CTX = 6  # 최소 분기 컨텍스트

def chronos_forecast(series):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                               device_map="cpu", torch_dtype=torch.float32)
    def f(ctx_vals):
        q, mean = pipe.predict_quantiles(
            context=torch.tensor(np.asarray(ctx_vals, dtype=np.float32)),
            prediction_length=1, quantile_levels=[0.5])
        return float(q[0, 0, 0])  # median
    return f

def moirai_forecast(series):
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.pandas import PandasDataset
    def f(ctx_vals):
        n = len(ctx_vals)
        module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
        model = MoiraiForecast(module=module, prediction_length=1, context_length=n,
                               patch_size="auto", num_samples=100, target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        pred = model.create_predictor(batch_size=1, device="cpu")
        idx = pd.period_range("2000Q1", periods=n, freq="Q").to_timestamp()
        df = pd.DataFrame({"y": np.asarray(ctx_vals, np.float32)}, index=idx)
        ds = PandasDataset(df, target="y", freq="Q")
        fc = list(pred.predict(ds))[0]
        return float(np.median(fc.samples[:, 0]))
    return f

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", required=True)
    a = ap.parse_args()
    grid, _ = H.load_grid()
    qflash = (grid[["tq", "flash"]].drop_duplicates()
              .assign(p=lambda d: d.tq.map(lambda x: pd.Period(x, "Q")))
              .sort_values("p").reset_index(drop=True))
    qflash = qflash[qflash.flash.notna()]
    fcast = {"chronos": chronos_forecast, "moirai": moirai_forecast}[a.backend](None)
    name = f"our_{a.backend}"
    qpred = {}
    vals = qflash.flash.tolist(); qs = qflash.tq.tolist()
    for i, q in enumerate(qs):
        if i < MIN_CTX:
            continue
        ctx = vals[:i]  # 과거 분기 flash만 (외삽)
        try:
            qpred[q] = fcast(ctx)
        except Exception as e:
            print(f"  {q} 예측 실패: {e}")
    # broadcast to grid
    rows = []
    for r in grid.itertuples(index=False):
        rows.append({"tq": r.tq, "vintage": r.vintage, "week_idx": r.week_idx,
                     "flash": r.flash, "model_name": name,
                     "y_pred": qpred.get(r.tq, np.nan)})
    pred = pd.DataFrame(rows)
    out = f"output/csv/_phase_b_{a.backend}.csv"
    pred.to_csv(out, index=False)
    n = pred.y_pred.notna().sum()
    print(f"{name}: {n}/{len(pred)} valid (분기 {len(qpred)}개 예측) -> {out}")
    if n:
        print(H.score(pred).to_string())
    print(f"PHASE_B_{a.backend.upper()}_DONE")

if __name__ == "__main__":
    main()
