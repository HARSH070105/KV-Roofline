#!/usr/bin/env python3
"""Week-6 validation: compare the roofline model with measurements, report MAPE, and fit calibration.

  python scripts/validate.py --meas data/measurements.csv                 # MAPE of predicted TPOT
  python scripts/validate.py --meas data/measurements.csv --fit --calib-out data/calib.json
  python scripts/validate.py --meas data/measurements.csv --calib data/calib.json
  python scripts/validate.py --hbm data/hbm_bytes.csv                      # MAPE of predicted HBM bytes/step

Fit: 3 scalars minimise the squared log-ratio between predicted and measured TPOT:
  bw_eff (achievable fraction of peak HBM BW), tc_eff (tensor-core efficiency), t_fixed_us (per-step overhead).
With few points, tc_eff is unidentifiable (decode is memory-bound), so by default only bw_eff and t_fixed are fitted
(--fit-tc to include it). Fit on a TRAIN subset and report MAPE on held-out points (--holdout k) so the
number you quote (deck target: 15% or better) is not in-sample.
Reported error is |pred - meas| / meas. Roof attainment error compares pct_of_roof with achieved/attainable.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from kvroofline import Compression, evaluate, get_hw, get_model  # noqa: E402
from kvroofline.engine import Calibration  # noqa: E402


def predict_row(r, calib):
    hw = get_hw(r["hw"], n_chips=int(r.get("n_chips", 1) or 1))
    bits = int(r.get("kv_bits", 16) or 16)
    comp = Compression(kv_bits=bits, group=int(r.get("group", 0) or 0)) if bits < 16 else Compression()
    return evaluate(get_model(r["model"]), hw, int(r["S"]), int(r["B"]), comp, calib=calib)


def predicted_tpot(df, calib):
    return np.array([predict_row(r, calib)["tpot_ms"] for _, r in df.iterrows()])


def mape(pred, meas):
    return float(np.mean(np.abs(pred - meas) / meas) * 100.0)


def fit(df, fit_tc=False):
    meas = df["tpot_ms"].to_numpy(float)

    def unpack(x):
        return Calibration(bw_eff=x[0], tc_eff=x[2] if fit_tc else 1.0, t_fixed_us=x[1])

    def resid(x):
        return np.log(predicted_tpot(df, unpack(x))) - np.log(meas)

    x0 = [0.8, 100.0, 0.6]
    lo, hi = [0.2, 0.0, 0.05], [1.0, 20000.0, 1.0]
    res = least_squares(resid, x0, bounds=(lo, hi), x_scale=[1.0, 100.0, 1.0])
    return unpack(res.x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meas")
    ap.add_argument("--hbm")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fit-tc", action="store_true")
    ap.add_argument("--holdout", type=int, default=0, help="hold out every k-th point from the fit (0 = none)")
    ap.add_argument("--calib")
    ap.add_argument("--calib-out", default="data/calib.json")
    a = ap.parse_args()

    if a.hbm:
        h = pd.read_csv(a.hbm)
        pred = []
        for _, r in h.iterrows():
            e = evaluate(get_model(r["model"]), get_hw(r["hw"]), int(r["S"]), int(r["B"]))
            pred.append(e["hbm_gb_per_step_chip"])
        h["pred_gb"] = pred
        h["err_pct"] = 100 * (h["pred_gb"] - h["hbm_gb_per_step"]) / h["hbm_gb_per_step"]
        print(h[["model", "S", "B", "hbm_gb_per_step", "pred_gb", "err_pct"]].to_string(index=False))
        print(f"\nMAPE of HBM bytes/step: {h['err_pct'].abs().mean():.1f}%")

    if a.meas:
        df = pd.read_csv(a.meas).dropna(subset=["tpot_ms"]).reset_index(drop=True)
        calib = Calibration()
        if a.calib:
            calib = Calibration(**json.load(open(a.calib)))
        train = df
        test = df.iloc[0:0]
        if a.holdout and len(df) > a.holdout:
            mask = (np.arange(len(df)) % a.holdout) == (a.holdout - 1)
            train, test = df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
        if a.fit:
            calib = fit(train, a.fit_tc)
            os.makedirs(os.path.dirname(a.calib_out) or ".", exist_ok=True)
            json.dump(calib.__dict__, open(a.calib_out, "w"), indent=2)
            print("fitted calibration:", calib, "->", a.calib_out)
        for name, d in (("train/all", train), ("held-out", test)):
            if len(d) == 0:
                continue
            p = predicted_tpot(d, calib)
            out = d.assign(pred_ms=p, err_pct=100 * (p - d["tpot_ms"]) / d["tpot_ms"])
            print(f"\n[{name}]")
            print(out[["model", "S", "B", "kv_bits", "tpot_ms", "pred_ms", "err_pct"]].to_string(index=False))
            print(f"MAPE TPOT: {mape(p, d['tpot_ms'].to_numpy(float)):.1f}%   (deck target: <= 15%)")
        if len(df):
            base = predicted_tpot(df, Calibration())
            print(f"\nUncalibrated (ideal roofline) MAPE on all points: {mape(base, df['tpot_ms'].to_numpy(float)):.1f}%")


if __name__ == "__main__":
    main()
