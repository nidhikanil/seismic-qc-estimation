"""
Seismic Qc Estimation Workflow

Python workflow for band-wise seismic Qc estimation using
ambient noise cross-correlation and autocorrelation methods.

This repository contains a generalized educational/research workflow.
No unpublished datasets or manuscript figures are included.
"""

import os
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, hilbert

warnings.filterwarnings("ignore")


# Configuration

YEARS = [2015, 2025]

DATA_DIRS = {
    2015: "./data/c1/2015/",
    2025: "./data/c1/2025/",
}

C3_DIR = "./data/c3/"
OUTPUT_DIR = "./output/"

SAVE_PLOTS = True

FS = 5.0

BANDS = {
    "0.25-0.5s": (1/0.5, 1/0.25),
    "0.5-1s":    (1/1.0, 1/0.5),
    "1-2.5s":    (1/2.5, 1/1.0),
    "2.5-5s":    (1/5.0, 1/2.5),
    "5-10s":     (1/10.0, 1/5.0),
    "10-20s":    (1/20.0, 1/10.0),
}

QC_GRID = np.logspace(1, 3.3, 250)

ALPHA = 1.0
VEL_MIN = 1.5
VEL_MAX = 5.0
LATE_FRAC = 0.25
QC_STD_MAX = 150.0


def load_npy(fpath):
    raw = np.load(fpath, allow_pickle=True)

    if raw.dtype == object:
        item = raw.item()

        if isinstance(item, dict):
            for k in ["data", "corr", "signal"]:
                if k in item:
                    return np.asarray(item[k], dtype=float).ravel()

        return np.asarray(item, dtype=float).ravel()

    return raw.ravel().astype(float)


def bandpass(data, flo, fhi, order=4):

    nyq = FS / 2.0

    b, a = butter(
        order,
        [max(flo, 1e-4) / nyq, min(fhi, nyq * 0.99) / nyq],
        btype="band"
    )

    return filtfilt(b, a, data)


def energy_env(sig, smooth_pts=None):

    env = (np.abs(hilbert(sig)) ** 2) / 2.0

    if smooth_pts and smooth_pts > 1:
        env = np.convolve(
            env,
            np.ones(smooth_pts) / smooth_pts,
            mode="same"
        )

    return env


def coda_log_norm(env):

    n = len(env)

    E_late = np.mean(env[int(n * (1 - LATE_FRAC)):]) + 1e-40

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(env / E_late)


def pick_rg(half_bp, t_half, dist_km):

    t_lo = dist_km / VEL_MAX
    t_hi = dist_km / VEL_MIN

    i_lo = max(int(np.searchsorted(t_half, t_lo)), 1)
    i_hi = min(int(np.searchsorted(t_half, t_hi)), len(half_bp) - 1)

    if i_lo >= i_hi:
        return None

    env = np.abs(hilbert(half_bp))

    return float(
        t_half[i_lo + int(np.argmax(env[i_lo:i_hi]))]
    )


def fit_qc(log_norm, t_half, t0, t1, fc):

    mask = (
        (t_half >= t0) &
        (t_half <= t1) &
        (t_half > 0)
    )

    if mask.sum() < 5:
        return None

    t = t_half[mask]

    obs = log_norm[mask] + ALPHA * np.log10(t)

    misfit = np.empty(len(QC_GRID))

    for i, Qc in enumerate(QC_GRID):

        model = -2 * np.pi * fc * t / Qc

        C = np.mean(obs - model)

        r = obs - (C + model)

        misfit[i] = float(np.dot(r, r))

    grad = np.diff(misfit)

    sc = np.where(np.diff(np.sign(grad)))[0]

    if len(sc) == 0:
        return None

    return float(QC_GRID[sc[0] + 1])


def estimate_qc(causal, t_half, dist_km, flo, fhi):

    T1 = 1.0 / fhi
    T2 = 1.0 / flo

    T_peak = 0.5 * (T1 + T2)

    fc = 1.0 / T_peak

    try:
        bp = bandpass(causal, flo, fhi)

    except Exception:
        return None

    Rg_t = pick_rg(bp, t_half, dist_km)

    if Rg_t is None:
        return None

    smooth_pts = max(1, int(T_peak * FS / 4))

    env = energy_env(bp, smooth_pts)

    log_nm = coda_log_norm(env)

    t_start = Rg_t + 6 * T_peak
    t_end = t_start + 20 * T_peak

    q = fit_qc(log_nm, t_half, t_start, t_end, fc)

    return q


def process_file(fpath):

    try:
        stacked = load_npy(fpath)

    except Exception:
        return None

    zero_idx = (len(stacked) - 1) // 2

    causal = stacked[zero_idx:]

    t_half = np.arange(len(causal)) / FS

    results = []

    for band_name, (flo, fhi) in BANDS.items():

        qc = estimate_qc(
            causal,
            t_half,
            dist_km=20,
            flo=flo,
            fhi=fhi
        )

        results.append((band_name, qc))

    return results


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = glob.glob("./data/*.npy")

    for fpath in files:

        print(f"Processing: {os.path.basename(fpath)}")

        results = process_file(fpath)

        if results is None:
            continue

        for band, qc in results:

            print(f"{band}  |  Qc = {qc}")


if __name__ == "__main__":
    main()