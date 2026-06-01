from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = ROOT / "analysis" / "outputs"


SURFACE_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def azimuth_deg(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    dx = float(x.iloc[-1] - x.iloc[0])
    dy = float(y.iloc[-1] - y.iloc[0])
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if float(aa.std(ddof=0)) == 0.0 or float(bb.std(ddof=0)) == 0.0:
        return np.nan
    return float(aa.corr(bb))


def robust_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    result = np.full_like(x, np.nan, dtype=float)
    if len(xp) < 2:
        return result

    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]

    # Drop duplicates in xp to keep interpolation stable.
    _, unique_idx = np.unique(xp, return_index=True)
    xp = xp[np.sort(unique_idx)]
    fp = fp[np.sort(unique_idx)]
    if len(xp) < 2:
        return result

    mask = (x >= xp.min()) & (x <= xp.max())
    if mask.any():
        result[mask] = np.interp(x[mask], xp, fp)
    return result


def summarize(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def eval_zone_metrics(df: pd.DataFrame, ps_idx: int) -> dict:
    n_rows = len(df)
    if ps_idx >= n_rows:
        return {
            "pred_len": 0,
            "pred_tvt_min": np.nan,
            "pred_tvt_max": np.nan,
            "pred_tvt_mean": np.nan,
            "pred_tvt_std": np.nan,
            "pred_tvt_abs_diff_mean": np.nan,
            "pred_tvt_abs_diff_p95": np.nan,
            "pred_tvt_abs_diff_max": np.nan,
            "pred_tvt_direction_pos_ratio": np.nan,
            "pred_tvt_direction_neg_ratio": np.nan,
            "pred_tvt_direction_zero_ratio": np.nan,
            "pred_gr_mean": np.nan,
            "pred_gr_std": np.nan,
            "pred_gr_nan_ratio": np.nan,
            "pred_tvt_start": np.nan,
            "pred_tvt_end": np.nan,
            "pred_tvt_net_change": np.nan,
            "pred_tvt_net_slope_per_ft": np.nan,
            "pred_tvt_sign_change_ratio": np.nan,
        }

    pred = df.iloc[ps_idx:].copy()
    tvt = pred["TVT"]
    gr = pred["GR"]
    tvt_diff = tvt.diff().dropna()
    tvt_start = float(tvt.iloc[0]) if tvt.notna().any() else np.nan
    tvt_end = float(tvt.iloc[-1]) if tvt.notna().any() else np.nan
    net_change = float(tvt_end - tvt_start) if np.isfinite(tvt_start) and np.isfinite(tvt_end) else np.nan

    # Sign-change ratio quantifies how often direction flips in local TVT changes.
    if len(tvt_diff) > 2:
        sign = np.sign(tvt_diff.to_numpy())
        sign = sign[sign != 0]
        if len(sign) > 1:
            sign_change_ratio = float(np.mean(sign[1:] != sign[:-1]))
        else:
            sign_change_ratio = np.nan
    else:
        sign_change_ratio = np.nan

    return {
        "pred_len": int(len(pred)),
        "pred_tvt_min": float(tvt.min()) if tvt.notna().any() else np.nan,
        "pred_tvt_max": float(tvt.max()) if tvt.notna().any() else np.nan,
        "pred_tvt_mean": float(tvt.mean()) if tvt.notna().any() else np.nan,
        "pred_tvt_std": float(tvt.std(ddof=1)) if tvt.notna().sum() > 1 else np.nan,
        "pred_tvt_abs_diff_mean": float(tvt_diff.abs().mean()) if len(tvt_diff) else np.nan,
        "pred_tvt_abs_diff_p95": float(tvt_diff.abs().quantile(0.95)) if len(tvt_diff) else np.nan,
        "pred_tvt_abs_diff_max": float(tvt_diff.abs().max()) if len(tvt_diff) else np.nan,
        "pred_tvt_direction_pos_ratio": float((tvt_diff > 0).mean()) if len(tvt_diff) else np.nan,
        "pred_tvt_direction_neg_ratio": float((tvt_diff < 0).mean()) if len(tvt_diff) else np.nan,
        "pred_tvt_direction_zero_ratio": float((tvt_diff == 0).mean()) if len(tvt_diff) else np.nan,
        "pred_gr_mean": float(gr.mean()) if gr.notna().any() else np.nan,
        "pred_gr_std": float(gr.std(ddof=1)) if gr.notna().sum() > 1 else np.nan,
        "pred_gr_nan_ratio": float(gr.isna().mean()),
        "pred_tvt_start": tvt_start,
        "pred_tvt_end": tvt_end,
        "pred_tvt_net_change": net_change,
        "pred_tvt_net_slope_per_ft": float(net_change / max(len(pred) - 1, 1)) if np.isfinite(net_change) else np.nan,
        "pred_tvt_sign_change_ratio": sign_change_ratio,
    }


def extract_ppt_text(ppt_path: Path) -> list[dict]:
    import re
    import zipfile
    from xml.etree import ElementTree as ET

    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    slides: list[dict] = []
    with zipfile.ZipFile(ppt_path, "r") as zf:
        names = sorted(
            [n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        for name in names:
            idx = int(re.search(r"\d+", name).group())
            root = ET.fromstring(zf.read(name))
            txt = [t.text.strip() for t in root.findall(".//a:t", ns) if t.text and t.text.strip()]
            slides.append({"slide": idx, "texts": txt})
    return slides


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_h = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    train_t = sorted(TRAIN_DIR.glob("*__typewell.csv"))
    test_h = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    test_t = sorted(TEST_DIR.glob("*__typewell.csv"))

    train_h_ids = {p.name.split("__")[0] for p in train_h}
    train_t_ids = {p.name.split("__")[0] for p in train_t}
    test_h_ids = {p.name.split("__")[0] for p in test_h}
    test_t_ids = {p.name.split("__")[0] for p in test_t}

    train_records: list[dict] = []
    typewell_records: list[dict] = []
    train_col_nan_counts: dict[str, float] = {}
    train_col_total_counts: dict[str, int] = {}

    for i, f in enumerate(train_h, 1):
        wid = f.name.split("__")[0]
        df = pd.read_csv(f)
        n = len(df)

        for c in df.columns:
            train_col_nan_counts[c] = train_col_nan_counts.get(c, 0.0) + float(df[c].isna().sum())
            train_col_total_counts[c] = train_col_total_counts.get(c, 0) + int(n)

        known_mask = df["TVT_input"].notna()
        known_len = int(known_mask.sum())
        ps_idx = known_len

        known = df.iloc[:known_len]
        pred = df.iloc[ps_idx:]

        md_diff = df["MD"].diff().dropna()
        xyz_step = np.sqrt(df["X"].diff() ** 2 + df["Y"].diff() ** 2 + df["Z"].diff() ** 2)
        surfaces_present = [c for c in SURFACE_COLS if c in df.columns]

        zone_m = eval_zone_metrics(df, ps_idx=ps_idx)

        rec = {
            "well_id": wid,
            "n_rows": int(n),
            "known_len": known_len,
            "pred_len": int(n - known_len),
            "known_ratio": float(known_len / n) if n else np.nan,
            "pred_ratio": float((n - known_len) / n) if n else np.nan,
            "ps_md": float(df["MD"].iloc[ps_idx]) if ps_idx < n else np.nan,
            "ps_z": float(df["Z"].iloc[ps_idx]) if ps_idx < n else np.nan,
            "ps_tvt_prev": float(df["TVT_input"].iloc[ps_idx - 1]) if ps_idx > 0 else np.nan,
            "ps_tvt_true_at_start": float(df["TVT"].iloc[ps_idx]) if ps_idx < n else np.nan,
            "gr_nan_ratio_total": float(df["GR"].isna().mean()),
            "gr_nan_ratio_known": float(known["GR"].isna().mean()) if known_len > 0 else np.nan,
            "gr_nan_ratio_pred": float(pred["GR"].isna().mean()) if len(pred) > 0 else np.nan,
            "gr_mean_total": float(df["GR"].mean()) if df["GR"].notna().any() else np.nan,
            "gr_std_total": float(df["GR"].std(ddof=1)) if df["GR"].notna().sum() > 1 else np.nan,
            "tvt_mean_total": float(df["TVT"].mean()) if df["TVT"].notna().any() else np.nan,
            "tvt_std_total": float(df["TVT"].std(ddof=1)) if df["TVT"].notna().sum() > 1 else np.nan,
            "tvt_min_total": float(df["TVT"].min()) if df["TVT"].notna().any() else np.nan,
            "tvt_max_total": float(df["TVT"].max()) if df["TVT"].notna().any() else np.nan,
            "corr_tvt_md_total": safe_corr(df["TVT"], df["MD"]),
            "corr_tvt_z_total": safe_corr(df["TVT"], df["Z"]),
            "corr_tvt_gr_total": safe_corr(df["TVT"], df["GR"]),
            "corr_tvt_gr_known": safe_corr(known["TVT"], known["GR"]) if known_len > 2 else np.nan,
            "corr_tvt_gr_pred": safe_corr(pred["TVT"], pred["GR"]) if len(pred) > 2 else np.nan,
            "corr_tvt_input_true_known": safe_corr(known["TVT_input"], known["TVT"]) if known_len > 2 else np.nan,
            "corr_gr_known_pred": safe_corr(known["GR"].reset_index(drop=True), pred["GR"].reset_index(drop=True))
            if known_len > 2 and len(pred) > 2
            else np.nan,
            "known_gr_mean": float(known["GR"].mean()) if known_len > 0 and known["GR"].notna().any() else np.nan,
            "known_gr_std": float(known["GR"].std(ddof=1)) if known_len > 1 and known["GR"].notna().sum() > 1 else np.nan,
            "known_tvt_mean": float(known["TVT"].mean()) if known_len > 0 else np.nan,
            "known_tvt_std": float(known["TVT"].std(ddof=1)) if known_len > 1 else np.nan,
            "azimuth_deg": azimuth_deg(df["X"], df["Y"]),
            "md_step_mean": float(md_diff.mean()) if len(md_diff) else np.nan,
            "md_step_std": float(md_diff.std(ddof=1)) if len(md_diff) > 1 else np.nan,
            "md_step_min": float(md_diff.min()) if len(md_diff) else np.nan,
            "md_step_max": float(md_diff.max()) if len(md_diff) else np.nan,
            "xyz_step_mean": float(xyz_step.mean()) if xyz_step.notna().any() else np.nan,
            "xyz_step_std": float(xyz_step.std(ddof=1)) if xyz_step.notna().sum() > 1 else np.nan,
            "xyz_step_min": float(xyz_step.min()) if xyz_step.notna().any() else np.nan,
            "xyz_step_max": float(xyz_step.max()) if xyz_step.notna().any() else np.nan,
            "ps_index": ps_idx,
            "has_pred_zone": int(ps_idx < n),
        }

        for c in surfaces_present:
            rec[f"surf_gap_{c}_mean"] = float((df[c] - df["Z"]).mean())
            rec[f"surf_gap_{c}_std"] = float((df[c] - df["Z"]).std(ddof=1))
            rec[f"corr_tvt_{c}"] = safe_corr(df["TVT"], df[c])

        rec.update(zone_m)
        rec["gr_mean_shift_pred_minus_known"] = (
            float(rec["pred_gr_mean"] - rec["known_gr_mean"])
            if np.isfinite(rec["pred_gr_mean"]) and np.isfinite(rec["known_gr_mean"])
            else np.nan
        )
        rec["gr_std_shift_pred_minus_known"] = (
            float(rec["pred_gr_std"] - rec["known_gr_std"])
            if np.isfinite(rec["pred_gr_std"]) and np.isfinite(rec["known_gr_std"])
            else np.nan
        )
        train_records.append(rec)

        tw_path = TRAIN_DIR / f"{wid}__typewell.csv"
        if tw_path.exists():
            tw = pd.read_csv(tw_path)
            tw_rec = {
                "well_id": wid,
                "tw_rows": int(len(tw)),
                "tw_gr_nan_ratio": float(tw["GR"].isna().mean()) if "GR" in tw.columns else np.nan,
                "tw_geology_nan_ratio": float(tw["Geology"].isna().mean()) if "Geology" in tw.columns else np.nan,
                "tw_geology_unique": int(tw["Geology"].nunique(dropna=True)) if "Geology" in tw.columns else 0,
                "tw_tvt_min": float(tw["TVT"].min()) if "TVT" in tw.columns else np.nan,
                "tw_tvt_max": float(tw["TVT"].max()) if "TVT" in tw.columns else np.nan,
                "tw_gr_mean": float(tw["GR"].mean()) if "GR" in tw.columns and tw["GR"].notna().any() else np.nan,
                "tw_gr_std": float(tw["GR"].std(ddof=1))
                if "GR" in tw.columns and tw["GR"].notna().sum() > 1
                else np.nan,
            }

            if known_len > 10 and {"TVT", "GR"}.issubset(tw.columns):
                kn = known[["TVT_input", "GR"]].dropna()
                twv = tw[["TVT", "GR"]].dropna()
                if len(kn) > 10 and len(twv) > 10:
                    interp_gr = robust_interp(
                        kn["TVT_input"].to_numpy(dtype=float),
                        twv["TVT"].to_numpy(dtype=float),
                        twv["GR"].to_numpy(dtype=float),
                    )
                    interp_s = pd.Series(interp_gr)
                    tw_rec["known_gr_corr_vs_typewell_gr"] = safe_corr(kn["GR"].reset_index(drop=True), interp_s)
                    tw_rec["known_gr_mae_vs_typewell_gr"] = float(np.nanmean(np.abs(kn["GR"].to_numpy() - interp_gr)))
                else:
                    tw_rec["known_gr_corr_vs_typewell_gr"] = np.nan
                    tw_rec["known_gr_mae_vs_typewell_gr"] = np.nan
            else:
                tw_rec["known_gr_corr_vs_typewell_gr"] = np.nan
                tw_rec["known_gr_mae_vs_typewell_gr"] = np.nan
            typewell_records.append(tw_rec)

        if i % 100 == 0 or i == len(train_h):
            print(f"Processed train wells: {i}/{len(train_h)}")

    train_df = pd.DataFrame(train_records)
    tw_df = pd.DataFrame(typewell_records)
    merged = train_df.merge(tw_df, on="well_id", how="left")

    test_records: list[dict] = []
    for f in test_h:
        wid = f.name.split("__")[0]
        df = pd.read_csv(f)
        n = len(df)
        known_len = int(df["TVT_input"].notna().sum())
        pred_len = int(n - known_len)
        rec = {
            "well_id": wid,
            "n_rows": int(n),
            "known_len": known_len,
            "pred_len": pred_len,
            "known_ratio": float(known_len / n) if n else np.nan,
            "pred_ratio": float(pred_len / n) if n else np.nan,
            "gr_nan_ratio_total": float(df["GR"].isna().mean()),
            "gr_nan_ratio_known": float(df.iloc[:known_len]["GR"].isna().mean()) if known_len > 0 else np.nan,
            "gr_nan_ratio_pred": float(df.iloc[known_len:]["GR"].isna().mean()) if pred_len > 0 else np.nan,
            "gr_mean_total": float(df["GR"].mean()) if df["GR"].notna().any() else np.nan,
            "gr_std_total": float(df["GR"].std(ddof=1)) if df["GR"].notna().sum() > 1 else np.nan,
            "azimuth_deg": azimuth_deg(df["X"], df["Y"]),
        }
        test_records.append(rec)

    test_df = pd.DataFrame(test_records)

    overlap_train_test = sorted(train_h_ids.intersection(test_h_ids))

    col_nan_ratio = {
        c: (train_col_nan_counts[c] / train_col_total_counts[c]) if train_col_total_counts[c] else np.nan
        for c in sorted(train_col_total_counts)
    }

    # Train summaries for metrics that matter to modeling/validation design.
    key_summaries = {
        "train_n_rows": summarize(merged["n_rows"]),
        "train_known_len": summarize(merged["known_len"]),
        "train_pred_len": summarize(merged["pred_len"]),
        "train_pred_ratio": summarize(merged["pred_ratio"]),
        "train_gr_nan_ratio_total": summarize(merged["gr_nan_ratio_total"]),
        "train_gr_nan_ratio_pred": summarize(merged["gr_nan_ratio_pred"]),
        "train_corr_tvt_gr_known": summarize(merged["corr_tvt_gr_known"]),
        "train_known_gr_corr_vs_typewell_gr": summarize(merged["known_gr_corr_vs_typewell_gr"]),
        "train_known_gr_mae_vs_typewell_gr": summarize(merged["known_gr_mae_vs_typewell_gr"]),
        "train_pred_tvt_abs_diff_mean": summarize(merged["pred_tvt_abs_diff_mean"]),
        "train_pred_tvt_abs_diff_p95": summarize(merged["pred_tvt_abs_diff_p95"]),
        "train_pred_tvt_abs_diff_max": summarize(merged["pred_tvt_abs_diff_max"]),
        "train_pred_tvt_net_change": summarize(merged["pred_tvt_net_change"]),
        "train_pred_tvt_net_slope_per_ft": summarize(merged["pred_tvt_net_slope_per_ft"]),
        "train_pred_tvt_sign_change_ratio": summarize(merged["pred_tvt_sign_change_ratio"]),
        "train_gr_mean_shift_pred_minus_known": summarize(merged["gr_mean_shift_pred_minus_known"]),
        "train_gr_std_shift_pred_minus_known": summarize(merged["gr_std_shift_pred_minus_known"]),
        "train_tw_geology_unique": summarize(merged["tw_geology_unique"]),
    }

    drift = {}
    for metric in ["n_rows", "known_len", "pred_len", "pred_ratio", "gr_nan_ratio_total", "gr_mean_total", "gr_std_total"]:
        tr = merged[metric].dropna()
        te = test_df[metric].dropna() if metric in test_df.columns else pd.Series(dtype=float)
        if len(tr) == 0 or len(te) == 0:
            drift[metric] = {"test_values": te.tolist(), "train_mean": np.nan, "train_std": np.nan, "test_zscores": []}
            continue
        tr_mean = float(tr.mean())
        tr_std = float(tr.std(ddof=1)) if len(tr) > 1 else 0.0
        if tr_std == 0.0:
            z = [np.nan for _ in te.tolist()]
        else:
            z = [float((v - tr_mean) / tr_std) for v in te.tolist()]
        drift[metric] = {
            "train_mean": tr_mean,
            "train_std": tr_std,
            "test_values": [float(v) for v in te.tolist()],
            "test_zscores": z,
        }

    ppt_file = ROOT / "AI_wellbore_geology_prediction_task_en.pptx"
    ppt_slides = extract_ppt_text(ppt_file) if ppt_file.exists() else []

    outputs = {
        "dataset_overview": {
            "train_horizontal_files": len(train_h),
            "train_typewell_files": len(train_t),
            "test_horizontal_files_visible": len(test_h),
            "test_typewell_files_visible": len(test_t),
            "train_ids_with_both_files": len(train_h_ids.intersection(train_t_ids)),
            "train_ids_missing_typewell": sorted(train_h_ids.difference(train_t_ids)),
            "visible_test_ids_with_both_files": len(test_h_ids.intersection(test_t_ids)),
            "visible_test_ids_missing_typewell": sorted(test_h_ids.difference(test_t_ids)),
            "visible_test_ids_also_in_train": overlap_train_test,
        },
        "train_column_nan_ratio": col_nan_ratio,
        "key_summaries": key_summaries,
        "train_test_drift_visible_sample": drift,
        "ppt_slide_count": len(ppt_slides),
    }

    merged.to_csv(OUT_DIR / "train_well_metrics.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_well_metrics_visible_sample.csv", index=False)
    tw_df.to_csv(OUT_DIR / "train_typewell_metrics.csv", index=False)
    pd.DataFrame(
        [{"slide": s["slide"], "text": " | ".join(s["texts"])} for s in ppt_slides]
    ).to_csv(OUT_DIR / "ppt_slides_text.csv", index=False)

    with (OUT_DIR / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {OUT_DIR / 'train_well_metrics.csv'}")
    print(f"Wrote: {OUT_DIR / 'train_typewell_metrics.csv'}")
    print(f"Wrote: {OUT_DIR / 'test_well_metrics_visible_sample.csv'}")
    print(f"Wrote: {OUT_DIR / 'ppt_slides_text.csv'}")
    print(f"Wrote: {OUT_DIR / 'eda_summary.json'}")


if __name__ == "__main__":
    main()
