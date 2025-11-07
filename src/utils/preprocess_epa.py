# utils/preprocess_epa.py
import zipfile, io, os, json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# ---- Config-like knobs (keep in sync with your YAML) ----
DATA_ZIP = "/mnt/data/out_epa.zip"          # your uploaded file
RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
SEQ_LEN  = 12                                 # sequence_length (L)
HORIZON  = 1                                  # forecast_horizon (H)
TRAIN_YEARS = (2010, 2018)                    # inclusive range used for samples
VAL_YEARS   = (2019, 2019)
TEST_YEARS  = (2020, 2022)

# Column name normalizer: map many possible GHGRP layouts into canonical names
CANDIDATES = {
    "facility_id": ["facility_id","facility","FACILITY_ID","ghgrp_facility_id","id"],
    "sector":      ["sector","subpart","sub_part_id","industry","NAICS","GHGRP_Subpart"],
    "year":        ["year","reporting_year","Year","REPORTING_YEAR"],
    "co2e":        ["co2e_emission","total_co2e","co2e","emissions_co2e_tonnes",
                    "CO2e","total_emissions_co2e","sum_co2e"]
}

def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of {names} found in columns: {df.columns.tolist()}")

def unzip_if_needed():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        for n in z.namelist():
            # only extract CSV/TSV/Parquet
            lower = n.lower()
            if lower.endswith(".csv") or lower.endswith(".tsv") or lower.endswith(".parquet"):
                z.extract(n, RAW_DIR)

def load_all_frames():
    files = list(RAW_DIR.rglob("*.csv")) + list(RAW_DIR.rglob("*.tsv")) + list(RAW_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No tabular files found in {RAW_DIR}.")
    dfs = []
    for f in files:
        if f.suffix.lower()==".csv":
            df = pd.read_csv(f)
        elif f.suffix.lower()==".tsv":
            df = pd.read_csv(f, sep="\t")
        else:
            df = pd.read_parquet(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    fid = pick_col(df, CANDIDATES["facility_id"])
    sec = pick_col(df, CANDIDATES["sector"])
    yr  = pick_col(df, CANDIDATES["year"])
    e   = pick_col(df, CANDIDATES["co2e"])
    df = df[[fid, sec, yr, e]].rename(columns={
        fid:"facility_id", sec:"sector", yr:"year", e:"co2e"
    })
    # clean types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["facility_id"] = df["facility_id"].astype(str)
    df["sector"] = df["sector"].astype(str)
    df["co2e"] = pd.to_numeric(df["co2e"], errors="coerce")
    df = df.dropna(subset=["year","co2e"]).copy()
    return df

def ensure_complete_years(df: pd.DataFrame) -> pd.DataFrame:
    # create a full grid per (facility, year) and fill missing with 0
    yrs = np.arange(df["year"].min(), df["year"].max()+1)
    facilities = df["facility_id"].unique()
    base = (pd.MultiIndex.from_product([facilities, yrs], names=["facility_id","year"])
                  .to_frame(index=False))
    # keep sector as the most frequent per facility
    sec_map = (df.groupby(["facility_id","sector"])
                 .size().reset_index(name="n")
                 .sort_values(["facility_id","n"], ascending=[True, False])
                 .drop_duplicates("facility_id")[["facility_id","sector"]])
    out = base.merge(sec_map, on="facility_id", how="left")
    out = out.merge(df[["facility_id","year","co2e"]], on=["facility_id","year"], how="left")
    out["co2e"] = out["co2e"].fillna(0.0)
    return out

def standardize(series: pd.Series):
    # log1p to tame heavy tails, then z-score across global distribution
    x = np.log1p(series.values.astype(float))
    mu, sigma = x.mean(), x.std() if series.size>1 else 1.0
    if sigma == 0: sigma = 1.0
    return (x - mu) / sigma, {"log1p": True, "mu": float(mu), "sigma": float(sigma)}

def build_tensors(df_full: pd.DataFrame):
    """
    Create (X, y_facility, y_sector, y_national) samples by sliding window.
    Shapes:
      X: [N, SEQ_LEN, 1]
      y_*: [N, HORIZON, 1]
    """
    years_all = np.arange(df_full["year"].min(), df_full["year"].max()+1)
    # standardize globally on co2e
    x_std, norm_meta = standardize(df_full["co2e"])
    df_full = df_full.copy()
    df_full["co2e_std"] = x_std

    # pre-compute sector & national panels (standardized the same way)
    # Use the same normalized values for consistency across levels
    sec_panel = (df_full.groupby(["sector","year"])["co2e_std"]
                       .sum().reset_index())
    nat_panel = (df_full.groupby(["year"])["co2e_std"]
                       .sum().reset_index().rename(columns={"co2e_std":"nat_std"}))

    samples = []
    for fac, g in df_full.groupby("facility_id"):
        g = g.sort_values("year")
        x_vec = g["co2e_std"].to_numpy()  # ordered by year
        y_vec_years = g["year"].to_numpy()

        for i in range(0, len(x_vec) - SEQ_LEN - HORIZON + 1):
            in_years = y_vec_years[i:i+SEQ_LEN]
            out_years = y_vec_years[i+SEQ_LEN:i+SEQ_LEN+HORIZON]

            # facility input & target
            X = x_vec[i:i+SEQ_LEN].reshape(SEQ_LEN, 1)
            y_fac = x_vec[i+SEQ_LEN:i+SEQ_LEN+HORIZON].reshape(HORIZON, 1)

            # sector target: sum standardized facility emissions for the sector for those out_years
            sec = g["sector"].iloc[0]
            sec_y = (sec_panel[(sec_panel["sector"]==sec) & (sec_panel["year"].isin(out_years))]
                        .sort_values("year")["co2e_std"].to_numpy())
            if len(sec_y) < HORIZON:
                # pad if somehow missing
                sec_y = np.pad(sec_y, (0, HORIZON-len(sec_y)))
            y_sec = sec_y.reshape(HORIZON, 1)

            # national target
            nat_y = (nat_panel[nat_panel["year"].isin(out_years)]
                       .sort_values("year")["nat_std"].to_numpy())
            if len(nat_y) < HORIZON:
                nat_y = np.pad(nat_y, (0, HORIZON-len(nat_y)))
            y_nat = nat_y.reshape(HORIZON, 1)

            samples.append({
                "facility_id": fac,
                "sector": sec,
                "in_years": in_years.tolist(),
                "out_years": out_years.tolist(),
                "X": X.astype(np.float32),
                "y_fac": y_fac.astype(np.float32),
                "y_sec": y_sec.astype(np.float32),
                "y_nat": y_nat.astype(np.float32)
            })

    # split by year ranges defined above (based on the OUT years)
    def which_split(out_years):
        last = max(out_years)
        if TRAIN_YEARS[0] <= last <= TRAIN_YEARS[1]:
            return "train"
        if VAL_YEARS[0] <= last <= VAL_YEARS[1]:
            return "val"
        if TEST_YEARS[0] <= last <= TEST_YEARS[1]:
            return "test"
        return None

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    for s in samples:
        split = which_split(s["out_years"])
        if split:
            buckets[split].append(s)

    def pack_and_save(name, items):
        if not items:
            return
        X = torch.from_numpy(np.stack([it["X"] for it in items], axis=0))            # [N, L, 1]
        y_fac = torch.from_numpy(np.stack([it["y_fac"] for it in items], axis=0))    # [N, H, 1]
        y_sec = torch.from_numpy(np.stack([it["y_sec"] for it in items], axis=0))    # [N, H, 1]
        y_nat = torch.from_numpy(np.stack([it["y_nat"] for it in items], axis=0))    # [N, H, 1]
        meta = {
            "facility_id": [it["facility_id"] for it in items],
            "sector": [it["sector"] for it in items],
            "in_years": [it["in_years"] for it in items],
            "out_years": [it["out_years"] for it in items],
            "normalization": {"type":"global_log1p_zscore", **norm_meta},
            "seq_len": SEQ_LEN, "horizon": HORIZON
        }
        torch.save({
            "facility_input": X,                 # what your LightningModule expects to .forward(...)
            "facility_target": y_fac,
            "sector_target": y_sec,
            "national_target": y_nat,
            "meta": meta
        }, PROC_DIR / f"{name}.pt")

    for k,v in buckets.items():
        pack_and_save(k, v)

    print(f"Saved: {[(k, len(v)) for k,v in buckets.items()]}")
    print(f"Files in {PROC_DIR}: {[p.name for p in PROC_DIR.glob('*.pt')]}")

if __name__ == "__main__":
    unzip_if_needed()
    df = load_all_frames()
    df = canonicalize(df)
    df = ensure_complete_years(df)
    build_tensors(df)
