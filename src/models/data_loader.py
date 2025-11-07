"""
PyTorch data loaders for time series forecasting (hierarchical + flat).

Improvements:
- Normalize facility/sector keys and year indices to avoid silent mismatches.
- Build ALL sliding windows once across a full year span.
- Split by the LAST INPUT YEAR (input_end_year) so common splits like
  train=2010–2019 / val=2020–2021 / test=2022–2023 work with sequence_length=10.

Windows quirks (seq_len=10, horizon=1):
- First window uses inputs 2010–2019 and predicts 2020.
- input_end_year is 2019; target_end_year is 2020.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("data_loader")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _year_index(series: pd.Series) -> pd.Series:
    """
    Ensure the index of a series is int years (e.g., 2010, 2011, ...).
    Accepts DatetimeIndex or int index.
    """
    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        out = series.copy()
        out.index = pd.Index(idx.year, name=idx.name)
        return out
    return series


def _normalize_keys(
    facility_series: Dict[Any, pd.Series],
    sector_series: Dict[Any, pd.Series],
    facility_metadata: pd.DataFrame,
    facility_id_col: str = "FACILITY_ID",
    industry_col: str = "INDUSTRY_TYPE",
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], pd.DataFrame]:
    """
    Normalize key types and indices across dicts/metadata.
    - Convert FACILITY_ID and INDUSTRY_TYPE to str.
    - Convert dict keys to str.
    - Convert all series indices to int years.
    """
    fm = facility_metadata.copy()
    if facility_id_col not in fm.columns:
        raise KeyError(f"Metadata missing column '{facility_id_col}'")
    if industry_col not in fm.columns:
        raise KeyError(f"Metadata missing column '{industry_col}'")

    fm[facility_id_col] = fm[facility_id_col].astype(str)
    fm[industry_col] = fm[industry_col].astype(str)

    fs_norm: Dict[str, pd.Series] = {str(k): _year_index(v) for k, v in facility_series.items()}
    ss_norm: Dict[str, pd.Series] = {str(k): _year_index(v) for k, v in sector_series.items()}

    return fs_norm, ss_norm, fm


def _check_loaded_shapes(
    facility_series: Dict[str, pd.Series],
    sector_series: Dict[str, pd.Series],
    national_series: pd.Series,
    facility_metadata: pd.DataFrame,
    industry_col: str = "INDUSTRY_TYPE",
) -> None:
    logger.info(f"Loaded {len(facility_series)} facility series")
    logger.info(f"Loaded {len(sector_series)} sector series")
    logger.info(f"Loaded national series with {len(national_series)} records")
    logger.info(f"Loaded facility metadata with {len(facility_metadata)} records")

    try:
        if facility_series:
            any_fac = next(iter(facility_series.values()))
            logger.info(f"Example facility years: min={int(min(any_fac.index))}, max={int(max(any_fac.index))}")
        if sector_series:
            any_sec = next(iter(sector_series.values()))
            logger.info(f"Example sector years:   min={int(min(any_sec.index))}, max={int(max(any_sec.index))}")
        if len(national_series) > 0:
            logger.info(f"National years:         min={int(min(national_series.index))}, max={int(max(national_series.index))}")
    except Exception:
        # If indices aren't numeric, windowing will fail later; we normalize earlier anyway.
        pass

    missing = sum(1 for s in facility_metadata[industry_col].unique() if s not in sector_series)
    if missing:
        logger.warning(f"{missing} sector keys from metadata are missing in sector_series (their facilities will be skipped).")


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """
    Flat time series dataset (no hierarchy).
    Builds sliding windows per series. Each sample carries:
      - input_end_year (for splitting)
      - target_end_year (for reference/metrics)
    """

    def __init__(
        self,
        series_dict: Dict[str, pd.Series],
        sequence_length: int = 10,
        forecast_horizon: int = 1,
        start_year: int = 2010,
        end_year: int = 2023,
    ):
        self.series_dict = series_dict
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.start_year = start_year
        self.end_year = end_year

        self.samples: List[Dict[str, Any]] = []
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        used = 0
        for series_id, series in self.series_dict.items():
            # Clip to span
            s = series[(series.index >= self.start_year) & (series.index <= self.end_year)]
            if len(s) < self.sequence_length + self.forecast_horizon:
                continue

            vals = s.values.astype(np.float32)
            years = s.index.values  # int years

            win = len(vals) - self.sequence_length - self.forecast_horizon + 1
            for i in range(win):
                input_seq = vals[i : i + self.sequence_length]
                target_seq = vals[i + self.sequence_length : i + self.sequence_length + self.forecast_horizon]

                input_end_year = int(years[i + self.sequence_length - 1])
                target_end_year = int(years[i + self.sequence_length + self.forecast_horizon - 1])

                self.samples.append(
                    {
                        "series_id": series_id,
                        "input": input_seq,
                        "target": target_seq,
                        "input_end_year": input_end_year,
                        "target_end_year": target_end_year,
                    }
                )
            used += 1

        logger.info(f"Created {len(self.samples)} samples from {used} series (flat dataset)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "series_id": s["series_id"],
            "input": torch.from_numpy(s["input"]).unsqueeze(-1),    # [seq_len, 1]
            "target": torch.from_numpy(s["target"]).unsqueeze(-1),  # [horizon, 1]
            "input_end_year": s["input_end_year"],
            "target_end_year": s["target_end_year"],
        }


class HierarchicalTimeSeriesDataset(Dataset):
    """
    Hierarchical dataset with facility, sector, and national series.
    Aligns by common years. Each sample carries input_end_year and target_end_year.
    """

    def __init__(
        self,
        facility_series: Dict[str, pd.Series],
        sector_series: Dict[str, pd.Series],
        national_series: pd.Series,
        facility_metadata: pd.DataFrame,
        sequence_length: int = 10,
        forecast_horizon: int = 1,
        start_year: int = 2010,
        end_year: int = 2023,
        facility_id_col: str = "FACILITY_ID",
        industry_col: str = "INDUSTRY_TYPE",
    ):
        self.facility_series = facility_series
        self.sector_series = sector_series
        self.national_series = national_series
        self.facility_metadata = facility_metadata
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.start_year = start_year
        self.end_year = end_year
        self.facility_id_col = facility_id_col
        self.industry_col = industry_col

        self.samples: List[Dict[str, Any]] = []
        self._prepare_hierarchical_samples()

    def _prepare_hierarchical_samples(self) -> None:
        created = 0
        used_facilities = 0

        # Pre-clip national once
        nat = self.national_series[(self.national_series.index >= self.start_year) & (self.national_series.index <= self.end_year)]

        for fac_id, fac_ts in self.facility_series.items():
            meta = self.facility_metadata[self.facility_metadata[self.facility_id_col] == fac_id]
            if meta.empty:
                continue
            sector_key = str(meta.iloc[0][self.industry_col])
            sec_ts = self.sector_series.get(sector_key)
            if sec_ts is None:
                continue

            fac = fac_ts[(fac_ts.index >= self.start_year) & (fac_ts.index <= self.end_year)]
            sec = sec_ts[(sec_ts.index >= self.start_year) & (sec_ts.index <= self.end_year)]

            common_years = sorted(list(set(fac.index) & set(sec.index) & set(nat.index)))
            if len(common_years) < self.sequence_length + self.forecast_horizon:
                continue

            f = np.array([fac[y] for y in common_years], dtype=np.float32)
            s = np.array([sec[y] for y in common_years], dtype=np.float32)
            n = np.array([nat[y] for y in common_years], dtype=np.float32)

            win = len(common_years) - self.sequence_length - self.forecast_horizon + 1
            for i in range(win):
                input_end_year = int(common_years[i + self.sequence_length - 1])
                target_end_year = int(common_years[i + self.sequence_length + self.forecast_horizon - 1])

                self.samples.append(
                    {
                        "facility_id": fac_id,
                        "sector": sector_key,
                        "facility_input": f[i : i + self.sequence_length],
                        "facility_target": f[i + self.sequence_length : i + self.sequence_length + self.forecast_horizon],
                        "sector_input": s[i : i + self.sequence_length],
                        "sector_target": s[i + self.sequence_length : i + self.sequence_length + self.forecast_horizon],
                        "national_input": n[i : i + self.sequence_length],
                        "national_target": n[i + self.sequence_length : i + self.sequence_length + self.forecast_horizon],
                        "input_end_year": input_end_year,
                        "target_end_year": target_end_year,
                    }
                )
                created += 1
            used_facilities += 1

        logger.info(f"Created {created} hierarchical samples from {used_facilities} facilities")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "facility_id": s["facility_id"],
            "sector": s["sector"],
            "facility_input": torch.from_numpy(s["facility_input"]).unsqueeze(-1),
            "facility_target": torch.from_numpy(s["facility_target"]).unsqueeze(-1),
            "sector_input": torch.from_numpy(s["sector_input"]).unsqueeze(-1),
            "sector_target": torch.from_numpy(s["sector_target"]).unsqueeze(-1),
            "national_input": torch.from_numpy(s["national_input"]).unsqueeze(-1),
            "national_target": torch.from_numpy(s["national_target"]).unsqueeze(-1),
            "input_end_year": s["input_end_year"],
            "target_end_year": s["target_end_year"],
        }


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def load_processed_data(data_dir: str = "data/processed") -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], pd.Series, pd.DataFrame]:
    """
    Load processed artifacts produced by preprocessing.

    Expects:
      - facility_series.pkl : Dict[facility_id -> pd.Series(year -> value)]
      - sector_series.pkl   : Dict[sector_name -> pd.Series(year -> value)]
      - national_series.pkl : pd.Series(year -> value)
      - facility_metadata.csv with FACILITY_ID, INDUSTRY_TYPE
    """
    p = Path(data_dir)
    with open(p / "facility_series.pkl", "rb") as f:
        facility_series = pickle.load(f)
    with open(p / "sector_series.pkl", "rb") as f:
        sector_series = pickle.load(f)
    with open(p / "national_series.pkl", "rb") as f:
        national_series = pickle.load(f)
    facility_metadata = pd.read_csv(p / "facility_metadata.csv")

    # Normalize keys & indices
    facility_series, sector_series, facility_metadata = _normalize_keys(
        facility_series, sector_series, facility_metadata
    )
    national_series = _year_index(national_series)

    _check_loaded_shapes(facility_series, sector_series, national_series, facility_metadata)
    return facility_series, sector_series, national_series, facility_metadata


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def create_dataloaders(
    data_dir: str = "data/processed",
    sequence_length: int = 10,
    forecast_horizon: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,     # Windows-safe default
    pin_memory: bool = False, # True if using CUDA
    # Split by LAST INPUT YEAR (input_end_year):
    train_years: Tuple[int, int] = (2010, 2019),
    val_years: Tuple[int, int] = (2020, 2021),
    test_years: Tuple[int, int] = (2022, 2023),
    hierarchical: bool = True,
    full_span: Tuple[int, int] = (2010, 2023),
    facility_id_col: str = "FACILITY_ID",
    industry_col: str = "INDUSTRY_TYPE",
) -> Dict[str, DataLoader]:
    """
    Build a full dataset over `full_span`, then split by each sample's LAST INPUT YEAR (`input_end_year`).

    Returns a dict with 'train' | 'val' | 'test' DataLoaders.
    """
    # Load & normalize
    facility_series, sector_series, national_series, facility_metadata = load_processed_data(data_dir)

    # Build full dataset once
    full_start, full_end = full_span
    if hierarchical:
        full_dataset: Dataset = HierarchicalTimeSeriesDataset(
            facility_series=facility_series,
            sector_series=sector_series,
            national_series=national_series,
            facility_metadata=facility_metadata,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            start_year=full_start,
            end_year=full_end,
            facility_id_col=facility_id_col,
            industry_col=industry_col,
        )
    else:
        full_dataset = TimeSeriesDataset(
            series_dict=facility_series,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            start_year=full_start,
            end_year=full_end,
        )

    # Split by input_end_year
    def _split_indices(ds: Dataset, start_y: int, end_y: int) -> List[int]:
        samples = getattr(ds, "samples", None)
        if samples is None:
            raise RuntimeError("Dataset must define `self.samples`.")
        return [i for i, s in enumerate(samples) if start_y <= int(s["input_end_year"]) <= end_y]

    train_idx = _split_indices(full_dataset, train_years[0], train_years[1])
    val_idx   = _split_indices(full_dataset,  val_years[0],  val_years[1])
    test_idx  = _split_indices(full_dataset, test_years[0],  test_years[1])

    # Helpful diagnostics if something is empty
    def _require_non_empty(name: str, idxs: List[int]) -> None:
        if len(idxs) == 0:
            # Compute quick coverage of input_end_years
            samples = getattr(full_dataset, "samples")
            yrs = [int(s["input_end_year"]) for s in samples]
            if yrs:
                yr_min, yr_max = min(yrs), max(yrs)
                logger.error(
                    f"{name} split has 0 samples. Available input_end_year range in full dataset: "
                    f"{yr_min}–{yr_max}. Your {name}_years={train_years if name=='train' else val_years if name=='val' else test_years}. "
                    f"Consider widening the range or reducing sequence_length."
                )
            raise ValueError(
                f"{name} split has 0 samples after filtering by input_end_year. "
                f"Params -> sequence_length={sequence_length}, forecast_horizon={forecast_horizon}, "
                f"{name}_years={train_years if name=='train' else val_years if name=='val' else test_years}, "
                f"full_span={full_span}"
            )

    _require_non_empty("train", train_idx)
    _require_non_empty("val", val_idx)
    _require_non_empty("test", test_idx)

    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    # DataLoaders (Windows-safe defaults)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    logger.info("Created dataloaders (split by input_end_year):")
    logger.info(f"  Train: {len(train_idx)} samples; batches ≈ {len(train_loader)}")
    logger.info(f"  Val:   {len(val_idx)} samples; batches ≈ {len(val_loader)}")
    logger.info(f"  Test:  {len(test_idx)} samples; batches ≈ {len(test_loader)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ---------------------------------------------------------------------
# Quick local test (safe on Windows; avoids multiprocessing workers)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    loaders = create_dataloaders(
        data_dir="data/processed",
        sequence_length=10,
        forecast_horizon=1,
        batch_size=32,
        num_workers=0,     # Windows-safe
        pin_memory=False,  # True if using CUDA
        train_years=(2010, 2019),
        val_years=(2020, 2021),
        test_years=(2022, 2023),
        hierarchical=True,
        full_span=(2010, 2023),
    )

    # Peek a batch to verify shapes without crashing
    batch = next(iter(loaders["train"]))
    logger.info("Sample train batch keys: %s", list(batch.keys()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info("  %s: %s", k, tuple(v.shape))
        else:
            logger.info("  %s: %s", k, type(v))
