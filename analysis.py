"""
Assignment Analysis Script

This script loads the five provided datasets and produces all deliverables for
the assignment:

- Funnel: Registrations → First Deposit → First Bet → Active in first 30 days
  - Overall conversion and segmented by acquisition channel and signup cohort
- Retention & Engagement: Active days in first 30 days; cohort buckets; deposit
  contribution by cohort
- Gap between first deposit and first bet: distribution and summary stats
- Player Segmentation: Top 10% by deposit (using first deposit as proxy)
- First deposit distribution: bins, histogram, correlation to early activity

Outputs are written to /workspace/output as CSV and HTML files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    import plotly.express as px  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_backend(path: Path, sheet_preference: Iterable[str] = ("Backend Data", "Sheet1")) -> pd.DataFrame:
    """Read an Excel file choosing the backend sheet and using row 2 as header.

    Drops the common 'Unnamed: 0' column if present.
    """
    xls = pd.ExcelFile(path)
    chosen = None
    for name in sheet_preference:
        if name in xls.sheet_names:
            chosen = name
            break
    if chosen is None:
        chosen = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=chosen, header=1)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])  # index-like column
    return df


def to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(c).strip().replace(" ", "_").replace("-", "_") for c in renamed.columns]
    return renamed


def load_datasets() -> dict:
    players = to_snake_case(read_backend(DATA_DIR / "Player_Details.xlsx"))
    first_bets = to_snake_case(read_backend(DATA_DIR / "First_Bet_Data.xlsx", sheet_preference=("Sheet1",)))
    first_deposits = to_snake_case(read_backend(DATA_DIR / "First_Deposit_Data.xlsx"))
    activity = to_snake_case(read_backend(DATA_DIR / "Player_Activity_Data.xlsx"))
    bonus = to_snake_case(read_backend(DATA_DIR / "BonusCost_Data.xlsx"))

    # Normalize IDs
    players.rename(columns={"src_player_id": "Src_Player_Id"}, inplace=True)
    activity.rename(columns={"src_player_id": "Src_Player_Id"}, inplace=True)
    bonus.rename(columns={"src_player_id": "Src_Player_Id", "Src_PLAYER_ID": "Src_Player_Id"}, inplace=True)

    # Parse dates
    for col in ("Signup_Date", "Date_Of_Birth"):
        if col in players.columns:
            players[col] = pd.to_datetime(players[col], errors="coerce")
    if "System_First_Bet_Datetime" in first_bets.columns:
        first_bets["System_First_Bet_Datetime"] = pd.to_datetime(first_bets["System_First_Bet_Datetime"], errors="coerce")
    if "First_Deposit_Date" in first_deposits.columns:
        first_deposits["First_Deposit_Date"] = pd.to_datetime(first_deposits["First_Deposit_Date"], errors="coerce")
    if "ActivityMonth" in activity.columns:
        activity["ActivityMonth"] = pd.to_datetime(activity["ActivityMonth"], errors="coerce")

    # Numeric coercions
    for col in ("First_Deposit_Amount",):
        if col in first_deposits.columns:
            first_deposits[col] = pd.to_numeric(first_deposits[col], errors="coerce")
    for col in ("ActivePlayerDays", "Bet_Amount", "Win_Amount", "Gross_Win", "Net_Gross_Win"):
        if col in activity.columns:
            activity[col] = pd.to_numeric(activity[col], errors="coerce")
    if "BONUS_COST" in bonus.columns:
        bonus["BONUS_COST"] = pd.to_numeric(bonus["BONUS_COST"], errors="coerce")

    return {
        "players": players,
        "first_bets": first_bets,
        "first_deposits": first_deposits,
        "activity": activity,
        "bonus": bonus,
    }


def build_base_tables(d: dict) -> pd.DataFrame:
    players: pd.DataFrame = d["players"].copy()
    first_deposits: pd.DataFrame = d["first_deposits"].copy()
    first_bets: pd.DataFrame = d["first_bets"].copy()
    activity: pd.DataFrame = d["activity"].copy()

    # Filter internal
    if "Internal_Player_YN" in players.columns:
        players = players[players["Internal_Player_YN"].fillna("N") != "Y"]
    players["acquisition_channel"] = players.get("acquisition_channel", "Unknown").fillna("Unknown")
    players["signup_month"] = players["Signup_Date"].dt.to_period("M").astype(str)

    # Aggregate activity by player-month to avoid double counting products
    agg_cols = [c for c in ("ActivePlayerDays", "Bet_Amount", "Win_Amount", "Gross_Win", "Net_Gross_Win") if c in activity.columns]
    act_m = (
        activity.groupby(["Src_Player_Id", "ActivityMonth"], as_index=False)[agg_cols].sum()
        if agg_cols else activity[["Src_Player_Id", "ActivityMonth"]].drop_duplicates()
    )
    act_m["month_start"] = act_m["ActivityMonth"].dt.to_period("M").dt.to_timestamp(how="start")
    act_m["month_end"] = act_m["ActivityMonth"].dt.to_period("M").dt.to_timestamp(how="end")

    # 30-day window overlap per player-month
    signup = players[["Src_Player_Id", "Signup_Date"]].dropna()
    m = act_m.merge(signup, on="Src_Player_Id", how="inner")
    ws = m["Signup_Date"]
    we = m["Signup_Date"] + pd.Timedelta(days=30)
    start = np.where(m["month_start"] > ws, m["month_start"], ws)
    end = np.where(m["month_end"] < we, m["month_end"], we)
    overlap_td = (pd.to_datetime(end) - pd.to_datetime(start))
    # Convert to fractional days using numpy timedelta to avoid resolution issues
    overlap_days = overlap_td / np.timedelta64(1, "D")
    overlap_days = np.where(overlap_days < 0, 0, overlap_days)
    if "ActivePlayerDays" in act_m.columns:
        window_days = np.minimum(m["ActivePlayerDays"].fillna(0).to_numpy(dtype=float), overlap_days)
    else:
        window_days = overlap_days
    active_30 = (
        pd.DataFrame({"Src_Player_Id": m["Src_Player_Id"], "active_days_30d": window_days})
        .groupby("Src_Player_Id", as_index=False)["active_days_30d"].sum()
    )

    # Base table
    base = players[["Src_Player_Id", "Signup_Date", "acquisition_channel", "signup_month"]].drop_duplicates()
    base = base.merge(
        first_deposits[["Src_Player_Id", "First_Deposit_Date", "First_Deposit_Amount", "First_Deposit_Channel", "First_Deposit_Method"]],
        on="Src_Player_Id",
        how="left",
    )
    base = base.merge(
        first_bets[["Src_Player_Id", "System_First_Bet_Datetime", "System_First_BetSlip_Amt", "System_First_Bet_Product_Group", "System_First_Bet_Product"]],
        on="Src_Player_Id",
        how="left",
    )
    base = base.merge(active_30, on="Src_Player_Id", how="left")
    base["active_days_30d"] = base["active_days_30d"].fillna(0)
    base["active_30d_flag"] = base["active_days_30d"] > 0

    # Stages
    base["stage_registration"] = True
    base["stage_first_deposit"] = base["First_Deposit_Date"].notna()
    base["stage_first_bet"] = base["System_First_Bet_Datetime"].notna()
    base["stage_active_30d"] = base["active_30d_flag"]

    return base


def compute_funnel(base: pd.DataFrame) -> None:
    counts = {
        "Registrations": int(base["stage_registration"].sum()),
        "First_Deposit": int(base["stage_first_deposit"].sum()),
        "First_Bet": int(base["stage_first_bet"].sum()),
        "Active_30d": int(base["stage_active_30d"].sum()),
    }
    conv_dep = counts["First_Deposit"] / counts["Registrations"] if counts["Registrations"] else np.nan
    conv_bet = base.loc[base["stage_first_deposit"], "stage_first_bet"].mean() if counts["First_Deposit"] else np.nan
    conv_active = base.loc[base["stage_first_bet"], "stage_active_30d"].mean() if counts["First_Bet"] else np.nan
    funnel_summary = pd.DataFrame(
        {
            "stage": ["Registrations", "First_Deposit", "First_Bet", "Active_30d"],
            "count": [counts[s] for s in ["Registrations", "First_Deposit", "First_Bet", "Active_30d"]],
            "conversion_from_prev": [np.nan, conv_dep, conv_bet, conv_active],
        }
    )
    funnel_summary.to_csv(OUTPUT_DIR / "funnel_overall.csv", index=False)

    def funnel_by(group_cols: List[str]) -> pd.DataFrame:
        out_rows = []
        for keys, sub in base.groupby(group_cols):
            regs = len(sub)
            dep = int(sub["stage_first_deposit"].sum())
            bet = int(sub["stage_first_bet"].sum())
            act30 = int(sub["stage_active_30d"].sum())
            row = {
                "count": regs,
                "First_Deposit": dep,
                "First_Bet": bet,
                "Active_30d": act30,
                "conv_dep": dep / regs if regs else np.nan,
                "conv_bet": sub.loc[sub["stage_first_deposit"], "stage_first_bet"].mean() if dep else np.nan,
                "conv_active": sub.loc[sub["stage_first_bet"], "stage_active_30d"].mean() if bet else np.nan,
            }
            if isinstance(keys, tuple):
                for c, k in zip(group_cols, keys):
                    row[c] = k
            else:
                row[group_cols[0]] = keys
            out_rows.append(row)
        return pd.DataFrame(out_rows).sort_values("count", ascending=False)

    funnel_by(["acquisition_channel"]).to_csv(OUTPUT_DIR / "funnel_by_acquisition_channel.csv", index=False)
    funnel_by(["signup_month"]).to_csv(OUTPUT_DIR / "funnel_by_signup_month.csv", index=False)

    return None


def retention_and_cohorts(base: pd.DataFrame, first_deposits: pd.DataFrame) -> None:
    # Buckets of active days in first 30 days
    bins = [-0.1, 0, 1, 2, 3, 5, 6, 10, 30]
    labels = ["0", "1", "2", "3", "4-5", "6", "7-10", "11-30"]
    base["active_days_bucket"] = pd.cut(base["active_days_30d"], bins=bins, labels=labels, right=True)

    cohort_counts = (
        base.groupby("active_days_bucket", dropna=False)["Src_Player_Id"].nunique().reset_index(name="players")
    )
    cohort_counts["share_%"] = (cohort_counts["players"] / cohort_counts["players"].sum()) * 100
    cohort_counts.to_csv(OUTPUT_DIR / "active_days_cohorts.csv", index=False)

    # Deposit contribution by cohort (proxy: first deposit amount only)
    player_dep = (
        first_deposits.groupby("Src_Player_Id", as_index=False)["First_Deposit_Amount"].sum().rename(columns={"First_Deposit_Amount": "total_deposit_amount"})
    )
    base_dep = base.merge(player_dep, on="Src_Player_Id", how="left")
    base_dep["total_deposit_amount"] = base_dep["total_deposit_amount"].fillna(0)
    deposit_by_cohort = (
        base_dep.groupby("active_days_bucket", dropna=False)["total_deposit_amount"].sum().reset_index()
    )
    deposit_by_cohort["share_%"] = (
        deposit_by_cohort["total_deposit_amount"] / deposit_by_cohort["total_deposit_amount"].sum() * 100
    )
    deposit_by_cohort.to_csv(OUTPUT_DIR / "deposit_contribution_by_active_cohort.csv", index=False)


def gap_deposit_to_bet(base: pd.DataFrame) -> pd.DataFrame:
    df = base.dropna(subset=["First_Deposit_Date", "System_First_Bet_Datetime"]).copy()
    df = df[df["System_First_Bet_Datetime"] >= df["First_Deposit_Date"]].copy()
    df["days_deposit_to_bet"] = (
        (df["System_First_Bet_Datetime"] - df["First_Deposit_Date"]).dt.total_seconds() / (24 * 3600)
    )
    stats_tbl = pd.DataFrame(
        {
            "metric": ["mean", "median", "p75", "max"],
            "value": [
                df["days_deposit_to_bet"].mean(),
                df["days_deposit_to_bet"].median(),
                df["days_deposit_to_bet"].quantile(0.75),
                df["days_deposit_to_bet"].max(),
            ],
        }
    )
    stats_tbl.to_csv(OUTPUT_DIR / "gap_deposit_to_bet_stats.csv", index=False)

    bins = [-0.01, 0, 1, 3, 7, 14, 30, 60, 180, 1e9]
    labels = ["same_day", "1d", "2-3d", "4-7d", "8-14d", "15-30d", "31-60d", "61-180d", ">180d"]
    hist = pd.cut(df["days_deposit_to_bet"], bins=bins, labels=labels)
    dist_tbl = hist.value_counts(dropna=False).reset_index()
    dist_tbl.columns = ["gap_bucket", "players"]
    dist_tbl["share_%"] = (dist_tbl["players"] / dist_tbl["players"].sum()) * 100
    dist_tbl.to_csv(OUTPUT_DIR / "gap_deposit_to_bet_distribution.csv", index=False)
    return stats_tbl


def top10_segmentation(base: pd.DataFrame) -> float:
    # Proxy: total deposit = first deposit amount (only info available)
    dep = (
        base.groupby("Src_Player_Id", as_index=False)["First_Deposit_Amount"].sum().rename(columns={"First_Deposit_Amount": "total_deposit_amount"})
    )
    dep = dep.sort_values("total_deposit_amount", ascending=False)
    cutoff = int(math.ceil(0.10 * len(dep))) if len(dep) else 0
    top10 = dep.head(cutoff)
    share = float(top10["total_deposit_amount"].sum() / dep["total_deposit_amount"].sum()) if len(dep) and dep["total_deposit_amount"].sum() > 0 else float("nan")
    pd.DataFrame({"metric": ["num_players", "top10_share_of_deposits"], "value": [len(dep), share]}).to_csv(
        OUTPUT_DIR / "top10_segmentation_summary.csv", index=False
    )
    top10[["Src_Player_Id", "total_deposit_amount"]].to_csv(OUTPUT_DIR / "top10_players_by_deposit.csv", index=False)
    return share


def first_deposit_visuals(base: pd.DataFrame) -> float:
    fd = base[["Src_Player_Id", "First_Deposit_Amount", "active_days_30d", "active_30d_flag"]].copy()
    fd = fd.dropna(subset=["First_Deposit_Amount"]).copy()
    fd["First_Deposit_Amount"] = pd.to_numeric(fd["First_Deposit_Amount"], errors="coerce")
    fd = fd.dropna(subset=["First_Deposit_Amount"]).copy()

    # Bins
    bins = [-0.01, 5, 10, 20, 50, 100, 200, 500, 1000, 1e9]
    labels = ["<=5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500", "501-1000", ">1000"]
    fd["bucket"] = pd.cut(fd["First_Deposit_Amount"], bins=bins, labels=labels)
    bucket_dist = fd["bucket"].value_counts().sort_index().reset_index()
    bucket_dist.columns = ["bucket", "players"]
    bucket_dist["share_%"] = (bucket_dist["players"] / bucket_dist["players"].sum()) * 100
    bucket_dist.to_csv(OUTPUT_DIR / "first_deposit_bucket_distribution.csv", index=False)

    corr = float(fd[["First_Deposit_Amount", "active_days_30d"]].corr().iloc[0, 1]) if len(fd) else float("nan")

    # Aggregations by bucket
    agg = (
        fd.groupby("bucket", dropna=False)
        .agg(players=("Src_Player_Id", "nunique"), avg_first_dep=("First_Deposit_Amount", "mean"), active_rate=("active_30d_flag", "mean"), avg_active_days=("active_days_30d", "mean"))
        .reset_index()
    )
    agg.to_csv(OUTPUT_DIR / "first_deposit_bucket_engagement.csv", index=False)

    # Visuals
    if PLOTLY_AVAILABLE and len(fd):
        try:
            px.histogram(fd, x="First_Deposit_Amount", nbins=80, title="First Deposit Amount Distribution", log_y=True).write_html(
                str(OUTPUT_DIR / "first_deposit_histogram.html")
            )
            sample = fd.sample(min(10000, len(fd)), random_state=42)
            px.scatter(sample, x="First_Deposit_Amount", y="active_days_30d", trendline="ols", title="First Deposit vs 30-day Active Days").write_html(
                str(OUTPUT_DIR / "first_deposit_vs_active_days_scatter.html")
            )
        except Exception:
            pass

    return corr


def main() -> None:
    data = load_datasets()
    base = build_base_tables(data)

    # Persist a light-weight base for reproducibility/debugging
    base_out_cols = [
        "Src_Player_Id",
        "Signup_Date",
        "acquisition_channel",
        "signup_month",
        "First_Deposit_Date",
        "First_Deposit_Amount",
        "System_First_Bet_Datetime",
        "active_days_30d",
        "active_30d_flag",
    ]
    base[base_out_cols].to_parquet(OUTPUT_DIR / "base_players.parquet", index=False)

    compute_funnel(base)
    retention_and_cohorts(base, data["first_deposits"])
    gap_stats = gap_deposit_to_bet(base)
    top10_share = top10_segmentation(base)
    corr = first_deposit_visuals(base)

    # Summarize KPIs to JSON
    funnel_overall = pd.read_csv(OUTPUT_DIR / "funnel_overall.csv").to_dict(orient="records")
    kpis = {
        "funnel_overall": funnel_overall,
        "gap_stats": {r["metric"]: float(r["value"]) for r in gap_stats.to_dict(orient="records")},
        "top10_deposit_share": float(top10_share) if not pd.isna(top10_share) else None,
        "first_deposit_vs_active_corr": float(corr) if not pd.isna(corr) else None,
    }
    (OUTPUT_DIR / "kpi_summary.json").write_text(json.dumps(kpis, indent=2))

    print("Analysis complete. Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

