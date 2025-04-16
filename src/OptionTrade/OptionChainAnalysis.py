import datetime
import math
import os
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.OptionTrade.BlackScholes import bs_delta, implied_volatility


def calculate_greeks(
    df: pd.DataFrame,
    expiration_date: str,
    date: str,
    option_type: str,
    r: float = 0.045,
):
    iv_list = []
    delta_list = []
    for i, data in df.iterrows():
        underlying_price = data["underlyingPrice"]
        strike = data["strike"]
        t = (
            datetime.datetime.strptime(expiration_date, "%Y-%m-%d")
            - datetime.datetime.strptime(date, "%Y-%m-%d")
        ).days / 365
        option_price = data["mid"]

        iv = implied_volatility(
            underlying_price, strike, t, r, option_price, option_type
        )
        delta = bs_delta(underlying_price, strike, t, r, option_price, option_type)

        iv_list.append(iv)
        delta_list.append(delta)

    df["iv"] = iv_list
    df["delta"] = delta_list


def integrate_option_chain(date: str):
    files = glob(f"data/{date}/SPY_*.csv")
    exp_date_list = sorted(set([f.split("_")[3].split(".")[0] for f in files]))

    merge_df = pd.DataFrame()
    for i, exp_date in enumerate(exp_date_list):

        call_df = pd.read_csv(f"data/{date}/SPY_CALL_{date}_{exp_date}.csv")
        put_df = pd.read_csv(f"data/{date}/SPY_PUT_{date}_{exp_date}.csv")

        if call_df["delta"].values.any():
            calculate_greeks(call_df, exp_date, date, "call")

        if put_df["delta"].values.any():
            calculate_greeks(put_df, exp_date, date, "put")

        df = pd.merge(
            call_df,
            put_df,
            on="strike",
            how="inner",
            suffixes=("_call", "_put"),
            validate="one_to_one",
        )
        df["exp_date"] = exp_date
        query_df = df.query(
            "(delta_call>=0.1 & delta_call<=0.55) | (delta_put>-0.55 & delta_put<=-0.1)"
        )
        if i == 0:
            merge_df = query_df
        else:
            merge_df = pd.concat([merge_df, query_df]).reset_index(drop=True)

    return merge_df


def create_open_interest_dataframe(df: pd.DataFrame):
    pivot_call_oi = df.pivot(
        index="strike", columns="exp_date", values="openInterest_call"
    )
    pivot_put_oi = df.pivot(
        index="strike", columns="exp_date", values="openInterest_put"
    )
    pivot_oi_diff = pivot_call_oi - pivot_put_oi
    pivot_oi_diff.fillna(0, inplace=True)
    return pivot_oi_diff


def create_volume_dataframe(df: pd.DataFrame):
    pivot_call_volume = df.pivot(
        index="strike", columns="exp_date", values="volume_call"
    )
    pivot_put_volume = df.pivot(index="strike", columns="exp_date", values="volume_put")
    pivot_volume_diff = pivot_call_volume - pivot_put_volume
    pivot_volume_diff.fillna(0, inplace=True)
    return pivot_volume_diff


def create_implied_volatility_dataframe(df: pd.DataFrame):
    pivot_call_iv = df.pivot(index="strike", columns="exp_date", values="iv_call")
    pivot_put_iv = df.pivot(index="strike", columns="exp_date", values="iv_put")
    pivot_call_iv.fillna(0, inplace=True)
    pivot_put_iv.fillna(0, inplace=True)
    pivot_iv_diff = pivot_call_iv - pivot_put_iv
    pivot_iv_diff.fillna(0, inplace=True)
    return pivot_call_iv, pivot_put_iv, pivot_iv_diff


def plot_open_interest_heatmap(
    pivot_oi_diff: pd.DataFrame, date: str, underlying_price: float
):
    mask = pivot_oi_diff < 0
    log_pivot_oi_diff = np.log(np.abs(pivot_oi_diff))
    log_pivot_oi_diff.replace([np.inf, -np.inf], 0, inplace=True)
    log_pivot_oi_diff[mask] = -1 * log_pivot_oi_diff[mask]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(log_pivot_oi_diff, cmap="RdBu_r", ax=ax)
    strike_index = list(pivot_oi_diff.index)  # [450, 460, 470]
    line_strike = math.floor(underlying_price)
    line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
    ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
    plt.title(f"Spread (Call - Put) Open Interest (logarithm) Heatmap on {date}")
    plt.xlabel("Expiration")
    plt.ylabel("Strike")
    fig.tight_layout()
    fig.savefig(f"heatmap/{date}/openInterest_heatmap_{date}.png")


def plot_volume_heatmap(
    pivot_volume_diff: pd.DataFrame, date: str, underlying_price: float
):
    mask = pivot_volume_diff < 0
    log_pivot_volume_diff = np.log(np.abs(pivot_volume_diff))
    log_pivot_volume_diff.replace([np.inf, -np.inf], 0, inplace=True)
    log_pivot_volume_diff[mask] = -1 * log_pivot_volume_diff[mask]
    fig, ax = plt.subplots(figsize=(10, 10))
    hm = sns.heatmap(log_pivot_volume_diff, cmap="RdBu_r", ax=ax)
    xmin, xmax = hm.get_xlim()
    strike_index = list(pivot_volume_diff.index)  # [450, 460, 470]
    line_strike = math.floor(underlying_price)
    line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
    ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
    plt.title(f"Spread (Call - Put) Volume (logarithm) Heatmap on {date}")
    plt.xlabel("Expiration")
    plt.ylabel("Strike")
    plt.tight_layout()
    fig.savefig(f"heatmap/{date}/volume_heatmap_{date}.png")


def plot_implied_volatility_call_heatmap(
    pivot_call_iv: pd.DataFrame, date: str, underlying_price: float
):
    fig, ax = plt.subplots(figsize=(10, 10))
    vmax = pivot_call_iv.values.max()
    vmin = pivot_call_iv.values.min()
    pivot_call_iv_ = pivot_call_iv.replace(0, np.nan)

    sns.heatmap(pivot_call_iv_, vmin=vmin, vmax=vmax, cmap="jet", ax=ax)
    ax.grid()
    strike_index = list(pivot_call_iv.index)  # [450, 460, 470]
    line_strike = math.floor(underlying_price)
    line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
    ax.axhline(y=line_pos, color="k", linestyle="--", linewidth=2)
    plt.title(f"Call IV Heatmap on {date}")
    plt.xlabel("Expiration")
    plt.ylabel("Strike")
    plt.tight_layout()
    fig.savefig(f"heatmap/{date}/impliedVolatility_Call_heatmap_{date}.png")


def plot_implied_volatility_put_heatmap(
    pivot_put_iv: pd.DataFrame, date: str, underlying_price: float
):
    fig, ax = plt.subplots(figsize=(10, 10))
    vmax = pivot_put_iv.values.max()
    vmin = pivot_put_iv.values.min()
    pivot_put_iv_ = pivot_put_iv.replace(0, np.nan)

    sns.heatmap(pivot_put_iv_, vmin=vmin, vmax=vmax, cmap="jet", ax=ax)
    ax.grid()
    strike_index = list(pivot_put_iv.index)  # [450, 460, 470]
    line_strike = math.floor(underlying_price)
    line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
    ax.axhline(y=line_pos, color="k", linestyle="--", linewidth=2)
    plt.title(f"Put IV Heatmap on {date}")
    plt.xlabel("Expiration")
    plt.ylabel("Strike")
    plt.tight_layout()
    fig.savefig(f"heatmap/{date}/impliedVolatility_Put_heatmap_{date}.png")


def plot_implied_volatility_spread_heatmap(
    pivot_call_iv: pd.DataFrame,
    pivot_put_iv: pd.DataFrame,
    date: str,
    underlying_price: float,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    iv_spread = pivot_call_iv - pivot_put_iv
    iv_spread.replace(0, np.nan, inplace=True)
    vmax = iv_spread.values.max()
    vmin = iv_spread.values.min()

    sns.heatmap(iv_spread, vmin=vmin, vmax=vmax, cmap="jet", ax=ax)
    ax.grid()
    strike_index = list(iv_spread.index)  # [450, 460, 470]
    line_strike = math.floor(underlying_price)
    line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
    ax.axhline(y=line_pos, color="k", linestyle="--", linewidth=2)
    plt.title(f"IV Spread (Call - Put) Heatmap on {date}")
    plt.xlabel("Expiration")
    plt.ylabel("Strike")
    plt.tight_layout()
    fig.savefig(f"heatmap/{date}/impliedVolatility_Spread_heatmap_{date}.png")


def heatmap_pipeline(date: str):
    df = integrate_option_chain(date)

    pivot_oi_diff = create_open_interest_dataframe(df)
    pivot_volume_diff = create_volume_dataframe(df)
    pivot_call_iv, pivot_put_iv, _ = create_implied_volatility_dataframe(df)

    underlying_price = df.head(1)["underlyingPrice_call"].values[0]

    if not os.path.exists(f"heatmap/{date}"):
        os.makedirs(f"heatmap/{date}")

    plot_open_interest_heatmap(pivot_oi_diff, date, underlying_price)
    plot_volume_heatmap(pivot_volume_diff, date, underlying_price)
    plot_implied_volatility_call_heatmap(pivot_call_iv, date, underlying_price)
    plot_implied_volatility_put_heatmap(pivot_put_iv, date, underlying_price)
    plot_implied_volatility_spread_heatmap(
        pivot_call_iv, pivot_put_iv, date, underlying_price
    )
