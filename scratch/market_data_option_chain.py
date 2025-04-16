##
import datetime
import math
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.OptionTrade.BlackScholes import bs_delta, bs_price, implied_volatility

##
date = "2025-04-14"
##
call_df = pd.read_csv("data/2025-04-09/SPY_CALL_2025-04-09_2025-05-16.csv")
put_df = pd.read_csv("data//2025-04-09//SPY_PUT_2025-04-09_2025-05-16.csv")
##
df = pd.merge(call_df, put_df, on="strike", how="inner", suffixes=["_call", "_put"])

##

query_df = df.query(
    "(delta_call>=0.1 & delta_call<=0.55) | (delta_put>-0.55 & delta_put<=-0.1)"
)
##

query_df.loc[:, "openInterestDiff"] = (
    query_df["openInterest_call"] - query_df["openInterest_put"]
)
query_df.loc[:, "volumeDiff"] = query_df["volume_call"] - query_df["volume_put"]
underlying_price = query_df.head(1)["underlyingPrice_put"].values[0]
##

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(query_df["strike"].values, query_df["openInterestDiff"].values)
ax.axhline(y=underlying_price, color="r", ls="--")
ax.set(
    xlabel="Open Interest Difference",
    ylabel="Strike",
    title="Open Interest Difference by Strike Expiration date: 2025-05-16",
)
ax.grid()
fig.tight_layout()
plt.show()
##
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(query_df["strike"].values, query_df["openInterest_call"].values)
ax.axhline(y=underlying_price, color="r", ls="--")

ax.grid()
fig.tight_layout()
plt.show()

##
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(query_df["strike"].values, query_df["volumeDiff"].values)
ax.axhline(y=underlying_price, color="r", ls="--")
ax.set(
    xlabel="Volume Difference",
    ylabel="Strike",
    title="Volume difference by Strike Expiration date: 2025-05-16",
)
ax.grid()
fig.tight_layout()
plt.show()

##
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(
    query_df["strike"].values,
    np.abs(query_df["volumeDiff"].values) - np.abs(query_df["openInterestDiff"].values),
)
ax.axhline(y=underlying_price, color="r", ls="--")
ax.set(
    xlabel="Volume Difference",
    ylabel="Strike",
    title="Volume difference by Strike Expiration date: 2025-05-16",
)
ax.grid()
fig.tight_layout()
plt.show()

##

files = glob(f"data/{date}/SPY_*.csv")
exp_date_list = sorted(set([f.split("_")[3].split(".")[0] for f in files]))

##
merge_df = pd.DataFrame()
for i, exp_date in enumerate(exp_date_list):

    call_df = pd.read_csv(f"data/{date}/SPY_CALL_{date}_{exp_date}.csv")
    put_df = pd.read_csv(f"data/{date}/SPY_PUT_{date}_{exp_date}.csv")
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

underlying_price = merge_df.head(1)["underlyingPrice_put"].values[0]


##
w_vol, w_oi = 1, 1  # 0.2
merge_df.loc[:, "callScore"] = (
    merge_df["volume_call"] * w_vol + merge_df["openInterest_call"] * w_oi
)
merge_df.loc[:, "putScore"] = (
    merge_df["volume_put"] * w_vol + merge_df["openInterest_put"] * w_oi
)
merge_df.loc[:, "score"] = merge_df["callScore"] - merge_df["putScore"]

##
pivot_call = merge_df.pivot(index="strike", columns="exp_date", values="callScore")
pivot_put = merge_df.pivot(index="strike", columns="exp_date", values="putScore")
pivot_call.fillna(0, inplace=True)
pivot_put.fillna(0, inplace=True)
pivot_score = pivot_call - pivot_put
##
pivot_call_oi = merge_df.pivot(
    index="strike", columns="exp_date", values="openInterest_call"
)
pivot_put_oi = merge_df.pivot(
    index="strike", columns="exp_date", values="openInterest_put"
)
pivot_call_oi.fillna(0, inplace=True)
pivot_put_oi.fillna(0, inplace=True)
pivot_oi_diff = pivot_call_oi - pivot_put_oi

pivot_call_volume = merge_df.pivot(
    index="strike", columns="exp_date", values="volume_call"
)
pivot_put_volume = merge_df.pivot(
    index="strike", columns="exp_date", values="volume_put"
)
pivot_call_volume.fillna(0, inplace=True)
pivot_put_volume.fillna(0, inplace=True)
pivot_volume_diff = pivot_call_volume - pivot_put_volume

pivot_call_iv = merge_df.pivot(index="strike", columns="exp_date", values="iv_call")
pivot_put_iv = merge_df.pivot(index="strike", columns="exp_date", values="iv_put")
pivot_call_iv.fillna(0, inplace=True)
pivot_put_iv.fillna(0, inplace=True)


##
fig, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(pivot_put, ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_put.index)
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Put Weighted Score Heatmap (Volume + OI)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()
##
fig, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(pivot_call, ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_call.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Call Weighted Score Heatmap (Volume + OI)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()
##
pivot_call

##
fig, ax = plt.subplots(figsize=(10, 10))
vmax = pivot_score.values.max()
vmin = pivot_score.values.min()
avmax = vmax if np.abs(vmax) > np.abs(vmin) else np.abs(vmin)
avmin = np.abs(vmax) if np.abs(vmax) < np.abs(vmin) else np.abs(vmin)

hm = sns.heatmap(pivot_score, vmin=-1 * avmin, vmax=avmin, cmap="RdBu_r", ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_score.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Call vs Put Weighted Score Heatmap (Volume + OI)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()

##
mask = pivot_score < 0
log_pivot_score = np.log(np.abs(pivot_score))
log_pivot_score.replace([np.inf, -np.inf], 0, inplace=True)
log_pivot_score[mask] = -1 * log_pivot_score[mask]
fig, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(log_pivot_score, cmap="RdBu_r", ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_score.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Call vs Put Weighted Score Heatmap (Volume + OI)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()

##

flights = sns.load_dataset("flights")
flights = flights.pivot(index="month", columns="year", values="passengers")
ax = sns.heatmap(flights, cbar=False)
ax.hlines([3], *ax.get_xlim())

plt.show()
##
vmax = pivot_oi_diff.values.max()
vmin = pivot_oi_diff.values.min()
avmax = vmax if np.abs(vmax) > np.abs(vmin) else np.abs(vmin)
avmin = np.abs(vmax) if np.abs(vmax) < np.abs(vmin) else np.abs(vmin)

fig, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(pivot_oi_diff, vmin=-1 * avmin, vmax=avmin, cmap="RdBu_r", ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_put.index)
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Call - Put Open Interest Heatmap)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()
##
vmax = pivot_volume_diff.values.max()
vmin = pivot_volume_diff.values.min()
avmax = vmax if np.abs(vmax) > np.abs(vmin) else np.abs(vmin)
avmin = np.abs(vmax) if np.abs(vmax) < np.abs(vmin) else np.abs(vmin)

fig, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(pivot_volume_diff, vmin=-1 * avmin, vmax=avmin, cmap="RdBu_r", ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_put.index)
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Call - Put Volume Heatmap)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()
##
mask = pivot_oi_diff < 0
log_pivot_oi_diff = np.log(np.abs(pivot_oi_diff))
log_pivot_oi_diff.replace([np.inf, -np.inf], 0, inplace=True)
log_pivot_oi_diff[mask] = -1 * log_pivot_oi_diff[mask]
fig, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(log_pivot_oi_diff, cmap="RdBu_r", ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_score.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Call - Put Open Interest (logarithm) Heatmap)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()

##
mask = pivot_volume_diff < 0
log_pivot_volume_diff = np.log(np.abs(pivot_volume_diff))
log_pivot_volume_diff.replace([np.inf, -np.inf], 0, inplace=True)
log_pivot_volume_diff[mask] = -1 * log_pivot_volume_diff[mask]
fig, ax = plt.subplots(figsize=(10, 10))
hm = sns.heatmap(log_pivot_volume_diff, cmap="RdBu_r", ax=ax)
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_score.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="blue", linestyle="--", linewidth=2)
plt.title("Call - Put Volume (logarithm) Heatmap)")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()

##

fig, ax = plt.subplots(figsize=(10, 10))
vmax = pivot_call_iv.values.max()
vmin = pivot_call_iv.values.min()
pivot_call_iv_ = pivot_call_iv.replace(0, np.nan)

hm = sns.heatmap(pivot_call_iv_, vmin=vmin, vmax=vmax, cmap="jet", ax=ax)
ax.grid()
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_score.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="k", linestyle="--", linewidth=2)
plt.title("Call IV Heatmap")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()
##
fig, ax = plt.subplots(figsize=(10, 10))
vmax = pivot_put_iv.values.max()
vmin = pivot_put_iv.values.min()
pivot_put_iv_ = pivot_put_iv.replace(0, np.nan)

hm = sns.heatmap(pivot_put_iv_, vmin=vmin, vmax=vmax, cmap="jet", ax=ax)
ax.grid()
xmin, xmax = hm.get_xlim()
strike_index = list(pivot_score.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="k", linestyle="--", linewidth=2)
plt.title("Put IV Heatmap")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()
##
fig, ax = plt.subplots(figsize=(10, 10))
iv_spread = pivot_call_iv - pivot_put_iv
iv_spread.replace(0, np.nan, inplace=True)
vmax = iv_spread.values.max()
vmin = iv_spread.values.min()

hm = sns.heatmap(iv_spread, vmin=vmin, vmax=vmax, cmap="jet", ax=ax)
ax.grid()
xmin, xmax = hm.get_xlim()
strike_index = list(iv_spread.index)  # [450, 460, 470]
line_strike = math.floor(underlying_price)
line_pos = strike_index.index(line_strike) + 0.5  # セルの中央に引くため +0.5
ax.axhline(y=line_pos, color="k", linestyle="--", linewidth=2)
plt.title("Call - Put IV Heatmap")
plt.xlabel("Expiration")
plt.ylabel("Strike")
plt.tight_layout()
plt.show()
##
df = pd.read_csv("data/2025-04-12/SPY_PUT_2025-04-12_2025-04-14.csv")
data = df.loc[100]
underlying_price = data["underlyingPrice"]
expiration_date = "2025-04-14"
strike = data["strike"]
t = (
    datetime.datetime.strptime(expiration_date, "%Y-%m-%d")
    - datetime.datetime(2025, 4, 11)
).days / 365
r = 0.045
option_type = "put"
option_price = data["mid"]
iv = implied_volatility(underlying_price, strike, t, r, option_price, option_type)
delta = bs_delta(underlying_price, strike, t, r, iv, option_type)
print(f"Strike: {strike}, IV: {iv:.3f}, delta: {delta:.3f}")
##
