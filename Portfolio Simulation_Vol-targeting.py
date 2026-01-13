import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from math import sqrt
import os

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {
        "name": "US500",
        "csv": "US500_1H_2012-now.csv",
    },
    {
        "name": "USTEC",
        "csv": "USTEC_1H_2012-now.csv",
    },
    {
        "name": "US30",
        "csv": "US30_1H_2012-now.csv",
    },
]

# ==========================
# PORTFÖLJ
# ==========================
START_CAPITAL = 50_000
MAX_GROSS_EXPOSURE = 1.25         # max 100% av equity i öppna positionsvärden
EQUAL_WEIGHT = True              # start: lika vikt
TARGET_GROSS_EXPOSURE = 1.25     # t.ex 1.0 = försök investera fullt när möjligt


# ==========================
# INSTRUMENT: $ per indexpunkt per kontrakt
# Om CFD $1/point: lämna 1.0 på alla.
# ==========================
POINT_VALUE = {
    "US500": 1.0,
    "USTEC": 1.0,
    "US30":  1.0,
}


# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5
SLIPPAGE_POINTS = 0.5
# Spread och kommission uttryckt i samma enhet som priset i din CSV (points)
FIXED_SPREAD_POINTS = 0.8
COMM_POINTS_PER_SIDE = 0.05  # per side (entry eller exit)

def commission_round_turn_points():
    """Kommission per round-turn (entry+exit) i points."""
    return 2.0 * COMM_POINTS_PER_SIDE

# ==========================
# VOL TARGETING (Carver-style)
# ==========================
USE_VOL_TARGETING = True

TARGET_VOL_ANN = 0.10       # t.ex. 10% annualiserad målvol
VOL_WINDOW_DAYS = 60        # 20/60/120 är vanliga
MIN_SCALE = 0.25            # sänk inte under 25% av basexponering
MAX_SCALE = 2.00            # öka inte över 200% (om ni tillåter leverage)
TRADING_DAYS_PER_YEAR = 252

# ============================================================
# SINGLE-MARKET BACKTEST
# ============================================================

def load_market_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime' i CSV.")

    df = df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla: {required_cols}")

    return df


def generate_trades_for_market(
    market_name: str,
    df: pd.DataFrame,
    exit_confirm_bars: int = 10,
    adx_threshold: float = 15,
    ema_fast_len: int = 70,
    ema_slow_len: int = 120,
) -> pd.DataFrame:

    df = df.copy()

    # Indicators
    df["ema_fast"] = df["close"].ewm(span=ema_fast_len, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_len, adjust=False).mean()

    adx_len = 14
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"],
        window=adx_len, fillna=False
    ).adx()

    use_spread_col = "spread_points" in df.columns

    def spread_points(row) -> float:
        return float(row["spread_points"]) if use_spread_col else FIXED_SPREAD_POINTS

    trades = []

    in_position = False
    entry_price = None
    entry_signal_time = None
    entry_fill_time = None
    exit_breach_count = 0

    idx = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        # Skip if indicators not ready
        if np.isnan(row["adx"]) or np.isnan(row["ema_slow"]) or np.isnan(prev_row["ema_slow"]):
            continue

        # --- Manage exit ---
        if in_position:
            # delayed recross confirmation
            if row["ema_fast"] <= row["ema_slow"]:
                exit_breach_count += 1
            else:
                exit_breach_count = 0

            if exit_breach_count >= exit_confirm_bars:
                spr = spread_points(next_row)
                exit_fill_price = float(next_row["open"] - HALF * spr - SLIPPAGE_POINTS)  # sell on bid
                exit_fill_time = idx[i + 1]

                trades.append({
                    "Market": market_name,
                    "Direction": "LONG",
                    "Entry Signal Time": entry_signal_time,
                    "Entry Fill Time": entry_fill_time,
                    "Exit Fill Time": exit_fill_time,
                    "Entry Price": float(entry_price),
                    "Exit Price": float(exit_fill_price),
                    "Exit Reason": f"ema_recross_{exit_confirm_bars}bars",
                    "Comm RT (points)": float(commission_round_turn_points()),
                })

                in_position = False
                entry_price = None
                entry_signal_time = None
                entry_fill_time = None
                exit_breach_count = 0

            if in_position:
                continue

        # --- Entry logic ---
        adx_ok = row["adx"] > adx_threshold
        cross_up = (prev_row["ema_fast"] < prev_row["ema_slow"]) and (row["ema_fast"] > row["ema_slow"])

        if cross_up and adx_ok:
            spr = spread_points(next_row)
            entry_fill_price = float(next_row["open"] + HALF * spr + SLIPPAGE_POINTS)  # buy on ask

            in_position = True
            entry_signal_time = ts
            entry_fill_time = idx[i + 1]
            entry_price = entry_fill_price
            exit_breach_count = 0

    # Forced exit at end (important)
    if in_position:
        last_row = df.iloc[-1]
        spr = float(last_row["spread_points"]) if "spread_points" in df.columns else FIXED_SPREAD_POINTS
        exit_fill_price = float(last_row["close"] - HALF * spr - SLIPPAGE_POINTS)
        exit_fill_time = df.index[-1]

        trades.append({
            "Market": market_name,
            "Direction": "LONG",
            "Entry Signal Time": entry_signal_time,
            "Entry Fill Time": entry_fill_time,
            "Exit Fill Time": exit_fill_time,
            "Entry Price": float(entry_price),
            "Exit Price": float(exit_fill_price),
            "Exit Reason": "forced_exit_end_of_test",
            "Comm RT (points)": float(commission_round_turn_points()),
        })

    return pd.DataFrame(trades)

def build_portfolio_mtm_cash(
    market_dfs: dict,
    trades_df: pd.DataFrame,
    start_capital: float,
    max_gross_exposure: float = 1.0,
    target_gross_exposure: float = 1.0,
    weights: dict | None = None,
    use_vol_targeting: bool = False,
    target_vol_ann: float = 0.10,
    vol_window_days: int = 60,
    min_scale: float = 0.25,
    max_scale: float = 2.0,
    trading_days_per_year: int = 252,
) -> tuple:

    tr = trades_df.copy()
    tr["Entry Fill Time"] = pd.to_datetime(tr["Entry Fill Time"])
    tr["Exit Fill Time"] = pd.to_datetime(tr["Exit Fill Time"])

    # Union time index
    all_index = None
    for mkt, df in market_dfs.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")]
        all_index = dfx.index if all_index is None else all_index.union(dfx.index)

    all_index = pd.DatetimeIndex(all_index.sort_values().unique())

    # Close matrix (ffill)
    closes = pd.DataFrame(index=all_index)
    for mkt, df in market_dfs.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")]
        closes[mkt] = dfx["close"].reindex(all_index).ffill()

    # Equal weights over markets (based on configured markets list)
    mkts = sorted(market_dfs.keys())
    n = len(mkts)
    mkts = sorted(market_dfs.keys())
    n = len(mkts)

    if weights is None:
        weights = {m: 1.0 / n for m in mkts}
    else:
        # säkerställ att alla marknader finns och normalisera
        missing = set(mkts) - set(weights.keys())
        if missing:
            raise ValueError(f"weights saknar marknader: {missing}")
        s = sum(weights[m] for m in mkts)
        if s <= 0:
            raise ValueError("weights summerar till 0 eller mindre.")
        weights = {m: weights[m] / s for m in mkts}

    print("\n[Portfolio] Using weights:")
    for k in sorted(weights.keys()):
        print(f"  {k}: {weights[k]:.4f}")

    # Group trades by entry/exit time
    entries = tr.sort_values(["Entry Fill Time", "Market"]).groupby("Entry Fill Time")
    exits = tr.sort_values(["Exit Fill Time", "Market"]).groupby("Exit Fill Time")

    # Positions: per market
    # store: contracts, entry_price, entry_notional
    positions = {}  # mkt -> dict

    cash = float(start_capital)
    equity_path = []
    gross_exposure_path = []
    open_positions_path = []

    # ==========================
    # VOL TARGET STATE (daily)
    # ==========================
    current_scale = 1.0
    daily_equity_last = None
    prev_day = None
    daily_returns_window = []  # list of daily returns (float)

    scale_path = []  # (ts, scale) för plotting/debug

    for ts in all_index:
        # ==========================
        # DAILY VOL UPDATE (once per day)
        # ==========================
        day = ts.date()
        if prev_day is None:
            prev_day = day
            # init daily_equity_last efter första MTM-beräkningen senare
        elif day != prev_day:
            # Vi har gått in i en ny dag. Uppdatera daily return från gårdagens sista equity.
            if daily_equity_last is not None and len(equity_path) > 0:
                # equity vid sista timestampen från föregående dag
                prev_equity = equity_path[-1][1]
                if daily_equity_last > 0:
                    r = (prev_equity / daily_equity_last) - 1.0
                    daily_returns_window.append(float(r))

                    # håll window-längden
                    if len(daily_returns_window) > vol_window_days:
                        daily_returns_window = daily_returns_window[-vol_window_days:]

                    # beräkna scale
                    if use_vol_targeting and len(daily_returns_window) >= max(10, min(20, vol_window_days)):
                        vol_daily = np.std(daily_returns_window, ddof=1)
                        vol_ann = vol_daily * np.sqrt(trading_days_per_year) if vol_daily > 0 else np.nan

                        if vol_ann and np.isfinite(vol_ann) and vol_ann > 0:
                            raw_scale = target_vol_ann / vol_ann
                            current_scale = float(np.clip(raw_scale, min_scale, max_scale))
                        else:
                            current_scale = 1.0

            # reset for new day
            daily_equity_last = equity_path[-1][1] if len(equity_path) > 0 else daily_equity_last
            prev_day = day
        # --- 1) Exits first ---
        if ts in exits.groups:
            block = exits.get_group(ts)
            for _, t in block.iterrows():
                mkt = t["Market"]
                if mkt not in positions:
                    continue

                pos = positions[mkt]
                contracts = pos["contracts"]
                entry_price = pos["entry_price"]
                exit_price = float(t["Exit Price"])
                pv = POINT_VALUE[mkt]

                # Notional values ($)
                exit_notional = contracts * exit_price * pv

                # PnL ($) from price move: (exit-entry) * contracts * pv
                pnl_gross = (exit_price - entry_price) * contracts * pv

                # Commission ($): round-turn points * pv * contracts
                comm_points = float(t.get("Comm RT (points)", 0.0))
                pnl_comm = comm_points * pv * contracts

                pnl_net = pnl_gross - pnl_comm

                # Realize cash: receive exit notional, pay commission
                cash += exit_notional
                cash -= pnl_comm

                del positions[mkt]

        # --- 2) Entries ---
        if ts in entries.groups:
            block = entries.get_group(ts)

            # Compute current MTM equity and gross exposure BEFORE placing new entries
            mtm_value = 0.0
            gross_exposure_value = 0.0
            for pmkt, pos in positions.items():
                px = float(closes.loc[ts, pmkt])
                pv = POINT_VALUE[pmkt]
                notional = pos["contracts"] * px * pv
                mtm_value += notional
                gross_exposure_value += notional

            equity_now = cash + mtm_value
            gross_pct_now = (gross_exposure_value / equity_now) if equity_now > 0 else 0.0

            for _, t in block.iterrows():
                mkt = t["Market"]
                if mkt in positions:
                    continue

                entry_price = float(t["Entry Price"])
                pv = POINT_VALUE[mkt]

                # Recompute available capacity (after prior entries this bar)
                mtm_value = 0.0
                gross_exposure_value = 0.0
                for pmkt, pos in positions.items():
                    px = float(closes.loc[ts, pmkt])
                    pv2 = POINT_VALUE[pmkt]
                    notional = pos["contracts"] * px * pv2
                    mtm_value += notional
                    gross_exposure_value += notional

                equity_now = cash + mtm_value
                gross_pct_now = (gross_exposure_value / equity_now) if equity_now > 0 else 0.0
                remaining_capacity_pct = max(0.0, max_gross_exposure - gross_pct_now)

                effective_target_exposure = target_gross_exposure
                if use_vol_targeting:
                    effective_target_exposure = target_gross_exposure * current_scale

                desired_notional = equity_now * effective_target_exposure * weights.get(mkt, 0.0)

                # Cap by remaining gross exposure
                cap_notional = equity_now * remaining_capacity_pct
                position_notional = min(desired_notional, cap_notional)

                # No leverage: cannot spend more cash than available
                position_notional = min(position_notional, cash)

                # Convert notional to contracts: notional = contracts * price * pv
                denom = entry_price * pv
                contracts = (position_notional / denom) if denom > 0 else 0.0

                if contracts <= 0:
                    continue

                # Spend cash
                cash -= contracts * entry_price * pv

                positions[mkt] = {
                    "contracts": contracts,
                    "entry_price": entry_price,
                }

        # --- 3) MTM at close ---
        mtm_value = 0.0
        gross_exposure_value = 0.0
        for pmkt, pos in positions.items():
            px = float(closes.loc[ts, pmkt])
            pv = POINT_VALUE[pmkt]
            notional = pos["contracts"] * px * pv
            mtm_value += notional
            gross_exposure_value += notional

        equity = cash + mtm_value
        equity_path.append((ts, equity))

        # init daily_equity_last första gången vi får ett equity-värde
        if daily_equity_last is None:
            daily_equity_last = equity
        scale_path.append((ts, current_scale))

        open_positions_path.append((ts, len(positions)))
        gross_exposure_path.append((ts, gross_exposure_value / equity if equity > 0 else 0.0))

    equity_series = pd.Series(
        [v for _, v in equity_path],
        index=pd.DatetimeIndex([t for t, _ in equity_path]),
        name="PortfolioEquity"
    )

    open_pos_series = pd.Series(
        [v for _, v in open_positions_path],
        index=pd.DatetimeIndex([t for t, _ in open_positions_path]),
        name="OpenPositions"
    )

    gross_exposure_series = pd.Series(
        [v for _, v in gross_exposure_path],
        index=pd.DatetimeIndex([t for t, _ in gross_exposure_path]),
        name="GrossExposurePct"
    )

    # Daily returns
    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    scale_series = pd.Series(
        data=[v for _, v in scale_path],
        index=pd.DatetimeIndex([t for t, _ in scale_path]),
        name="ExposureScale"
    )

    return equity_series, daily_returns, open_pos_series, gross_exposure_series, scale_series

def portfolio_metrics_from_equity(equity_series: pd.Series, daily_returns: pd.Series, trading_days=252) -> dict:
    # CAGR
    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    # Max DD
    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    max_dd = dd.min()

    # Sharpe ann.
    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(trading_days) if sd and sd > 0 else np.nan

    calmar = (cagr / abs(max_dd)) if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        "Equity Start": float(equity_series.iloc[0]),
        "Equity End": float(equity_series.iloc[-1]),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "Max Drawdown %": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        "Sharpe (ann.)": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "Avg Daily Return": float(mu) if pd.notna(mu) else np.nan,
        "Daily Vol": float(sd) if pd.notna(sd) else np.nan,
    }

def erc_weights(cov: np.ndarray, tol=1e-10, max_iter=50_000):
    """
    Equal Risk Contribution weights via multiplicative updates.
    Input: covariance matrix (numpy array)
    Output: weights that sum to 1
    """
    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(max_iter):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n

        # convergence check
        if np.max(np.abs(rc - target)) < tol:
            break

        # multiplicative update
        w = w * (target / np.maximum(rc, 1e-16))
        w = w / w.sum()

    return w

def build_daily_close_returns(market_dfs: dict) -> pd.DataFrame:
    """
    Skapar en DataFrame med daily close returns per marknad.
    Index: datum
    Kolumner: marknadsnamn
    """
    daily_closes = []
    for mkt, df in market_dfs.items():
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()

        # daily last close
        daily = s.resample("1D").last().dropna()
        daily = daily.rename(mkt)
        daily_closes.append(daily)

    closes_df = pd.concat(daily_closes, axis=1).dropna()
    returns_df = closes_df.pct_change().dropna()
    return returns_df


def build_strategy_daily_returns_per_market(
    market_dfs: dict,
    portfolio_trades: pd.DataFrame,
    start_capital: float,
) -> pd.DataFrame:
    """
    Returnerar en DataFrame med dagliga % returns för STRATEGIN per marknad.
    Varje marknad simuleras standalone (vikten=1) med samma backtestmotor.
    """
    rets = {}
    for mkt, df in market_dfs.items():
        trades_mkt = portfolio_trades[portfolio_trades["Market"] == mkt].copy()
        if trades_mkt.empty:
            continue

        eq, daily_ret, _, _, _ = build_portfolio_mtm_cash(
            market_dfs={mkt: df},
            trades_df=trades_mkt,
            start_capital=start_capital,
            max_gross_exposure=1.0,
            target_gross_exposure=1.0,
            weights={mkt: 1.0},
            use_vol_targeting=False,  # viktigt: returns för strategin utan vol-target
        )
        rets[mkt] = daily_ret.rename(mkt)

    returns_df = pd.concat(rets.values(), axis=1).dropna()
    return returns_df

def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving Block Bootstrap: bygger en sekvens av index med block av längd block_len.
    """
    if block_len < 1:
        raise ValueError("block_len måste vara >= 1")

    out = []
    while len(out) < n:
        start = rng.integers(0, n)
        block = [(start + j) % n for j in range(block_len)]
        out.extend(block)
    return np.array(out[:n], dtype=int)

def metrics_from_daily_returns(daily_returns: pd.Series, start_equity: float = 1.0, trading_days: int = 252) -> dict:
    # equity curve (compounded)
    equity = (1.0 + daily_returns).cumprod() * start_equity

    # CAGR
    n_days = (equity.index[-1] - equity.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    # Max drawdown
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min()

    # Sharpe
    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(trading_days) if sd and sd > 0 else np.nan

    calmar = (cagr / abs(max_dd)) if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "MaxDD": float(max_dd) if pd.notna(max_dd) else np.nan,  # negativt tal
        "Sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "EndEquity": float(equity.iloc[-1]),
    }

def bootstrap_erc_portfolio(
    strategy_returns_df: pd.DataFrame,   # columns = markets, rows = daily returns
    n_iter: int = 5000,
    block_len: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    För varje iteration:
      1) block-resample av daily strategy returns
      2) ERC weights på sample covariance
      3) portfolio returns = R @ w
      4) metrics på portfolio returns
    Returnerar DataFrame med metrics + vikter per iteration.
    """
    rng = np.random.default_rng(seed)

    markets = list(strategy_returns_df.columns)
    n = len(strategy_returns_df)
    if n < 2 * block_len:
        raise ValueError("För få datapunkter relativt block_len. Sänk block_len eller använd mer data.")

    results = []

    base_index = strategy_returns_df.index

    for k in range(n_iter):
        idx = block_bootstrap_indices(n=n, block_len=block_len, rng=rng)

        sample = strategy_returns_df.iloc[idx].copy()
        # behåll datumindex bara för snygg equity; ordningen representerar resamplet
        sample.index = base_index[:len(sample)]

        cov = sample.cov().values
        w = erc_weights(cov)
        w = np.array(w, dtype=float)
        w = w / w.sum()

        port_ret = pd.Series(sample.values @ w, index=sample.index, name="port_ret")

        m = metrics_from_daily_returns(port_ret)

        row = {
            "iter": k,
            "Sharpe": m["Sharpe"],
            "CAGR": m["CAGR"],
            "MaxDD": m["MaxDD"],
            "Calmar": m["Calmar"],
            "EndEquity": m["EndEquity"],
        }

        for i, mk in enumerate(markets):
            row[f"w_{mk}"] = float(w[i])

        results.append(row)

    return pd.DataFrame(results)

# ==========================
# MAIN
# ==========================
market_dfs = {}
all_trades = []

for m in markets:
    name = m["name"]
    df = load_market_df(m["csv"])
    market_dfs[name] = df

    tdf = generate_trades_for_market(
        market_name=name,
        df=df,
        exit_confirm_bars=10,
        adx_threshold=15,
        ema_fast_len=70,
        ema_slow_len=120,
    )

    if not tdf.empty:
        all_trades.append(tdf)

if not all_trades:
    raise RuntimeError("Inga trades genererades för någon marknad.")

portfolio_trades = pd.concat(all_trades, ignore_index=True)

# ==========================
# ERC WEIGHTS (DAILY CLOSE RETURNS)
# ==========================
returns_df = build_daily_close_returns(market_dfs)
cov = returns_df.cov().values
w = erc_weights(cov)

erc_weights_dict = dict(zip(returns_df.columns, w))

print("\n--- ERC weights (based on daily close returns) ---")
for k, v in erc_weights_dict.items():
    print(f"{k}: {v:.4f}")

# 1) Bygg dagliga STRATEGI-returns per marknad
strategy_returns_df = build_strategy_daily_returns_per_market(
    market_dfs=market_dfs,
    portfolio_trades=portfolio_trades,
    start_capital=START_CAPITAL,
)

print("\nStrategy returns DF shape:", strategy_returns_df.shape)
print(strategy_returns_df.describe())

# 2) Bootstrap + ERC per resample
boot = bootstrap_erc_portfolio(
    strategy_returns_df=strategy_returns_df,
    n_iter=5000,
    block_len=20,   # typiskt 20-60. Börja 20.
    seed=42,
)

# 3) Sammanfatta distributioner
def summarize(series: pd.Series):
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p05": float(series.quantile(0.05)),
        "p95": float(series.quantile(0.95)),
    }

print("\n--- BOOTSTRAP DISTRIBUTIONS (ERC each resample) ---")
for col in ["Sharpe", "CAGR", "MaxDD", "Calmar", "EndEquity"]:
    s = summarize(boot[col].dropna())
    print(col, s)

print("\n--- WEIGHT DISTRIBUTIONS ---")
for mk in strategy_returns_df.columns:
    s = summarize(boot[f"w_{mk}"])
    print(mk, s)

# (valfritt) plot histogram för Sharpe och MaxDD
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.hist(boot["Sharpe"].dropna(), bins=50)
plt.title("Bootstrap distribution of Sharpe (ERC each resample)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.hist(boot["MaxDD"].dropna(), bins=50)
plt.title("Bootstrap distribution of Max Drawdown (fraction, negative)")
plt.grid(True)
plt.tight_layout()
plt.show()

equity_series, daily_returns, open_pos_series, gross_exposure_series, scale_series = build_portfolio_mtm_cash(
    market_dfs=market_dfs,
    trades_df=portfolio_trades,
    start_capital=START_CAPITAL,
    max_gross_exposure=MAX_GROSS_EXPOSURE,
    target_gross_exposure=TARGET_GROSS_EXPOSURE,
    weights=erc_weights_dict,
    use_vol_targeting=USE_VOL_TARGETING,
    target_vol_ann=TARGET_VOL_ANN,
    vol_window_days=VOL_WINDOW_DAYS,
    min_scale=MIN_SCALE,
    max_scale=MAX_SCALE,
    trading_days_per_year=TRADING_DAYS_PER_YEAR,
)

# equity_series: MTM equity per timestamp (1H)
daily_equity = equity_series.resample("1D").last().dropna()
daily_pnl = daily_equity.diff().dropna()                    # $ per dag
daily_pnl_pct = daily_equity.pct_change().dropna()          # % per dag

max_daily_loss_usd = daily_pnl.min()        # negativt värde
max_daily_loss_pct = daily_pnl_pct.min()    # negativt värde

print("Max daily loss ($), close-to-close:", float(max_daily_loss_usd))
print("Max daily loss (%), close-to-close:", float(max_daily_loss_pct * 100))

# Grupp per kalenderdag
g = equity_series.groupby(equity_series.index.date)

daily_intraday_dd_pct = []
daily_intraday_dd_usd = []

for d, s in g:
    s = s.dropna()
    if len(s) < 2:
        continue

    peak = s.cummax()
    dd_usd = s - peak                       # <= 0
    dd_pct = (s / peak) - 1.0               # <= 0

    daily_intraday_dd_usd.append(dd_usd.min())
    daily_intraday_dd_pct.append(dd_pct.min())

max_intraday_daily_loss_usd = float(np.min(daily_intraday_dd_usd))
max_intraday_daily_loss_pct = float(np.min(daily_intraday_dd_pct))

print("Max intraday daily loss ($):", max_intraday_daily_loss_usd)
print("Max intraday daily loss (%):", max_intraday_daily_loss_pct * 100)

# Hitta datumet för värsta intraday-DD
worst_day = None
worst_dd = 0.0

for d, s in g:
    s = s.dropna()
    if len(s) < 2:
        continue
    dd = (s / s.cummax()) - 1.0
    day_worst = float(dd.min())
    if day_worst < worst_dd:
        worst_dd = day_worst
        worst_day = d

print("Worst intraday DD day:", worst_day)
print("Worst intraday DD (%):", worst_dd * 100)


metrics = portfolio_metrics_from_equity(equity_series, daily_returns)

print("\n--- PORTFÖLJ METRICS (CASH, MTM, EQUAL WEIGHT) ---")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print("\nSanity: Open positions at end:", int(open_pos_series.iloc[-1]))
print("Avg gross exposure %:", float(gross_exposure_series.mean() * 100.0))
print("Max gross exposure %:", float(gross_exposure_series.max() * 100.0))

# Plots
plt.figure(figsize=(12, 5))
plt.plot(equity_series.index, equity_series.values)
plt.title("Portfolio Equity (Cash MTM, Equal Weight)")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(open_pos_series.index, open_pos_series.values)
plt.title("Open Positions")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(gross_exposure_series.index, gross_exposure_series.values)
plt.title("Gross Exposure %")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(scale_series.index, scale_series.values)
plt.title("Exposure Scale (Vol Targeting)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Scale median:", float(scale_series.median()))
print("Scale p05:", float(scale_series.quantile(0.05)))
print("Scale p95:", float(scale_series.quantile(0.95)))