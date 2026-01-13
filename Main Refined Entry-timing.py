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
        "csv": "US500_1H_2012-2020.csv",
    },
]


# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5
SLIPPAGE_POINTS = 0.5
# Spread och kommission uttryckt i samma enhet som priset i din CSV (points)
FIXED_SPREAD_POINTS = 0.8
COMM_POINTS_PER_SIDE = 0.05  # per side (entry eller exit)

ADX_THRESHOLD = 15          # du testade 20
BREAKOUT_N = 5             # testintervall 3,5,7,10


def commission_round_turn_points():
    """Kommission per round-turn (entry+exit) i points."""
    return 2.0 * COMM_POINTS_PER_SIDE

def run_backtest_for_market(market_name: str, csv_path: str):
    print("\n" + "="*70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("="*70 + "\n")

    # ==========================
    # 1. Ladda data
    # ==========================

    df = pd.read_csv(csv_path)

    # Anpassa kolumnnamn om de skiljer sig
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()

    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    # Ta reda på minsta pris-enhet (för att bestämma pip_size)
    '''
    diffs = df["close"].diff().abs()
    tick_est = diffs[diffs > 0].quantile(0.01)  # robust: ignorerar outliers
    print("Estimated min step ~", tick_est)'''


    # ==========================
    # 2. Indikatorer
    # ==========================

    # EMA (snabb)
    df['ema_fast'] = df['close'].ewm(span=70, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=120, adjust=False).mean()

    # ADX (14) via ta
    adx_len = 14
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=adx_len,
        fillna=False
    ).adx()

    # Highest high för breakout-entry (N bars tillbaka, exkluderar nuvarande bar)
    df["hh_N"] = df["high"].shift(1).rolling(BREAKOUT_N).max()

    USE_SPREAD_COLUMN = 'spread_points' in df.columns

    def get_spread_points(row):
        if USE_SPREAD_COLUMN:
            return float(row['spread_points'])
        return FIXED_SPREAD_POINTS

    # ==========================
    # 4. Backtest-loop
    # ==========================

    trades = []

    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None


    idx_list = df.index.to_list()

    trend_state = "NEUTRAL"  # "BULL" eller "NEUTRAL"
    entry_armed = False  # armar entry efter cross, avväpnar efter entry

    for i in range(1, len(df) - 1):
        ts = idx_list[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]
        # ======================
        # Om vi redan är i trade: kolla SL/TP
        # ======================
        if in_position:
            exit_price = None
            exit_reason = None

            if row["ema_fast"] <= row["ema_medium"]:
                spread = get_spread_points(next_row)
                exit_price = next_row["open"] - HALF * spread - SLIPPAGE_POINTS
                exit_reason = 'ema_touch_exit'

            if exit_price is not None:
                exit_time = idx_list[i + 1]
                if pos_direction == 'LONG':
                    pnl = (exit_price - entry_price) - commission_round_turn_points()

                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Direction': pos_direction,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Exit Reason': exit_reason,
                    'pnl': pnl,
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None


            # Om vi fortfarande är i position -> hoppa entrylogik
        if in_position:
            continue

        ema_fast = row['ema_fast']
        prev_ema_fast = prev_row['ema_fast']
        prev_ema_medium = prev_row['ema_medium']
        ema_medium = row['ema_medium']
        adx = row["adx"]

        # Hoppa om ADX inte är "ready"
        if np.isnan(adx):
            continue

        if np.isnan(ema_medium):
            continue

        # ======================
        # 1) Trend-state via EMA-cross (regimtrigger)
        # ======================
        cross_up = (prev_ema_fast < prev_ema_medium) and (ema_fast > ema_medium)
        cross_down = (prev_ema_fast > prev_ema_medium) and (ema_fast < ema_medium)

        # Om cross upp: aktivera bullish state och arma en entry
        if cross_up:
            trend_state = "BULL"
            entry_armed = True

        # Om cross ner: stäng state och disarma
        if cross_down:
            trend_state = "NEUTRAL"
            entry_armed = False

        # ======================
        # 2) Trendkvalitet (ADX-filter)
        # ======================
        adx_filter = adx > ADX_THRESHOLD

        # ======================
        # 3) Entry-timing: breakout/continuation efter cross
        # close > highest(high, N) där hh_N redan är shiftad och rullad
        # ======================
        hh_N = row.get("hh_N", np.nan)
        breakout_ok = (not np.isnan(hh_N)) and (row["close"] > hh_N)

        long_entry_signal = (
                (trend_state == "BULL")
                and entry_armed
                and adx_filter
                and breakout_ok
        )

        # EN trade åt gången
        if long_entry_signal:
            pos_direction = 'LONG'
            entry_time = ts

            next_open = next_row['open']
            spread = get_spread_points(next_row)
            entry_price = next_open + HALF * spread + SLIPPAGE_POINTS  # LONG: köp på ask

            in_position = True

            entry_armed = False  # max 1 trade per trend_state
    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()

    # --- Extra statistik ---
    trades_df["is_win"] = trades_df["pnl"] > 0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()  # negativt

    winrate = trades_df["is_win"].mean()

    # Expectancy per trade
    expectancy = trades_df["pnl"].mean()

    # Drawdown
    roll_max = trades_df["equity"].cummax()
    dd = trades_df["equity"] - roll_max
    max_dd = dd.min()  # negativt
    max_dd_points = abs(max_dd)  # positivt för rapportering

    # Longest losing streak (räknat i trades)
    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    # “Sharpe” på trade-nivå (inte tidsnormaliserad)
    pnl_std = trades_df["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    stats = {
        "Market": market_name,
        "Trades": int(len(trades_df)),
        "Total PnL (points)": float(trades_df["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
        'Spread (points)': float(spread),
        'Commission RT (points)': float(commission_round_turn_points()),
    }

    print("\n--- STATS ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # PLOT (som du redan får)
    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Exit Time"], trades_df["equity"])
    plt.title(f"Equity curve - {market_name}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (points)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return stats, trades_df

# ==========================
# KÖR BACKTEST + SLUTSUMMERING + COMBINED EQUITY & STATS
# ==========================

all_results = []
all_trades = []

for m in markets:
    try:
        stats, trades_df = run_backtest_for_market(
            m["name"],
            m["csv"],
        )
        if stats is not None and trades_df is not None:
            trades_df["Market"] = m["name"]
            all_results.append(stats)
            all_trades.append(trades_df)
    except Exception as e:
        print(f"\n*** FEL för {m['name']} ({m['csv']}): {e}\n")