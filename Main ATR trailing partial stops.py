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


#fråga chat om kombinerad statistik utifrån nuvarande kod

# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5
SLIPPAGE_POINTS = 0.5
# Spread och kommission uttryckt i samma enhet som priset i din CSV (points)
FIXED_SPREAD_POINTS = 0.8
COMM_POINTS_PER_SIDE = 0.05  # per side (entry eller exit)

ADX_THRESHOLD = 15

ATR_LEN = 14
STOP_ATR = 3.0        # initial stop = entry - STOP_ATR*ATR (testa 2, 3, 4)
TRAIL_ATR = 3.0       # chandelier trail = peak - TRAIL_ATR*ATR (testa 2, 3, 4)

PARTIAL_FRACTION = 0.0  # t.ex. 0.33 / 0.5
PARTIAL_R = 1.0         # partial target = entry + PARTIAL_R * initial_R (testa 1.0, 1.5, 2.0)

def commission_round_turn_points():
    """Kommission per round-turn (entry+exit) i points."""
    return 2.0 * COMM_POINTS_PER_SIDE

def commission_points(qty: float) -> float:
    """Kommission per sida, skalar med qty."""
    return COMM_POINTS_PER_SIDE * float(qty)


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

    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=ATR_LEN,
        fillna=False
    ).average_true_range()

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

    # --- NYTT för ATR trailing + partials ---
    qty = 0.0
    realized_pnl = 0.0

    initial_stop = None
    trail_stop = None
    peak_price = None

    partial_taken = False
    partial_target = None

    idx_list = df.index.to_list()

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
            exit_time = None

            # ATR måste finnas för trailing
            atr = row["atr"]
            if np.isnan(atr):
                continue

            # 1) Uppdatera peak och trailing stop (chandelier)
            peak_price = max(peak_price, row["high"])
            new_trail = peak_price - TRAIL_ATR * atr
            trail_stop = max(trail_stop, new_trail)

            # Effective stop får aldrig vara sämre än initial stop
            effective_stop = max(initial_stop, trail_stop)

            # 2) Partial take-profit (om ej redan taget)
            if (not partial_taken) and (row["high"] >= partial_target):
                part_qty = qty * PARTIAL_FRACTION
                if part_qty > 0:
                    spread = get_spread_points(next_row)

                    # target fill = target eller bättre om gap upp
                    fill_raw = max(partial_target, next_row["open"])
                    fill = fill_raw - HALF * spread - SLIPPAGE_POINTS  # sälj på bid

                    realized_pnl += (fill - entry_price) * part_qty - commission_points(part_qty)
                    qty -= part_qty
                    partial_taken = True

            # 3) Stop-out (intrabar)
            if row["low"] <= effective_stop:
                spread = get_spread_points(next_row)

                # stop fill = stop eller sämre om gap ner
                fill_raw = min(effective_stop, next_row["open"])
                fill = fill_raw - HALF * spread - SLIPPAGE_POINTS

                realized_pnl += (fill - entry_price) * qty - commission_points(qty)

                exit_price = fill
                exit_reason = "atr_trailing_stop"
                exit_time = idx_list[i + 1]

                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Direction': pos_direction,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Exit Reason': exit_reason,
                    'pnl': realized_pnl,
                })

                # Reset state
                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None

                qty = 0.0
                realized_pnl = 0.0
                initial_stop = None
                trail_stop = None
                peak_price = None
                partial_taken = False
                partial_target = None

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

        # EMA Crossover-logik

        adx_filter = adx > 15

        cross_ema = prev_ema_fast < prev_ema_medium and ema_fast > ema_medium

        # Slutlig entry-signal
        long_entry_signal = cross_ema and adx_filter
        # EN trade åt gången
        if long_entry_signal:
            pos_direction = 'LONG'
            entry_time = ts

            next_open = next_row['open']
            spread = get_spread_points(next_row)
            entry_price = next_open + HALF * spread + SLIPPAGE_POINTS  # LONG: köp på ask
            # Position sizing (enhet)
            qty = 1.0
            realized_pnl = -commission_points(qty)  # entry commission direkt

            # ATR måste finnas
            atr_entry = row["atr"]
            if np.isnan(atr_entry):
                continue

            # Initial stop och R-baserad partial
            initial_stop = entry_price - STOP_ATR * atr_entry
            R = entry_price - initial_stop
            partial_target = entry_price + PARTIAL_R * R
            partial_taken = False

            # Trailing init (peak startar från entry-baren)
            peak_price = row["high"]
            trail_stop = peak_price - TRAIL_ATR * atr_entry

            in_position = True

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