import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# =========================
# CONFIG
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "YOUR_TWELVEDATA_API_KEY_HERE")

UAE_TZ = ZoneInfo("Asia/Dubai")

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]

SUGGEST_CACHE_TTL_SEC = 10 * 60
SUGGEST_CACHE = {"ts": 0, "data": None}

USER_STATE = {}  # user_id -> {"pair": "EURUSD"}

# =========================
# UI
# =========================
def home_keyboard():
    rows = []
    row = []
    for i, p in enumerate(PAIRS, start=1):
        row.append(InlineKeyboardButton(p, callback_data=f"PAIR|{p}"))
        if i % 2 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("✨ Smart Suggestions", callback_data="ACT|SUGGEST")])
    return InlineKeyboardMarkup(rows)

def actions_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Get Signal", callback_data="ACT|GET"),
            InlineKeyboardButton("🔁 Refresh", callback_data="ACT|REFRESH"),
        ],
        [
            InlineKeyboardButton("📌 Change Pair", callback_data="ACT|CHOOSE_PAIR"),
            InlineKeyboardButton("✨ Suggestions", callback_data="ACT|SUGGEST"),
        ],
    ])

def suggestions_keyboard(sugg_pairs):
    rows = []
    for s in sugg_pairs[:4]:
        rows.append([InlineKeyboardButton(f"Select {s['pair']}", callback_data=f"PAIR|{s['pair']}")])
    rows.append([InlineKeyboardButton("⬅ Back to Pairs", callback_data="ACT|CHOOSE_PAIR")])
    return InlineKeyboardMarkup(rows)

# =========================
# DATA
# =========================
def fetch_candles(symbol: str, interval: str, outputsize: int = 260) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(data.get("message", "TwelveData error"))

    values = data.get("values", [])
    if not values:
        raise RuntimeError("No candle data returned")

    df = pd.DataFrame(values)
    df.rename(columns={"datetime": "time"}, inplace=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]]

# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.bfill().fillna(50)

def stochastic(df: pd.DataFrame, k_period=14, d_period=3, smooth=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k.bfill().fillna(50), d.bfill().fillna(50)

def atr(df: pd.DataFrame, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().bfill()

def candle_confirm(df: pd.DataFrame):
    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    if c > o:
        return "BULL"
    if c < o:
        return "BEAR"
    return "DOJI"

# =========================
# LOGIC
# =========================
def trend_bias_1h_30m(symbol: str):
    df_1h = fetch_candles(symbol, "1h")
    df_30m = fetch_candles(symbol, "30min")

    def bias_one(df):
        c = df["close"]
        e50 = ema(c, 50).iloc[-1]
        e200 = ema(c, 200).iloc[-1]
        price = c.iloc[-1]
        if price > e200 and e50 > e200:
            return "CALL"
        if price < e200 and e50 < e200:
            return "PUT"
        return "SKIP"

    b1 = bias_one(df_1h)
    b2 = bias_one(df_30m)
    if b1 == b2 and b1 in ("CALL", "PUT"):
        return b1
    return "SKIP"

def volatility_profile_from_1m(symbol: str):
    df_1m = fetch_candles(symbol, "1min", outputsize=220)
    a = atr(df_1m, 14).iloc[-1]
    price = df_1m["close"].iloc[-1]
    atr_pct = (a / price) * 100.0 if price else 0.0

    candle_size = abs(df_1m["close"].iloc[-1] - df_1m["open"].iloc[-1])
    spike = candle_size > (2.2 * a)

    if atr_pct < 0.03:
        return {"tf": "1min", "expiry": "3 Minutes", "label": "LOW", "atr_pct": atr_pct, "spike": spike}
    if atr_pct < 0.07:
        return {"tf": "2min", "expiry": "5 Minutes", "label": "NORMAL", "atr_pct": atr_pct, "spike": spike}
    if atr_pct < 0.12:
        return {"tf": "5min", "expiry": "10–15 Minutes", "label": "HIGH", "atr_pct": atr_pct, "spike": spike}
    return {"tf": None, "expiry": None, "label": "TOO_WILD", "atr_pct": atr_pct, "spike": True}

def compute_signal_auto(symbol: str):
    bias = trend_bias_1h_30m(symbol)
    if bias == "SKIP":
        return {
            "status": "SKIP",
            "signal": "—",
            "strength": 0,
            "auto_tf": "—",
            "expiry": "—",
            "vol_label": "—",
            "atr_pct": None,
            "reason": "Trend unclear on 1H/30M (EMA alignment not clean).",
        }

    vol = volatility_profile_from_1m(symbol)
    if vol["label"] == "TOO_WILD":
        return {
            "status": "SKIP",
            "signal": "—",
            "strength": 0,
            "auto_tf": "—",
            "expiry": "—",
            "vol_label": f"TOO WILD ({vol['atr_pct']:.3f}% ATR)",
            "atr_pct": vol["atr_pct"],
            "reason": "Volatility too high / spike. Best to skip for accuracy.",
        }

    entry_tf = vol["tf"]
    expiry = vol["expiry"]
    vol_label = f"{vol['label']} ({vol['atr_pct']:.3f}% ATR)"

    df_15m = fetch_candles(symbol, "15min")
    df_e = fetch_candles(symbol, entry_tf)

    recent_low = df_15m["low"].rolling(40).min().iloc[-1]
    recent_high = df_15m["high"].rolling(40).max().iloc[-1]
    price = df_e["close"].iloc[-1]
    rng = max(recent_high - recent_low, 1e-9)
    pos = (price - recent_low) / rng

    zone_ok = (pos <= 0.45) if bias == "CALL" else (pos >= 0.55)

    r = rsi(df_e["close"], 14)
    k, d = stochastic(df_e, 14, 3, 3)

    last_rsi, prev_rsi = r.iloc[-1], r.iloc[-2]
    last_k, prev_k = k.iloc[-1], k.iloc[-2]
    last_d, prev_d = d.iloc[-1], d.iloc[-2]
    cc = candle_confirm(df_e)

    score = 0
    reasons = []

    score += 20
    reasons.append("Trend aligned (1H/30M EMA).")

    if zone_ok:
        score += 20
        reasons.append("Pullback zone OK (15M range).")
    else:
        reasons.append("Zone weak (not in pullback area).")

    rsi_ok = False
    if bias == "CALL" and prev_rsi < 45 and last_rsi > prev_rsi:
        rsi_ok = True
    if bias == "PUT" and prev_rsi > 55 and last_rsi < prev_rsi:
        rsi_ok = True
    if rsi_ok:
        score += 20
        reasons.append("RSI confirmation.")
    else:
        reasons.append("RSI not ideal.")

    stoch_ok = False
    if bias == "CALL" and prev_k < prev_d and last_k > last_d and last_k < 40:
        stoch_ok = True
    if bias == "PUT" and prev_k > prev_d and last_k < last_d and last_k > 60:
        stoch_ok = True
    if stoch_ok:
        score += 20
        reasons.append("Stochastic confirmation.")
    else:
        reasons.append("Stochastic not ideal.")

    candle_ok = (bias == "CALL" and cc == "BULL") or (bias == "PUT" and cc == "BEAR")
    if candle_ok:
        score += 20
        reasons.append("Candle confirmation.")
    else:
        reasons.append("Candle not confirming.")

    if vol["spike"]:
        score = max(0, score - 40)
        reasons.append("Spike detected (skip recommended).")

    status = "TRADE" if score >= 80 else ("MEDIUM" if score >= 60 else "SKIP")
    signal = bias if status != "SKIP" else "—"

    return {
        "status": status,
        "signal": signal,
        "strength": score,
        "auto_tf": entry_tf,
        "expiry": expiry,
        "vol_label": vol_label,
        "atr_pct": vol["atr_pct"],
        "reason": "; ".join(reasons[:4]),
    }

def compute_smart_suggestions():
    now = time.time()
    if SUGGEST_CACHE["data"] and (now - SUGGEST_CACHE["ts"] < SUGGEST_CACHE_TTL_SEC):
        return SUGGEST_CACHE["data"]

    suggestions = []
    for pair in PAIRS:
        try:
            bias = trend_bias_1h_30m(pair)
            vol = volatility_profile_from_1m(pair)
            if bias == "SKIP":
                quality = 0
                note = "Trend unclear"
            elif vol["label"] == "TOO_WILD":
                quality = 10
                note = "Too volatile"
            else:
                vol_bonus = {"LOW": 20, "NORMAL": 30, "HIGH": 15}.get(vol["label"], 0)
                quality = 50 + vol_bonus
                note = f"{bias} bias, {vol['label']}"
            suggestions.append({
                "pair": pair,
                "bias": bias,
                "vol": vol["label"],
                "atr_pct": vol["atr_pct"],
                "quality": quality,
                "note": note,
            })
        except Exception:
            suggestions.append({
                "pair": pair,
                "bias": "—",
                "vol": "—",
                "atr_pct": None,
                "quality": 0,
                "note": "Data error",
            })

    suggestions = sorted(suggestions, key=lambda x: x["quality"], reverse=True)
    SUGGEST_CACHE["ts"] = now
    SUGGEST_CACHE["data"] = suggestions
    return suggestions

# =========================
# TELEGRAM
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    USER_STATE[user_id] = USER_STATE.get(user_id, {"pair": "EURUSD"})
    text = (
        "📊 *Forex Signal Bot (Professional)*\n\n"
        "1) Pehle *pair select* karein\n"
        "2) Phir *Get Signal* dabayen\n\n"
        "✨ Smart Suggestions bhi available hain."
    )
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=home_keyboard())

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    user_id = q.from_user.id
    USER_STATE[user_id] = USER_STATE.get(user_id, {"pair": "EURUSD"})
    data = q.data

    if data.startswith("PAIR|"):
        pair = data.split("|", 1)[1]
        USER_STATE[user_id]["pair"] = pair
        msg = (
            f"✅ *Selected Pair:* {pair}\n\n"
            "Ab aap *Get Signal* dabayen.\n"
            "_(Timeframe + Expiry auto volatility se suggest honge)_"
        )
        await q.edit_message_text(msg, parse_mode="Markdown", reply_markup=actions_keyboard())
        return

    if data.startswith("ACT|"):
        act = data.split("|", 1)[1]

        if act == "CHOOSE_PAIR":
            await q.edit_message_text("📌 *Select Pair:*", parse_mode="Markdown", reply_markup=home_keyboard())
            return

        if act == "SUGGEST":
            sugg = compute_smart_suggestions()
            lines = ["✨ *Smart Suggestions (Trend + Volatility)*\n"]
            for s in sugg[:4]:
                atr_txt = f"{s['atr_pct']:.3f}%" if s["atr_pct"] is not None else "—"
                lines.append(f"• *{s['pair']}* → {s['note']} | ATR: {atr_txt}")
            lines.append("\nSelect karne ke liye button dabayen:")
            await q.edit_message_text("\n".join(lines), parse_mode="Markdown", reply_markup=suggestions_keyboard(sugg))
            return

        if act in ("GET", "REFRESH"):
            pair = USER_STATE[user_id]["pair"]
            uae_now = datetime.now(UAE_TZ).strftime("%d %b %Y, %I:%M %p")

            try:
                result = compute_signal_auto(pair)
            except Exception as e:
                await q.edit_message_text(
                    f"❌ Data error: {str(e)}\n\nCheck your TWELVEDATA_KEY or try again.",
                    reply_markup=actions_keyboard(),
                )
                return

            badge = "✅ *TRADE*" if result["status"] == "TRADE" else ("⚠️ *MEDIUM*" if result["status"] == "MEDIUM" else "❌ *SKIP*")

            text = (
                f"📌 *PAIR:* {pair}\n"
                f"🎯 *SIGNAL:* {result['signal']}\n"
                f"⏱ *AUTO TF:* {result['auto_tf']}\n"
                f"⌛ *AUTO EXPIRY:* {result['expiry']}\n"
                f"📈 *VOLATILITY:* {result['vol_label']}\n"
                f"💪 *STRENGTH:* {result['strength']}%\n"
                f"🕒 *ENTRY TIME (UAE):* {uae_now}\n"
                f"{badge}\n\n"
                f"🧠 *Reason:* {result['reason']}"
            )
            await q.edit_message_text(text, parse_mode="Markdown", reply_markup=actions_keyboard())
            return

# =========================
# MAIN
# =========================
def main():
    if "YOUR_BOT_TOKEN_HERE" in BOT_TOKEN or "YOUR_TWELVEDATA_API_KEY_HERE" in TWELVEDATA_KEY:
        print("Set BOT_TOKEN and TWELVEDATA_KEY env vars (or edit in code).")
        return

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_callback))

    print("Bot running...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
