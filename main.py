import os
import math
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import requests
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# -----------------------------
# Config
# -----------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "").strip()
PORT = int(os.getenv("PORT", "10000"))  # Render port (for Web Service)
UAE_TZ = timezone(timedelta(hours=4))

# Pairs list (can add more)
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD", "USD/CHF", "NZD/USD"]

# User state memory (simple in-memory)
USER_STATE: Dict[int, Dict[str, str]] = {}  # chat_id -> {"pair": "EUR/USD", "tf": "5min"}

# TwelveData intervals we support
TF_OPTIONS = ["1min", "5min", "15min", "30min", "1h"]


# -----------------------------
# Utilities (Indicators)
# -----------------------------
def ema(values: List[float], period: int) -> float:
    if len(values) < period:
        return float("nan")
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return float("nan")
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    trs = []
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        trs.append(tr)
    # simple moving avg ATR
    if len(trs) < period:
        return float("nan")
    return sum(trs[-period:]) / period

def now_uae_str() -> str:
    return datetime.now(UAE_TZ).strftime("%Y-%m-%d %I:%M:%S %p (UAE)")

def normalize_symbol(pair: str) -> str:
    # TwelveData Forex usually accepts "EUR/USD"
    return pair.strip().upper()

def auto_timeframe_from_vol(atr_percent: float) -> str:
    # Simple heuristic (you can adjust)
    # Higher volatility => shorter TF
    if atr_percent >= 0.25:
        return "1min"
    if atr_percent >= 0.15:
        return "5min"
    if atr_percent >= 0.08:
        return "15min"
    if atr_percent >= 0.04:
        return "30min"
    return "1h"

def expiry_from_tf(tf: str) -> str:
    # Binary-style expiry suggestion (approx)
    return {
        "1min": "1–2 min",
        "5min": "5–10 min",
        "15min": "15–30 min",
        "30min": "30–45 min",
        "1h": "60–90 min",
    }.get(tf, "5–10 min")


# -----------------------------
# TwelveData Fetch
# -----------------------------
async def fetch_candles(symbol: str, interval: str, outputsize: int = 60) -> Tuple[List[float], List[float], List[float]]:
    """
    Returns (close, high, low) lists in chronological order.
    """
    if not TWELVEDATA_KEY:
        raise RuntimeError("TWELVEDATA_KEY missing in Render Environment Variables.")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
    }

    def _do():
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    data = await asyncio.to_thread(_do)

    if isinstance(data, dict) and data.get("status") == "error":
        # TwelveData error object
        msg = data.get("message", "Unknown TwelveData error")
        raise RuntimeError(msg)

    values = data.get("values")
    if not values:
        raise RuntimeError("No candle data returned (values empty). Check symbol/interval.")

    # TwelveData returns newest first -> reverse to chronological
    values = list(reversed(values))

    close = [float(v["close"]) for v in values]
    high = [float(v["high"]) for v in values]
    low  = [float(v["low"])  for v in values]
    return close, high, low


def build_signal_text(symbol: str, interval: str, close: List[float], high: List[float], low: List[float]) -> str:
    # Indicators
    c = close[-50:] if len(close) >= 50 else close
    e9 = ema(c, 9)
    e21 = ema(c, 21)
    r = rsi(c, 14)
    a = atr(high[-50:], low[-50:], close[-50:], 14)

    last = close[-1]
    atr_pct = (a / last) * 100 if (a and not math.isnan(a) and last) else float("nan")

    # Very simple direction logic (NOT a guarantee)
    direction = "WAIT"
    reason = []
    if not math.isnan(e9) and not math.isnan(e21):
        if e9 > e21:
            direction = "CALL (Buy)"
            reason.append("EMA9 > EMA21 (uptrend)")
        elif e9 < e21:
            direction = "PUT (Sell)"
            reason.append("EMA9 < EMA21 (downtrend)")
    if not math.isnan(r):
        if r >= 70:
            reason.append("RSI high (overbought) – be careful")
        elif r <= 30:
            reason.append("RSI low (oversold) – be careful")

    tf_auto = auto_timeframe_from_vol(atr_pct) if not math.isnan(atr_pct) else interval
    expiry = expiry_from_tf(tf_auto)

    lines = []
    lines.append("📌 *Signal*")
    lines.append(f"• Pair: *{symbol}*")
    lines.append(f"• Timeframe: *{interval}* (auto suggestion: *{tf_auto}*)")
    lines.append(f"• Entry time: *{now_uae_str()}*")
    lines.append(f"• Direction: *{direction}*")
    if not math.isnan(r):
        lines.append(f"• RSI(14): *{r:.1f}*")
    if not math.isnan(atr_pct):
        lines.append(f"• ATR%: *{atr_pct:.3f}%*")
    if reason:
        lines.append("• Notes: " + "; ".join(reason))
    lines.append(f"• Suggested expiry: *{expiry}*")
    lines.append("")
    lines.append("⚠️ *Note:* This is indicator-based analysis, not guaranteed profit. Risk manage always.")

    return "\n".join(lines)


# -----------------------------
# Telegram UI
# -----------------------------
def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Get Signal", callback_data="GET_SIGNAL"),
         InlineKeyboardButton("🔁 Refresh", callback_data="REFRESH")],
        [InlineKeyboardButton("📌 Change Pair", callback_data="CHANGE_PAIR"),
         InlineKeyboardButton("✨ Suggestions", callback_data="SUGGESTIONS")],
    ])

def pairs_kb() -> InlineKeyboardMarkup:
    rows = []
    for p in PAIRS:
        rows.append([InlineKeyboardButton(p.replace("/", ""), callback_data=f"PAIR|{p}")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="BACK_MENU")])
    return InlineKeyboardMarkup(rows)

def suggestions_kb() -> InlineKeyboardMarkup:
    # show top 4 common pairs
    sugg = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY"]
    rows = []
    for p in sugg:
        rows.append([InlineKeyboardButton(f"Select {p.replace('/', '')}", callback_data=f"PAIR|{p}")])
    rows.append([InlineKeyboardButton("⬅️ Back to Pairs", callback_data="CHANGE_PAIR")])
    return InlineKeyboardMarkup(rows)

def get_state(chat_id: int) -> Dict[str, str]:
    if chat_id not in USER_STATE:
        USER_STATE[chat_id] = {"pair": "EUR/USD", "tf": "5min"}
    return USER_STATE[chat_id]


# -----------------------------
# Handlers
# -----------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_state(chat_id)
    msg = (
        "✅ Bot is live!\n\n"
        f"Current Pair: *{st['pair']}*\n"
        f"Default TF: *{st['tf']}*\n\n"
        "Use buttons below:"
    )
    await update.message.reply_text(msg, reply_markup=main_menu_kb(), parse_mode="Markdown")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat_id
    st = get_state(chat_id)

    data = q.data or ""

    # Pair selection
    if data.startswith("PAIR|"):
        pair = data.split("|", 1)[1]
        st["pair"] = pair
        await q.edit_message_text(
            f"✅ Pair selected: *{pair}*\nNow press *Get Signal*.",
            reply_markup=main_menu_kb(),
            parse_mode="Markdown"
        )
        return

    if data == "CHANGE_PAIR":
        await q.edit_message_text(
            "📌 Select a pair:",
            reply_markup=pairs_kb()
        )
        return

    if data == "SUGGESTIONS":
        await q.edit_message_text(
            "✨ Smart Suggestions (Tap to select):",
            reply_markup=suggestions_kb()
        )
        return

    if data == "BACK_MENU":
        await q.edit_message_text(
            f"Menu\nCurrent Pair: *{st['pair']}*",
            reply_markup=main_menu_kb(),
            parse_mode="Markdown"
        )
        return

    # Get Signal / Refresh
    if data in ("GET_SIGNAL", "REFRESH"):
        symbol = normalize_symbol(st["pair"])
        interval = st.get("tf", "5min")

        try:
            close, high, low = await fetch_candles(symbol, interval, outputsize=80)
            text = build_signal_text(symbol, interval, close, high, low)
            await q.edit_message_text(text, reply_markup=main_menu_kb(), parse_mode="Markdown")
        except Exception as e:
            # Clear & helpful error
            err = str(e)
            await q.edit_message_text(
                f"❌ Data error: {err}\n\n"
                f"✅ Fix tips:\n"
                f"1) Press *Change Pair* and select a pair\n"
                f"2) Ensure *TWELVEDATA_KEY* is correct in Render\n\n"
                f"Current Pair: *{st['pair']}*",
                reply_markup=main_menu_kb(),
                parse_mode="Markdown"
            )
        return

    # Fallback
    await q.edit_message_text("Menu", reply_markup=main_menu_kb())


# -----------------------------
# Health server (fix Render port scan for Web Service)
# -----------------------------
async def run_health_server():
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            return  # silent

    def _serve():
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()

    await asyncio.to_thread(_serve)


async def post_init(app: Application):
    # start health server in background (only for Render Web Service)
    asyncio.create_task(run_health_server())


def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN missing. Add BOT_TOKEN in Render Environment Variables.")

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(on_callback))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
