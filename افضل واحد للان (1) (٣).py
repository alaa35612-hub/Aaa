#!/usr/bin/env python3
"""Smart Money Algo Pro E5 – ICT strategy scanner (rewritten).

This module rebuilds the multi-timeframe ICT strategy logic from scratch with a
focus on clarity, sequential rule evaluation, and console reporting.  The code
keeps the public entry point compatible with previous automation while the core
implementation now follows a three-step checklist:

1. Higher timeframe bias confirmation (BOS/CHOCH style break in structure).
2. Lower timeframe confirmation (liquidity sweep followed by displacement/MSS).
3. Entry zone validation (FVG, Order Block, or OTE retracement preference).

Scoring adheres to the specification supplied by the user where OTE is optional
and treated as a bonus.  Killzone and Silver Bullet sessions are optional and do
not block trade ideas when tzdata is unavailable.  Each symbol is analysed in
isolation so users can observe which conditions are met and which remain
pending.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ccxt = None  # type: ignore

try:  # Python >=3.9
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ZoneInfo = None  # type: ignore

# ---------------------------------------------------------------------------
# 1) SETTINGS (top-of-file as requested)
# ---------------------------------------------------------------------------


@dataclass
class ICTStrategySettings:
    enabled: bool = True
    higher_timeframe: str = "15m"
    lower_timeframe: str = "1m"
    higher_lookback: int = 300
    lower_lookback: int = 900
    ote_range: Tuple[float, float] = (0.62, 0.79)
    prefer_ote: bool = True
    require_ote: bool = False
    require_liquidity_sweep: bool = True
    allow_fvg: bool = True
    allow_order_block: bool = True
    risk_per_trade: float = 0.01
    use_silver_bullet: bool = True
    use_killzones: bool = True


@dataclass
class ICTScannerSettings:
    enabled: bool = True
    max_symbols: int = 60
    market_quote: str = "USDT"
    higher_limit: int = 500
    lower_limit: int = 1500
    verbose: bool = True


@dataclass
class ICTPerformanceSettings:
    enabled: bool = True
    metric_name: str = "percentage"
    minimum_value: float = 5.0
    top_n: int = 10
    prefer_positive: bool = True
    display_units: str = "%"


ICT_STRATEGY_SETTINGS = ICTStrategySettings()
ICT_SCANNER_SETTINGS = ICTScannerSettings()
ICT_PERFORMANCE_SETTINGS = ICTPerformanceSettings()

# Strategy scoring configuration from the user specification -----------------

ICT_SETTINGS: Dict[str, Any] = {
    "higher_timeframe": ICT_STRATEGY_SETTINGS.higher_timeframe,
    "lower_timeframe": ICT_STRATEGY_SETTINGS.lower_timeframe,
    "ote_range": ICT_STRATEGY_SETTINGS.ote_range,
    "prefer_ote": ICT_STRATEGY_SETTINGS.prefer_ote,
    "require_ote": ICT_STRATEGY_SETTINGS.require_ote,
    "require_liquidity_sweep": ICT_STRATEGY_SETTINGS.require_liquidity_sweep,
    "allow_fvg": ICT_STRATEGY_SETTINGS.allow_fvg,
    "allow_order_block": ICT_STRATEGY_SETTINGS.allow_order_block,
    "risk_per_trade": ICT_STRATEGY_SETTINGS.risk_per_trade,
    "use_silver_bullet": ICT_STRATEGY_SETTINGS.use_silver_bullet,
    "use_killzones": ICT_STRATEGY_SETTINGS.use_killzones,
}

KILLZONES_NY = {
    "london": (time(3, 0), time(4, 0)),
    "silver_am": (time(10, 0), time(11, 0)),
    "ny_pm": (time(14, 0), time(15, 0)),
}

ENTRY_WEIGHTS = {
    "bullish_fvg": 2.5,
    "bearish_fvg": 2.5,
    "demand_ob": 1.5,
    "supply_ob": 1.5,
    "in_ote": 1.0 if ICT_SETTINGS["prefer_ote"] else 0.0,
    "mss_tag": 0.5,
    "session_ok": 0.5,
}

MIN_SCORE_LONG = 2.5
MIN_SCORE_SHORT = 2.5

# ---------------------------------------------------------------------------
# 2) DATA MODELS
# ---------------------------------------------------------------------------


@dataclass
class ConditionStep:
    step: int
    name: str
    satisfied: bool
    detail: str
    pending: bool = False


@dataclass
class StrategyDecision:
    side: str
    decision: bool
    score: float
    reasons: List[Tuple[str, float]]
    risk: Dict[str, Any]


@dataclass
class StrategyEvaluation:
    symbol: str
    entry_price: float
    ote_zone: Optional[Tuple[float, float]]
    ote_zone_short: Optional[Tuple[float, float]]
    long_steps: List[ConditionStep]
    short_steps: List[ConditionStep]
    long_decision: StrategyDecision
    short_decision: StrategyDecision


@dataclass
class SymbolAnalysis:
    symbol: str
    ticker: Dict[str, Any]
    evaluation: StrategyEvaluation
    facts: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3) TIMEZONE HANDLING
# ---------------------------------------------------------------------------


_NY_TZ: Optional[ZoneInfo] = None
_NY_TZ_FAILED: bool = False


def _get_new_york_tz() -> Optional[ZoneInfo]:
    global _NY_TZ, _NY_TZ_FAILED
    if _NY_TZ_FAILED:
        return None
    if _NY_TZ is not None:
        return _NY_TZ
    if ZoneInfo is None:
        _NY_TZ_FAILED = True
        return None
    try:
        _NY_TZ = ZoneInfo("America/New_York")
    except Exception:
        _NY_TZ = None
        _NY_TZ_FAILED = True
    return _NY_TZ


def in_killzone_now(now_utc: Optional[datetime] = None) -> bool:
    if not ICT_SETTINGS["use_killzones"]:
        return True
    tz = _get_new_york_tz()
    if tz is None:
        return True
    now_utc = now_utc or datetime.utcnow()
    now_t = now_utc.astimezone(tz).time()
    for start, end in KILLZONES_NY.values():
        if start <= now_t <= end:
            return True
    return False

# ---------------------------------------------------------------------------
# 4) LOW-LEVEL HELPERS
# ---------------------------------------------------------------------------


def timeframe_to_minutes(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 60 * 24
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _ohlcv_column(ohlcv: Sequence[Sequence[float]], index: int) -> List[float]:
    return [float(row[index]) for row in ohlcv]


def compute_htf_bias(ohlcv: Sequence[Sequence[float]]) -> Tuple[bool, bool]:
    highs = _ohlcv_column(ohlcv, 2)
    lows = _ohlcv_column(ohlcv, 3)
    closes = _ohlcv_column(ohlcv, 4)
    if len(closes) < 20:
        return False, False
    lookback = min(50, len(closes) - 1)
    prev_high = max(highs[-lookback - 1 : -1])
    prev_low = min(lows[-lookback - 1 : -1])
    last_close = closes[-1]
    return last_close > prev_high, last_close < prev_low


def detect_liquidity_sweep(ohlcv: Sequence[Sequence[float]]) -> Tuple[bool, bool]:
    lows = _ohlcv_column(ohlcv, 3)
    highs = _ohlcv_column(ohlcv, 2)
    closes = _ohlcv_column(ohlcv, 4)
    if len(ohlcv) < 10:
        return False, False
    recent_low = lows[-2]
    recent_high = highs[-2]
    prior_low = min(lows[-8:-2])
    prior_high = max(highs[-8:-2])
    sell_side = recent_low < prior_low and closes[-2] > prior_low
    buy_side = recent_high > prior_high and closes[-2] < prior_high
    return sell_side, buy_side


def detect_displacement(ohlcv: Sequence[Sequence[float]]) -> Tuple[bool, bool]:
    opens = _ohlcv_column(ohlcv, 1)
    closes = _ohlcv_column(ohlcv, 4)
    highs = _ohlcv_column(ohlcv, 2)
    lows = _ohlcv_column(ohlcv, 3)
    if len(ohlcv) < 10:
        return False, False
    bodies = [abs(c - o) for o, c in zip(opens[-10:-1], closes[-10:-1])]
    avg_body = statistics.fmean(bodies) if bodies else 0.0
    last_body = abs(closes[-2] - opens[-2])
    last_range = highs[-2] - lows[-2]
    bullish = closes[-2] > opens[-2] and last_body > avg_body * 1.3
    bearish = closes[-2] < opens[-2] and last_body > avg_body * 1.3
    bullish = bullish or (closes[-1] > highs[-2] and last_range > avg_body * 1.1)
    bearish = bearish or (closes[-1] < lows[-2] and last_range > avg_body * 1.1)
    return bullish, bearish


def detect_fvg(ohlcv: Sequence[Sequence[float]]) -> Tuple[bool, bool]:
    if len(ohlcv) < 3:
        return False, False
    highs = _ohlcv_column(ohlcv, 2)
    lows = _ohlcv_column(ohlcv, 3)
    bullish = lows[-1] > highs[-3]
    bearish = highs[-1] < lows[-3]
    return bullish, bearish


def detect_order_block(ohlcv: Sequence[Sequence[float]]) -> Tuple[bool, bool]:
    if len(ohlcv) < 4:
        return False, False
    opens = _ohlcv_column(ohlcv, 1)
    closes = _ohlcv_column(ohlcv, 4)
    highs = _ohlcv_column(ohlcv, 2)
    lows = _ohlcv_column(ohlcv, 3)
    # Use the last three completed candles
    o_prev, c_prev = opens[-3], closes[-3]
    o_last, c_last = opens[-2], closes[-2]
    bullish = c_prev < o_prev and c_last > highs[-3]
    bearish = c_prev > o_prev and c_last < lows[-3]
    return bullish, bearish


def compute_ote_zones(ohlcv: Sequence[Sequence[float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if len(ohlcv) < 30:
        return None, None
    highs = _ohlcv_column(ohlcv, 2)
    lows = _ohlcv_column(ohlcv, 3)
    recent_high = max(highs[-60:])
    recent_low = min(lows[-60:])
    ote_low_ratio, ote_high_ratio = ICT_SETTINGS["ote_range"]
    # Long OTE (discount)
    range_size = recent_high - recent_low
    if range_size <= 0:
        return None, None
    ote_long_low = recent_high - range_size * ote_high_ratio
    ote_long_high = recent_high - range_size * ote_low_ratio
    ote_short_low = recent_low + range_size * ote_low_ratio
    ote_short_high = recent_low + range_size * ote_high_ratio
    return (min(ote_long_low, ote_long_high), max(ote_long_low, ote_long_high)), (
        min(ote_short_low, ote_short_high), max(ote_short_low, ote_short_high)
    )


def price_in_zone(price: float, zone: Optional[Tuple[float, float]]) -> bool:
    if zone is None:
        return False
    lower, upper = zone
    return lower <= price <= upper

# ---------------------------------------------------------------------------
# 5) SCORING HELPERS (provided by user spec)
# ---------------------------------------------------------------------------


def score_entry(side: str, facts: Dict[str, bool]) -> Tuple[float, Dict[str, float]]:
    if side == "long" and not facts.get("htf_bullish_bias", False):
        return 0.0, {}
    if side == "short" and not facts.get("htf_bearish_bias", False):
        return 0.0, {}
    if ICT_SETTINGS["require_liquidity_sweep"] and not facts.get("liquidity_sweep", False):
        return 0.0, {}
    if not facts.get("displacement_or_mss", False):
        return 0.0, {}
    entry_keys = [
        "bullish_fvg",
        "demand_ob",
        "in_ote",
    ] if side == "long" else ["bearish_fvg", "supply_ob", "in_ote"]
    if not any(facts.get(k, False) for k in entry_keys):
        return 0.0, {}
    total = 0.0
    detail: Dict[str, float] = {}
    for key in entry_keys:
        if facts.get(key, False):
            weight = ENTRY_WEIGHTS.get(key, 0.0)
            if weight:
                detail[key] = weight
                total += weight
    if facts.get("displacement_or_mss", False):
        bonus = ENTRY_WEIGHTS.get("mss_tag", 0.0)
        if bonus:
            detail["mss_tag"] = bonus
            total += bonus
    if in_killzone_now():
        bonus = ENTRY_WEIGHTS.get("session_ok", 0.0)
        if bonus:
            detail["session_ok"] = bonus
            total += bonus
    return total, detail


def pass_threshold(side: str, score: float) -> bool:
    threshold = MIN_SCORE_LONG if side == "long" else MIN_SCORE_SHORT
    return score >= threshold


def should_enter_long(facts: Dict[str, bool]) -> StrategyDecision:
    score, detail = score_entry("long", facts)
    return StrategyDecision(
        side="long",
        decision=pass_threshold("long", score),
        score=round(score, 2),
        reasons=sorted(detail.items(), key=lambda x: -x[1]),
        risk={
            "stop_basis": "behind_sweep_low",
            "tp_basis": "next_liquidity_pool",
            "risk_per_trade": ICT_SETTINGS["risk_per_trade"],
        },
    )


def should_enter_short(facts: Dict[str, bool]) -> StrategyDecision:
    score, detail = score_entry("short", facts)
    return StrategyDecision(
        side="short",
        decision=pass_threshold("short", score),
        score=round(score, 2),
        reasons=sorted(detail.items(), key=lambda x: -x[1]),
        risk={
            "stop_basis": "behind_sweep_high",
            "tp_basis": "next_liquidity_pool",
            "risk_per_trade": ICT_SETTINGS["risk_per_trade"],
        },
    )

# ---------------------------------------------------------------------------
# 6) STRATEGY DETECTION
# ---------------------------------------------------------------------------


def _build_condition_sequence(
    side: str,
    facts: Dict[str, bool],
    ote_zone: Optional[Tuple[float, float]],
    entry_price: float,
) -> Tuple[List[ConditionStep], StrategyDecision]:
    steps: List[ConditionStep] = []
    pending = False

    if side == "long":
        bias_met = facts.get("htf_bullish_bias", False)
        steps.append(
            ConditionStep(
                step=1,
                name="حدث هيكلي صاعد على HTF",
                satisfied=bias_met,
                detail="تم رصد BOS/CHOCH صاعد" if bias_met else "لم يظهر كسر صاعد واضح",
                pending=not bias_met,
            )
        )
        if not bias_met:
            pending = True
        sweep_met = facts.get("liquidity_sweep", False) and facts.get("displacement_or_mss", False)
        steps.append(
            ConditionStep(
                step=2,
                name="سحب سيولة (sell-side) + اندفاع MSS على LTF",
                satisfied=sweep_met and not pending,
                detail="تم سحب السيولة مع اندفاع" if sweep_met else "السويب أو الاندفاع غير مكتمل",
                pending=pending or not sweep_met,
            )
        )
        if pending or not sweep_met:
            pending = True
        entry_met = any(
            facts.get(k, False)
            for k in ("bullish_fvg", "demand_ob", "in_ote")
        )
        entry_detail = []
        if facts.get("bullish_fvg"):
            entry_detail.append("FVG صاعد")
        if facts.get("demand_ob"):
            entry_detail.append("بلوك طلب")
        if facts.get("in_ote"):
            entry_detail.append("داخل OTE")
        if not entry_detail:
            entry_detail.append("لم يتم العثور على مناطق مطابقة")
        steps.append(
            ConditionStep(
                step=3,
                name="منطقة دخول صاعدة (FVG / OB / OTE)",
                satisfied=entry_met and not pending,
                detail="، ".join(entry_detail),
                pending=pending or not entry_met,
            )
        )
    else:
        bias_met = facts.get("htf_bearish_bias", False)
        steps.append(
            ConditionStep(
                step=1,
                name="حدث هيكلي هابط على HTF",
                satisfied=bias_met,
                detail="تم رصد BOS/CHOCH هابط" if bias_met else "لم يظهر كسر هابط واضح",
                pending=not bias_met,
            )
        )
        if not bias_met:
            pending = True
        sweep_met = facts.get("liquidity_sweep", False) and facts.get("displacement_or_mss", False)
        steps.append(
            ConditionStep(
                step=2,
                name="سحب سيولة (buy-side) + اندفاع MSS على LTF",
                satisfied=sweep_met and not pending,
                detail="تم سحب السيولة مع اندفاع" if sweep_met else "السويب أو الاندفاع غير مكتمل",
                pending=pending or not sweep_met,
            )
        )
        if pending or not sweep_met:
            pending = True
        entry_met = any(
            facts.get(k, False)
            for k in ("bearish_fvg", "supply_ob", "in_ote")
        )
        entry_detail = []
        if facts.get("bearish_fvg"):
            entry_detail.append("FVG هابط")
        if facts.get("supply_ob"):
            entry_detail.append("بلوك عرض")
        if facts.get("in_ote"):
            entry_detail.append("داخل OTE")
        if not entry_detail:
            entry_detail.append("لم يتم العثور على مناطق مطابقة")
        steps.append(
            ConditionStep(
                step=3,
                name="منطقة دخول هابطة (FVG / OB / OTE)",
                satisfied=entry_met and not pending,
                detail="، ".join(entry_detail),
                pending=pending or not entry_met,
            )
        )

    decision = should_enter_long(facts) if side == "long" else should_enter_short(facts)
    return steps, decision


def detect_ict_strategy(
    symbol: str,
    htf_ohlcv: Sequence[Sequence[float]],
    ltf_ohlcv: Sequence[Sequence[float]],
    entry_price: float,
) -> StrategyEvaluation:
    htf_bullish, htf_bearish = compute_htf_bias(htf_ohlcv)
    sweep_sell, sweep_buy = detect_liquidity_sweep(ltf_ohlcv)
    disp_bull, disp_bear = detect_displacement(ltf_ohlcv)
    fvg_bull, fvg_bear = detect_fvg(ltf_ohlcv)
    ob_bull, ob_bear = detect_order_block(ltf_ohlcv)
    ote_long, ote_short = compute_ote_zones(ltf_ohlcv)

    long_facts = {
        "htf_bullish_bias": htf_bullish,
        "liquidity_sweep": sweep_sell,
        "displacement_or_mss": disp_bull,
        "bullish_fvg": fvg_bull if ICT_SETTINGS["allow_fvg"] else False,
        "demand_ob": ob_bull if ICT_SETTINGS["allow_order_block"] else False,
        "in_ote": price_in_zone(entry_price, ote_long),
    }
    short_facts = {
        "htf_bearish_bias": htf_bearish,
        "liquidity_sweep": sweep_buy,
        "displacement_or_mss": disp_bear,
        "bearish_fvg": fvg_bear if ICT_SETTINGS["allow_fvg"] else False,
        "supply_ob": ob_bear if ICT_SETTINGS["allow_order_block"] else False,
        "in_ote": price_in_zone(entry_price, ote_short),
    }

    long_steps, long_decision = _build_condition_sequence("long", long_facts, ote_long, entry_price)
    short_steps, short_decision = _build_condition_sequence("short", short_facts, ote_short, entry_price)

    return StrategyEvaluation(
        symbol=symbol,
        entry_price=entry_price,
        ote_zone=ote_long,
        ote_zone_short=ote_short,
        long_steps=long_steps,
        short_steps=short_steps,
        long_decision=long_decision,
        short_decision=short_decision,
    )

# ---------------------------------------------------------------------------
# 7) DATA ACCESS
# ---------------------------------------------------------------------------


def build_exchange() -> Any:
    if ccxt is None:
        raise RuntimeError("ccxt غير متوفر، الرجاء تثبيته لتشغيل الماسح")
    exchange = ccxt.binance({"enableRateLimit": True})
    return exchange


def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return data


def fetch_symbol_list(exchange: Any, quote: str, max_symbols: int) -> List[str]:
    markets = exchange.load_markets()
    symbols = [m for m in markets if m.endswith(f"/{quote}") and markets[m]["active"]]
    symbols.sort()
    return symbols[:max_symbols]


def fetch_tickers(exchange: Any, symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    try:
        return exchange.fetch_tickers(symbols)
    except Exception:
        tickers: Dict[str, Dict[str, Any]] = {}
        for sym in symbols:
            tickers[sym] = exchange.fetch_ticker(sym)
        return tickers

# ---------------------------------------------------------------------------
# 8) REPORTING UTILITIES
# ---------------------------------------------------------------------------


def format_currency(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def format_zone(zone: Optional[Tuple[float, float]]) -> str:
    if zone is None:
        return "غير متاح"
    lower, upper = zone
    return f"{format_currency(lower)} → {format_currency(upper)}"


def print_condition_steps(steps: Sequence[ConditionStep]) -> None:
    for step in steps:
        prefix = "✅" if step.satisfied else ("⏳" if step.pending else "❌")
        print(f"    {prefix} [{step.step}] {step.name}: {step.detail}")


def print_strategy_decision(decision: StrategyDecision) -> None:
    verdict = "مكتملة" if decision.decision else "غير مكتملة"
    print(f"    النتيجة: {decision.side.upper()} {verdict} | الدرجة: {decision.score:.2f}")
    if decision.reasons:
        reasons = ", ".join(f"{name} (+{weight})" for name, weight in decision.reasons)
        print(f"      الأسباب: {reasons}")
    else:
        print("      الأسباب: لا توجد مكافآت حالية")
    print(
        "      إدارة المخاطر: SL -> {stop}, TP -> {tp}, مخاطرة: {risk:.2%}".format(
            stop=decision.risk["stop_basis"],
            tp=decision.risk["tp_basis"],
            risk=decision.risk["risk_per_trade"],
        )
    )


def print_symbol_analysis(analysis: SymbolAnalysis) -> None:
    ticker = analysis.ticker
    change = ticker.get("percentage")
    change_text = f"{change:.2f}{ICT_PERFORMANCE_SETTINGS.display_units}" if change is not None else "N/A"
    print("\n" + "=" * 78)
    print(
        f"رمز: {analysis.symbol} | السعر الحالي: {format_currency(analysis.evaluation.entry_price)} | تغير 24h: {change_text}"
    )
    print("-" * 78)
    eval_obj = analysis.evaluation
    print("  نطاق OTE للشراء :", format_zone(eval_obj.ote_zone))
    print("  نطاق OTE للبيع  :", format_zone(eval_obj.ote_zone_short))
    print("  --- خطوات سيناريو الشراء ---")
    print_condition_steps(eval_obj.long_steps)
    print_strategy_decision(eval_obj.long_decision)
    print("  --- خطوات سيناريو البيع ---")
    print_condition_steps(eval_obj.short_steps)
    print_strategy_decision(eval_obj.short_decision)


def print_performance_rankings(tickers: Dict[str, Dict[str, Any]]) -> None:
    if not ICT_PERFORMANCE_SETTINGS.enabled:
        return
    metric = ICT_PERFORMANCE_SETTINGS.metric_name
    min_value = ICT_PERFORMANCE_SETTINGS.minimum_value
    prefer_positive = ICT_PERFORMANCE_SETTINGS.prefer_positive
    candidates: List[Tuple[str, float]] = []
    for symbol, data in tickers.items():
        value = data.get(metric)
        if value is None:
            continue
        if prefer_positive and value < min_value:
            continue
        if not prefer_positive and value > -min_value:
            continue
        candidates.append((symbol, float(value)))
    reverse = prefer_positive
    candidates.sort(key=lambda item: item[1], reverse=reverse)
    top = candidates[: ICT_PERFORMANCE_SETTINGS.top_n]
    print("\n" + "#" * 78)
    direction = "الأعلى" if prefer_positive else "الأدنى"
    print(
        f"تحديد أولوية الرابحين {direction} باستخدام المقياس '{metric}' وحد أدنى {min_value:.2f}."
    )
    if not top:
        print("لا توجد رموز تطابق معيار الأداء الحالي.")
        return
    for rank, (symbol, value) in enumerate(top, 1):
        print(f"  {rank:>2}. {symbol:<15} -> {value:.2f}{ICT_PERFORMANCE_SETTINGS.display_units}")

# ---------------------------------------------------------------------------
# 9) SCANNING PIPELINE
# ---------------------------------------------------------------------------


def analyse_symbol(
    exchange: Any,
    symbol: str,
    htf: str,
    ltf: str,
    ticker: Dict[str, Any],
) -> SymbolAnalysis:
    htf_ohlcv = fetch_ohlcv(exchange, symbol, htf, ICT_SCANNER_SETTINGS.higher_limit)
    ltf_ohlcv = fetch_ohlcv(exchange, symbol, ltf, ICT_SCANNER_SETTINGS.lower_limit)
    price = float(ticker.get("last") or ticker.get("close") or ltf_ohlcv[-1][4])
    evaluation = detect_ict_strategy(symbol, htf_ohlcv, ltf_ohlcv, price)
    facts = {
        "htf_bullish_bias": evaluation.long_decision.score > 0,
        "htf_bearish_bias": evaluation.short_decision.score > 0,
    }
    return SymbolAnalysis(symbol=symbol, ticker=ticker, evaluation=evaluation, facts=facts)


def scan_exchange(symbols: Optional[Iterable[str]] = None) -> List[SymbolAnalysis]:
    exchange = build_exchange()
    if symbols is None:
        symbols = fetch_symbol_list(
            exchange,
            ICT_SCANNER_SETTINGS.market_quote,
            ICT_SCANNER_SETTINGS.max_symbols,
        )
    symbols_list = list(symbols)
    tickers = fetch_tickers(exchange, symbols_list)
    analyses: List[SymbolAnalysis] = []
    for idx, symbol in enumerate(symbols_list, 1):
        if ICT_SCANNER_SETTINGS.verbose:
            print(f"[{idx}/{len(symbols_list)}] تحليل {symbol} ...")
        try:
            ticker = tickers.get(symbol, {})
            analyses.append(
                analyse_symbol(
                    exchange,
                    symbol,
                    ICT_STRATEGY_SETTINGS.higher_timeframe,
                    ICT_STRATEGY_SETTINGS.lower_timeframe,
                    ticker,
                )
            )
        except Exception as exc:
            print(f"    ⚠️ تعذر تحليل {symbol}: {exc}")
    print_performance_rankings(tickers)
    return analyses

# ---------------------------------------------------------------------------
# 10) CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICT strategy scanner")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="قائمة رموز مخصصة (مثال: BTC/USDT ETH/USDT)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if not ICT_STRATEGY_SETTINGS.enabled:
        print("⚠️ تم تعطيل استراتيجية ICT في الإعدادات.")
        return 0
    args = parse_args(argv)
    analyses = scan_exchange(args.symbols)
    for analysis in analyses:
        print_symbol_analysis(analysis)
    if not analyses:
        print("لم يتم تحليل أية رموز.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
