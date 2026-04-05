import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq
from datetime import datetime, timedelta, timezone
import streamlit.components.v1 as components  # 👈 ⭐ 이 줄을 추가해 주세요!

# 한국 시간(KST) 설정
KST = timezone(timedelta(hours=9))

st.set_page_config(page_title="차트 분석 어드바이저", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp { background-color: #0d0f14; }
[data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #1e2130; }

/* 스크롤바 커스텀 */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a3040; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4a5060; }

/* 버튼 스타일 */
div.stButton > button { background: #4ade80; color: #0d1a0f; border: none; border-radius: 8px; font-weight: 700; font-size: 14px; padding: 10px 24px; width: 100%; margin-bottom: 8px; transition: all 0.2s ease; }
div.stButton > button:hover { background: #22c55e; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(74, 222, 128, 0.2); }

/* AI 분석 박스 글로우 효과 */
.analysis-box { 
    background: linear-gradient(145deg, #111620 0%, #0d0f14 100%); 
    border: 1px solid #1e3040; 
    border-left: 3px solid #4ade80; 
    border-radius: 10px; 
    padding: 20px 24px; 
    font-size: 14px; 
    line-height: 1.9; 
    color: #c8d0e0; 
    white-space: pre-wrap; 
    box-shadow: 0 0 15px rgba(74, 222, 128, 0.05);
}

/* 공통 카드 클래스 애니메이션 */
.hover-card { transition: all 0.3s ease; }
.hover-card:hover { transform: translateY(-3px); box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4); border-color: #2a3040 !important; }

.signal-card { background: #151820; border: 1px solid #1e2130; border-radius: 10px; padding: 14px 18px; margin-bottom: 8px; }
.confidence-bar { height: 8px; border-radius: 4px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

KR_UNIVERSE = {
    "삼성전자":"005930.KS","SK하이닉스":"000660.KS","LG에너지솔루션":"373220.KS",
    "삼성바이오로직스":"207940.KS","현대차":"005380.KS","기아":"000270.KS",
    "셀트리온":"068270.KS","카카오":"035720.KS","네이버":"035420.KS",
    "LG화학":"051910.KS","삼성SDI":"006400.KS","POSCO홀딩스":"005490.KS",
    "KB금융":"105560.KS","신한지주":"055550.KS","하나금융":"086790.KS",
    "현대모비스":"012330.KS","SK텔레콤":"017670.KS","LG전자":"066570.KS",
    "크래프톤":"259960.KS","엔씨소프트":"036570.KS",
    "한화에어로스페이스":"012450.KS","HD현대중공업":"329180.KS",
    "HPSP":"403870.KQ","에코프로":"086520.KQ","에코프로비엠":"247540.KQ",
    "리노공업":"058470.KQ","알테오젠":"196170.KQ","HLB":"028300.KQ",
    "클래시스":"214150.KQ","레인보우로보틱스":"277810.KQ",
    "이오테크닉스":"039030.KQ","파크시스템스":"140860.KQ",
    "원익IPS":"240810.KQ","피에스케이":"319660.KQ",
    "덕산네오룩스":"213580.KQ","루닛":"328130.KQ",
}

def search_ticker(query):
    query = query.strip()
    if query.endswith(".KS") or query.endswith(".KQ") or (query.isupper() and len(query) <= 5):
        return query.upper()
    q = query.lower()
    for name, ticker in KR_UNIVERSE.items():
        if q in name.lower():
            return ticker
    try:
        results = yf.Search(query, max_results=3)
        if results.quotes:
            return results.quotes[0]["symbol"]
    except:
        pass
    return query.upper()

# ══════════════════════════════════════
# 데이터 수집
# ══════════════════════════════════════
def get_stock_data(ticker, period, interval='1d'):
    try:
        raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if raw is None or raw.empty: return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        return raw.dropna(how="all")
    except:
        return None

def get_weekly_data(ticker):
    """주봉 데이터 (멀티타임프레임용)"""
    try:
        raw = yf.download(ticker, period="2y", interval="1wk", auto_adjust=True, progress=False)
        if raw is None or raw.empty: return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        return raw.dropna(how="all")
    except:
        return None

# ══════════════════════════════════════
# 지표 계산 함수들
# ══════════════════════════════════════
def calc_indicators(df):
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    df["MA5"]   = close.rolling(5).mean()
    df["MA20"]  = close.rolling(20).mean()
    df["MA60"]  = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2*bb_std
    df["BB_lower"] = bb_mid - 2*bb_std
    df["BB_mid"]   = bb_mid
    df["BB_width"] = (df["BB_upper"]-df["BB_lower"])/bb_mid

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100/(1+rs)

    rsi     = df["RSI"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    df["StochRSI_K"] = ((rsi-rsi_min)/(rsi_max-rsi_min+1e-10)).rolling(3).mean()*100
    df["StochRSI_D"] = df["StochRSI_K"].rolling(3).mean()

    ema12 = close.ewm(span=12,adjust=False).mean()
    ema26 = close.ewm(span=26,adjust=False).mean()
    df["MACD"]        = ema12-ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_hist"]   = df["MACD"]-df["MACD_signal"]

    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["Vol_MA20"] = vol.rolling(20).mean()

    high9  = high.rolling(9).max();  low9  = low.rolling(9).min()
    high26 = high.rolling(26).max(); low26 = low.rolling(26).min()
    high52 = high.rolling(52).max(); low52 = low.rolling(52).min()
    df["Tenkan"]   = (high9  + low9)  / 2          
    df["Kijun"]    = (high26 + low26) / 2          
    df["SenkouA"]  = ((df["Tenkan"]+df["Kijun"])/2).shift(26)   
    df["SenkouB"]  = ((high52+low52)/2).shift(26)               
    df["Chikou"]   = close.shift(-26)                           

    return df

def calc_fibonacci(df):
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    recent = min(60, len(df)-1)
    h_idx = high.iloc[-recent:].idxmax()
    l_idx = low.iloc[-recent:].idxmin()
    swing_high = float(high[h_idx])
    swing_low  = float(low[l_idx])
    diff = swing_high - swing_low
    retracement = {
        "0%":    swing_high,
        "23.6%": swing_high - diff*0.236,
        "38.2%": swing_high - diff*0.382,
        "50%":   swing_high - diff*0.500,
        "61.8%": swing_high - diff*0.618,
        "78.6%": swing_high - diff*0.786,
        "100%":  swing_low,
    }
    extension = {
        "127.2%": swing_low + diff*1.272,
        "161.8%": swing_low + diff*1.618,
        "200%":   swing_low + diff*2.000,
        "261.8%": swing_low + diff*2.618,
    }
    return retracement, extension, swing_high, swing_low

def detect_divergence(df):
    close = df["Close"].squeeze().values[-30:]
    rsi   = df["RSI"].values[-30:]
    results = []
    price_highs = [i for i in range(1,len(close)-1) if close[i]>close[i-1] and close[i]>close[i+1]]
    if len(price_highs) >= 2:
        p1,p2 = price_highs[-2], price_highs[-1]
        if close[p2]>close[p1] and rsi[p2]<rsi[p1]:
            results.append(("약세 다이버전스","⚠️ 가격 신고점+RSI 하락 → 5파 완성/반전 가능"))
    price_lows = [i for i in range(1,len(close)-1) if close[i]<close[i-1] and close[i]<close[i+1]]
    if len(price_lows) >= 2:
        p1,p2 = price_lows[-2], price_lows[-1]
        if close[p2]<close[p1] and rsi[p2]>rsi[p1]:
            results.append(("강세 다이버전스","✅ 가격 신저점+RSI 상승 → 2파/4파 완료/반등 가능"))
    return results

# ══════════════════════════════════════
# 핵심: 교차 검증 엔진
# ══════════════════════════════════════
def cross_validate(df_daily, df_weekly):
    signals = []
    score   = 0
    max_score = 0
    close_d = df_daily["Close"].squeeze()
    cur = float(close_d.iloc[-1])

    # 1. 엘리어트
    max_score += 20
    rsi_cur  = float(df_daily["RSI"].iloc[-1])
    rsi_prev = float(df_daily["RSI"].iloc[-2])
    macd_h   = float(df_daily["MACD_hist"].iloc[-1])
    macd_hp  = float(df_daily["MACD_hist"].iloc[-2])
    ma5  = float(df_daily["MA5"].iloc[-1])
    ma20 = float(df_daily["MA20"].iloc[-1])
    ma60 = float(df_daily["MA60"].iloc[-1])
    vol_cur  = float(df_daily["Volume"].squeeze().iloc[-1])
    vol_ma20 = float(df_daily["Vol_MA20"].iloc[-1])
    vol_ratio = vol_cur/vol_ma20 if vol_ma20 else 1

    elliott_label = "파악 어려움"
    if rsi_cur < 40 and rsi_cur > rsi_prev and macd_h > macd_hp:
        elliott_label = "2파/4파 완료 추정 (RSI 반등+MACD 상승 전환)"
        score += 20
        signals.append(("🌊 엘리어트", "매수", elliott_label, 20))
    elif rsi_cur < 35:
        score += 10
        elliott_label = "조정 구간 (과매도)"
        signals.append(("🌊 엘리어트", "관망→매수준비", elliott_label, 10))
    elif ma5 > ma20 > ma60 and vol_ratio > 1.5:
        score += 15
        elliott_label = "3파 진행 중 추정 (정배열+거래량)"
        signals.append(("🌊 엘리어트", "추세추종매수", elliott_label, 15))
    else:
        signals.append(("🌊 엘리어트", "관망", elliott_label, 0))

    # 2. 일목
    max_score += 25
    ichi_score = 0
    ichi_signals = []
    try:
        tenkan  = float(df_daily["Tenkan"].iloc[-1])
        kijun   = float(df_daily["Kijun"].iloc[-1])
        senkou_a = float(df_daily["SenkouA"].iloc[-1])
        senkou_b = float(df_daily["SenkouB"].iloc[-1])
        cloud_top = max(senkou_a, senkou_b)
        cloud_bot = min(senkou_a, senkou_b)
        if cur > cloud_top: ichi_score += 5; ichi_signals.append("가격>구름(상승장)")
        elif cur < cloud_bot: ichi_score -= 5; ichi_signals.append("가격<구름(하락장)")
        if tenkan > kijun: ichi_score += 5; ichi_signals.append("전환선>기준선(단기상승)")
        tenkan_prev = float(df_daily["Tenkan"].iloc[-2]); kijun_prev  = float(df_daily["Kijun"].iloc[-2])
        if tenkan > kijun and tenkan_prev <= kijun_prev: ichi_score += 10; ichi_signals.append("전환/기준 골든크로스!")
        if senkou_a > senkou_b: ichi_score += 5; ichi_signals.append("상승 구름(미래 지지)")
        ichi_score = max(0, min(25, ichi_score))
        score += ichi_score
        ichi_label = " / ".join(ichi_signals) if ichi_signals else "신호 없음"
        direction = "매수" if ichi_score >= 15 else "관망→매수준비" if ichi_score >= 8 else "관망"
        signals.append(("☁️ 일목균형표", direction, ichi_label, ichi_score))
    except:
        signals.append(("☁️ 일목균형표", "계산오류", "데이터 부족", 0))

    # 3. 주봉
    max_score += 20
    mtf_score = 0
    mtf_label = "주봉 데이터 없음"
    if df_weekly is not None and len(df_weekly) >= 26:
        try:
            df_weekly = calc_indicators(df_weekly)
            w_close = df_weekly["Close"].squeeze()
            w_ma20  = float(df_weekly["MA20"].iloc[-1])
            w_ma60  = float(df_weekly["MA60"].iloc[-1]) if len(df_weekly)>=60 else w_ma20
            w_rsi   = float(df_weekly["RSI"].iloc[-1])
            w_macd_h = float(df_weekly["MACD_hist"].iloc[-1])
            w_macd_hp = float(df_weekly["MACD_hist"].iloc[-2])
            w_cur   = float(w_close.iloc[-1])
            w_tenkan = float(df_weekly["Tenkan"].iloc[-1])
            w_kijun  = float(df_weekly["Kijun"].iloc[-1])

            mtf_sigs = []
            if w_cur > w_ma20 > w_ma60: mtf_score += 8; mtf_sigs.append("주봉 정배열(대세 상승)")
            elif w_cur > w_ma20: mtf_score += 4; mtf_sigs.append("주봉 단기 상승")
            if 40 <= w_rsi <= 60: mtf_score += 4; mtf_sigs.append(f"주봉RSI중립({w_rsi:.0f})-진입유리")
            elif w_rsi < 40: mtf_score += 6; mtf_sigs.append(f"주봉RSI과매도({w_rsi:.0f})-반등구간")
            if w_macd_h > 0 and w_macd_hp <= 0: mtf_score += 6; mtf_sigs.append("주봉 MACD 골든크로스!")
            elif w_macd_h > w_macd_hp: mtf_score += 2; mtf_sigs.append("주봉 MACD 상승 중")
            if w_tenkan > w_kijun: mtf_score += 4; mtf_sigs.append("주봉 전환>기준")
            mtf_score = min(20, mtf_score)
            mtf_label = " / ".join(mtf_sigs) if mtf_sigs else "신호 없음"
            direction = "매수" if mtf_score >= 14 else "관망→매수준비" if mtf_score >= 8 else "관망"
            signals.append(("📅 멀티타임프레임(주봉)", direction, mtf_label, mtf_score))
            score += mtf_score
        except:
            signals.append(("📅 멀티타임프레임(주봉)", "계산오류", "계산 실패", 0))
    else:
        signals.append(("📅 멀티타임프레임(주봉)", "데이터부족", mtf_label, 0))

    # 4. 피보나치
    max_score += 15
    fib_score = 0
    fib_label = "지지구간 미확인"
    try:
        ret, ext, sh, sl = calc_fibonacci(df_daily)
        fib_sigs = []
        for label, val in ret.items():
            if abs(cur-val)/val < 0.02:
                if label in ["38.2%","50%","61.8%"]: fib_score += 10; fib_sigs.append(f"피보 {label} 지지({val:,.0f})")
                elif label in ["23.6%","78.6%"]: fib_score += 5; fib_sigs.append(f"피보 {label} 지지({val:,.0f})")
        for label, val in ext.items():
            if abs(cur-val)/val < 0.02: fib_sigs.append(f"피보 확장 {label} 도달 — 익절 고려")
        fib_score = min(15, fib_score)
        fib_label = " / ".join(fib_sigs) if fib_sigs else "주요 레벨 미근접"
        direction = "매수" if fib_score >= 10 else "참고" if fib_score >= 5 else "대기"
        signals.append(("📐 피보나치", direction, fib_label, fib_score))
        score += fib_score
    except:
        signals.append(("📐 피보나치", "계산오류", "계산 실패", 0))

    # 5. 다이버전스
    max_score += 20
    mom_score = 0
    mom_sigs = []
    divs = detect_divergence(df_daily)
    for d_type, d_msg in divs:
        if "강세" in d_type: mom_score += 15; mom_sigs.append(d_msg)
        elif "약세" in d_type: mom_score -= 5;  mom_sigs.append(d_msg)

    if macd_h > 0 and macd_hp <= 0: mom_score += 5; mom_sigs.append("MACD 골든크로스")
    stoch_k = float(df_daily["StochRSI_K"].iloc[-1]); stoch_kp = float(df_daily["StochRSI_K"].iloc[-2])
    if stoch_k < 20 and stoch_k > stoch_kp: mom_score += 5; mom_sigs.append(f"StochRSI 과매도 반등({stoch_k:.0f})")

    mom_score = max(0, min(20, mom_score))
    mom_label = " / ".join(mom_sigs) if mom_sigs else "다이버전스 없음"
    direction = "매수" if mom_score >= 15 else "관망→매수준비" if mom_score >= 8 else "관망"
    signals.append(("⚡ 다이버전스+모멘텀", direction, mom_label, mom_score))
    score += mom_score

    confidence = int(score / max_score * 100) if max_score else 0
    buy_count = sum(1 for s in signals if s[1] in ["매수","추세추종매수"])

    if confidence >= 75 and buy_count >= 3: verdict, verdict_color = "강한 매수", "#4ade80"
    elif confidence >= 60 and buy_count >= 2: verdict, verdict_color = "매수", "#86efac"
    elif confidence >= 45: verdict, verdict_color = "관망→매수준비", "#fbbf24"
    elif confidence < 30: verdict, verdict_color = "관망/매도검토", "#f87171"
    else: verdict, verdict_color = "관망", "#94a3b8"

    return {
        "signals": signals, "score": score, "max_score": max_score,
        "confidence": confidence, "buy_count": buy_count,
        "verdict": verdict, "verdict_color": verdict_color, "divs": divs,
    }

# ══════════════════════════════════════
# 차트 빌더
# ══════════════════════════════════════
def build_chart(df, ticker, show_fib=True, show_ichi=True):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.18, 0.14],
        vertical_spacing=0.02,
        subplot_titles=["", "RSI + StochRSI", "MACD", ""]
    )
    close = df["Close"].squeeze(); high = df["High"].squeeze(); low = df["Low"].squeeze()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(), high=high, low=low, close=close, name="주가",
        increasing_fillcolor="#4ade80", increasing_line_color="#4ade80",
        decreasing_fillcolor="#f87171", decreasing_line_color="#f87171",
    ), row=1, col=1)

    for ma, color in [("MA5","#60a5fa"),("MA20","#fbbf24"),("MA60","#a78bfa"),("MA120","#fb923c")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma].squeeze(), name=ma, line=dict(color=color, width=1.2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"].squeeze(), line=dict(color="#94a3b8",width=0.7,dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"].squeeze(), line=dict(color="#94a3b8",width=0.7,dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,0.04)", showlegend=False), row=1, col=1)

    if show_ichi:
        fig.add_trace(go.Scatter(x=df.index, y=df["Tenkan"].squeeze(), name="전환선", line=dict(color="#f43f5e",width=1.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Kijun"].squeeze(),  name="기준선", line=dict(color="#3b82f6",width=1.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SenkouA"].squeeze(), name="선행A", line=dict(color="#22c55e",width=0.5), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SenkouB"].squeeze(), name="선행B", line=dict(color="#ef4444",width=0.5), fill="tonexty", fillcolor="rgba(34,197,94,0.07)", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Chikou"].squeeze(), name="후행스팬", line=dict(color="#a78bfa",width=0.8,dash="dot")), row=1, col=1)

    if show_fib:
        try:
            ret, ext, sh, sl = calc_fibonacci(df)
            fib_colors = {"0%":"#f87171","23.6%":"#fb923c","38.2%":"#fbbf24","50%":"#a3e635","61.8%":"#4ade80","78.6%":"#34d399","100%":"#60a5fa"}
            for label, val in ret.items():
                fig.add_hline(y=val, line_dash="dot", line_color=fib_colors.get(label,"#fff"),
                              line_width=0.7, annotation_text=f" {label} {val:,.0f}",
                              annotation_position="right", annotation_font_size=9,
                              annotation_font_color=fib_colors.get(label,"#fff"), row=1, col=1)
        except: pass

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(), name="RSI", line=dict(color="#c084fc",width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["StochRSI_K"].squeeze(), name="StochK", line=dict(color="#38bdf8",width=1.0)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["StochRSI_D"].squeeze(), name="StochD", line=dict(color="#f472b6",width=1.0)), row=2, col=1)
    for lvl, clr in [(70,"#f87171"),(50,"#475569"),(30,"#4ade80")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, line_width=0.7, row=2, col=1)

    hist_vals  = df["MACD_hist"].squeeze().fillna(0).tolist()
    bar_colors = ["#4ade80" if v>=0 else "#f87171" for v in hist_vals]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"].squeeze(), name="MACD Hist", marker_color=bar_colors, opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(), name="MACD", line=dict(color="#60a5fa",width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"].squeeze(), name="Signal", line=dict(color="#fbbf24",width=1.2)), row=3, col=1)

    vol_vals = df["Volume"].squeeze().tolist(); close_vals = df["Close"].squeeze().tolist()
    vol_colors = ["#94a3b8"] + ["#4ade80" if close_vals[i] >= close_vals[i-1] else "#f87171" for i in range(1, len(close_vals))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(), name="거래량", marker_color=vol_colors, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_MA20"].squeeze(), name="Vol MA20", line=dict(color="#fbbf24",width=1.0)), row=4, col=1)

    fig.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#5a6070", size=10),
        xaxis_rangeslider_visible=False, height=760,
        margin=dict(l=0, r=110, t=20, b=0),
        legend=dict(bgcolor="#111318", bordercolor="#1e2130", borderwidth=1, orientation="h", x=0, y=1.01, font=dict(size=9)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#151820", font_size=12, font_family="DM Mono")
    )
    for row in [1,2,3,4]:
        fig.update_xaxes(gridcolor="#1a1e2a", row=row, col=1)
        fig.update_yaxes(gridcolor="#1a1e2a", row=row, col=1)
    return fig

def summarize(df):
    l, p = df.iloc[-1], df.iloc[-2]
    def f(col):
        try: return float(np.squeeze(l[col]))
        except: return 0.0
    close = f("Close")
    return dict(
        close=close, change_pct=(close/float(np.squeeze(p["Close"]))-1)*100,
        rsi=f("RSI"), macd=f("MACD"), macd_sig=f("MACD_signal"), macd_hist=f("MACD_hist"),
        stoch_k=f("StochRSI_K"), stoch_d=f("StochRSI_D"),
        ma5=f("MA5"), ma20=f("MA20"), ma60=f("MA60"), ma120=f("MA120"),
        bb_upper=f("BB_upper"), bb_lower=f("BB_lower"), bb_width=f("BB_width"),
        atr=f("ATR"), vol=f("Volume"), vol_ma20=f("Vol_MA20"),
        tenkan=f("Tenkan"), kijun=f("Kijun"),
        high_52w=float(df["High"].squeeze().rolling(252).max().iloc[-1]),
        low_52w=float(df["Low"].squeeze().rolling(252).min().iloc[-1]),
    )

def build_prompt(ticker, ind, period_label, cv_result, fib_info, w_summary=None):
    al = "정배열" if ind["ma5"]>ind["ma20"]>ind["ma60"] else "역배열" if ind["ma5"]<ind["ma20"]<ind["ma60"] else "혼조"
    vr = ind["vol"]/ind["vol_ma20"] if ind["vol_ma20"] else 1
    sig_text = "\n".join([f"  [{s[0]}] {s[1]}: {s[2]} (점수:{s[3]})" for s in cv_result["signals"]])
    w_text = ""
    if w_summary:
        w_text = f"""
=== 주봉 지표 (대세 확인) ===
주봉 현재가: {w_summary.get('close',0):,.0f}
주봉 MA20: {w_summary.get('ma20',0):,.0f} / MA60: {w_summary.get('ma60',0):,.0f}
주봉 RSI: {w_summary.get('rsi',0):.1f}
주봉 MACD Hist: {w_summary.get('macd_hist',0):+.4f}
주봉 전환선: {w_summary.get('tenkan',0):,.0f} / 기준선: {w_summary.get('kijun',0):,.0f}"""

    return f"""당신은 엘리어트 파동 + 일목균형표 + 멀티타임프레임 분석 전문가입니다.
주식 차트는 모든 시장 정보가 이미 반영되어 있다는 전제로, 기술적 지표만으로 분석해주세요.

=== 교차검증 결과 (신뢰도: {cv_result['confidence']}%) ===
종합 판정: {cv_result['verdict']}
매수 신호 일치: {cv_result['buy_count']}/5 기법
{sig_text}

=== 일봉 지표 ({period_label}) ===
현재가: {ind['close']:,.0f} ({ind['change_pct']:+.2f}%)
52주 고가: {ind['high_52w']:,.0f} / 저가: {ind['low_52w']:,.0f}
이동평균: MA5={ind['ma5']:,.0f} MA20={ind['ma20']:,.0f} MA60={ind['ma60']:,.0f} [{al}]
RSI: {ind['rsi']:.1f} | StochRSI K: {ind['stoch_k']:.1f}
MACD Hist: {ind['macd_hist']:+.4f}
볼린저: 상단={ind['bb_upper']:,.0f} 하단={ind['bb_lower']:,.0f} 밴드폭={ind['bb_width']:.3f}
일목 전환선: {ind['tenkan']:,.0f} / 기준선: {ind['kijun']:,.0f}
ATR: {ind['atr']:,.0f} | 거래량: 평균 대비 {vr:.1f}배
{w_text}

=== 피보나치 ===
{fib_info}

=== 분석 요청 ===
1. 🌊 엘리어트 파동 위치
   - 현재 파동 위치와 근거 (교차검증 결과 참고)
   - 다음 예상 시나리오 (강세/약세)

2. ☁️ 일목균형표 해석
   - 구름 위치, 전환/기준선 관계
   - 지지/저항 레벨

3. 📅 멀티타임프레임 종합
   - 주봉(대세) vs 일봉(단기) 방향 일치 여부
   - 불일치 시 어느 쪽 우선할지

4. 📐 피보나치 매매 전략
   - 현재 피보나치 위치
   - 구체적 진입가, 손절가 (ATR={ind['atr']:,.0f} 기준), 1차/2차 목표가

5. ✅ 최종 종합 판단
   - 신뢰도 {cv_result['confidence']}% 기반 매수/매도/관망
   - 이 분석이 틀릴 수 있는 조건 (파동 무효화 레벨)
"""

def score_stock(name, ticker):
    try:
        df = get_stock_data(ticker, "6mo")
        if df is None or len(df) < 60: return None
        df = calc_indicators(df)
        df_w = get_weekly_data(ticker)
        cv = cross_validate(df, df_w)
        close = float(df["Close"].squeeze().iloc[-1])
        atr   = float(df["ATR"].iloc[-1])
        bb_lower = float(df["BB_lower"].iloc[-1])
        target_buy  = round(max(bb_lower, close*0.97), 0)
        target_sell = round(close + atr*3, 0)
        return dict(
            ticker=ticker, name=name,
            score=cv["confidence"], signals=cv["signals"],
            verdict=cv["verdict"], verdict_color=cv["verdict_color"],
            buy_count=cv["buy_count"],
            cur=close, atr=atr,
            bb_lower=bb_lower, bb_upper=float(df["BB_upper"].iloc[-1]),
            rsi=float(df["RSI"].iloc[-1]),
            target_buy=target_buy, target_sell=target_sell, df=df,
        )
    except:
        return None

def build_screening_chart(r):
    df = r["df"]
    close = df["Close"].squeeze()
    try: ret, ext, sh, sl = calc_fibonacci(df)
    except: ret, ext = {}, {}
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(), low=df["Low"].squeeze(), close=close, name=r["name"], increasing_fillcolor="#4ade80", increasing_line_color="#4ade80", decreasing_fillcolor="#f87171", decreasing_line_color="#f87171"))
    for span,color,label in [(20,"#fbbf24","MA20"),(60,"#a78bfa","MA60")]: fig.add_trace(go.Scatter(x=df.index, y=close.rolling(span).mean(), name=label, line=dict(color=color,width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["Tenkan"].squeeze(), name="전환선", line=dict(color="#f43f5e",width=1.0)))
    fig.add_trace(go.Scatter(x=df.index, y=df["Kijun"].squeeze(), name="기준선", line=dict(color="#3b82f6",width=1.0)))
    if "61.8%" in ret: fig.add_hline(y=ret["61.8%"], line_dash="dot", line_color="#4ade80", line_width=0.8, annotation_text=f" Fib 61.8% {ret['61.8%']:,.0f}", annotation_font_color="#4ade80", annotation_font_size=9)
    fig.add_hline(y=r["target_buy"],  line_color="#4ade80", line_width=2, line_dash="dash", annotation_text=f"  매수가 {r['target_buy']:,.0f}", annotation_position="right", annotation_font_color="#4ade80")
    fig.add_hline(y=r["target_sell"], line_color="#f87171", line_width=2, line_dash="dash", annotation_text=f"  목표가 {r['target_sell']:,.0f}", annotation_position="right", annotation_font_color="#f87171")
    fig.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(color="#5a6070",size=10), xaxis_rangeslider_visible=False, height=380, margin=dict(l=0,r=130,t=10,b=0), legend=dict(bgcolor="#111318",bordercolor="#1e2130",borderwidth=1,orientation="h",x=0,y=1.08,font=dict(size=9)))
    fig.update_xaxes(gridcolor="#1a1e2a"); fig.update_yaxes(gridcolor="#1a1e2a")
    return fig

# ══════════════════════════════════════
# 세션 초기화
# ══════════════════════════════════════
if "mode" not in st.session_state:
    st.session_state.mode = "home"
if "screen_result" not in st.session_state:
    st.session_state.screen_result = None

# ══════════════════════════════════════
# 사이드바
# ══════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 차트 분석기")
    
    # ⭐ 여기에 데이터 갱신 시간을 표시할 '빈 공간(Placeholder)'을 미리 만들어 둡니다.
    refresh_placeholder = st.empty() 
    
    st.markdown("---")
    ticker_input = st.text_input("종목 코드", value="005930.KS", help="삼성전자, HPSP, 애플 등 이름도 가능")
    timeframe_map = {
        "일봉": {"period":"6mo",  "interval":"1d",  "label":"일봉 (6개월)"},
        "주봉": {"period":"2y",   "interval":"1wk", "label":"주봉 (2년)"},
        "월봉": {"period":"5y",   "interval":"1mo", "label":"월봉 (5년)"},
        "연봉": {"period":"max",  "interval":"3mo", "label":"연봉 (전체)"},
    }
    period_label = st.selectbox("차트 봉", list(timeframe_map.keys()), index=0)
    tf           = timeframe_map[period_label]
    period       = tf["period"]
    interval     = tf["interval"]
    show_fib  = st.toggle("피보나치 레벨", value=True)
    show_ichi = st.toggle("일목균형표",   value=True)
    api_key   = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="console.groq.com 무료 발급")
    st.markdown("---")
    if st.button("🏠 홈으로",             use_container_width=True):
        st.session_state.mode = "home"; st.rerun()
    if st.button("🔍 종목 분석",         use_container_width=True):
        st.session_state.mode = "analyze"; st.rerun()
    if st.button("🏆 매수 Top5 추천",    use_container_width=True):
        st.session_state.mode = "screen"
        st.session_state.screen_result = None; st.rerun()
    st.markdown("<div style='font-size:11px;color:#2a3040;margin-top:12px;line-height:1.8;'>삼성전자 005930.KS<br>SK하이닉스 000660.KS<br>HPSP 403870.KQ<br>애플 AAPL / 엔비디아 NVDA</div>", unsafe_allow_html=True)

# ══════════════════════════════════════
# 홈
# ══════════════════════════════════════
if st.session_state.mode == "home":
    @st.cache_data(ttl=300)
    def get_all_home_data():
        index_tickers = {
            "코스피":"^KS11","코스닥":"^KQ11","나스닥":"^IXIC",
            "S&P500":"^GSPC","달러/원":"KRW=X","VIX":"^VIX","금":"GC=F","원유":"CL=F",
        }
        fund_tickers = {
            "SPY(미국전체)":"SPY","QQQ(나스닥100)":"QQQ",
            "EEM(신흥국)":"EEM","IEF(미국채10Y)":"IEF",
        }
        sector_tickers = {
            "기술(XLK)":"XLK","헬스케어(XLV)":"XLV","금융(XLF)":"XLF",
            "에너지(XLE)":"XLE","산업재(XLI)":"XLI","소비재(XLY)":"XLY",
            "필수소비재(XLP)":"XLP","유틸리티(XLU)":"XLU","부동산(XLRE)":"XLRE",
            "소재(XLB)":"XLB","통신(XLC)":"XLC",
        }
        sector_stocks = {
            "XLK":{"leader":["NVDA","MSFT","AAPL"],"dark":["PLTR","ARM","SMCI"]},
            "XLV":{"leader":["LLY","UNH","JNJ"],   "dark":["RXRX","TVTX","NUVL"]},
            "XLF":{"leader":["BRK-B","JPM","V"],   "dark":["HOOD","SOFI","AFRM"]},
            "XLE":{"leader":["XOM","CVX","SLB"],   "dark":["SM","CIVI","MGY"]},
            "XLI":{"leader":["GE","CAT","HON"],    "dark":["KTOS","HII","DRS"]},
            "XLY":{"leader":["AMZN","TSLA","MCD"], "dark":["RIVN","LCID","NKLA"]},
            "XLP":{"leader":["WMT","PG","KO"],     "dark":["COTY","SFM","GO"]},
            "XLU":{"leader":["NEE","DUK","SO"],    "dark":["VST","NRG","AES"]},
            "XLRE":{"leader":["PLD","AMT","EQIX"], "dark":["IIPR","COLD","REXR"]},
            "XLB":{"leader":["LIN","APD","SHW"],   "dark":["MP","ALTM","CTRA"]},
            "XLC":{"leader":["META","GOOGL","DIS"],"dark":["RDDT","SNAP","PINS"]},
        }
        def fetch(ticker, period="1mo"):
            try:
                df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                if df is None or df.empty: return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df.dropna(how="all")
            except: return None

        indices = {}
        for name, t in index_tickers.items():
            df = fetch(t, "5d")
            if df is None: continue
            close = df["Close"].squeeze().dropna()
            if len(close) < 2: continue
            cur = float(close.iloc[-1]); prev = float(close.iloc[-2])
            indices[name] = {"cur":cur, "chg":(cur/prev-1)*100}

        fund_flow = {}
        for name, t in fund_tickers.items():
            df = fetch(t, "1mo")
            if df is None: continue
            close = df["Close"].squeeze().dropna()
            vol   = df["Volume"].squeeze().dropna()
            turnover = (close * vol / 1e8).dropna()
            if len(turnover) < 10: continue
            recent_avg = float(turnover.iloc[-5:].mean())
            prev_avg   = float(turnover.iloc[-15:-5].mean())
            chg_pct = (recent_avg/prev_avg-1)*100 if prev_avg else 0
            price_chg = float((close.iloc[-1]/close.iloc[0]-1)*100)
            fund_flow[name] = {"recent_vol":recent_avg,"vol_chg":chg_pct,"price_chg":price_chg}

        sector_data = {}
        for name, t in sector_tickers.items():
            df1w = fetch(t, "5d"); df1m = fetch(t, "1mo")
            if df1w is None or df1m is None: continue
            c1w = df1w["Close"].squeeze().dropna()
            c1m = df1m["Close"].squeeze().dropna()
            if len(c1w)<2 or len(c1m)<2: continue
            ret1w = float((c1w.iloc[-1]/c1w.iloc[0]-1)*100)
            ret1m = float((c1m.iloc[-1]/c1m.iloc[0]-1)*100)
            sector_data[name] = {"ticker":t,"ret1w":ret1w,"ret1m":ret1m,"stocks":sector_stocks.get(t,{})}

        stock_perf = {}
        all_stocks = set()
        for v in sector_stocks.values():
            all_stocks.update(v.get("leader",[])); all_stocks.update(v.get("dark",[]))
        for s in list(all_stocks)[:30]:
            df = fetch(s, "1mo")
            if df is None: continue
            close = df["Close"].squeeze().dropna()
            if len(close) < 2: continue
            stock_perf[s] = float((close.iloc[-1]/close.iloc[0]-1)*100)
            
        # ⭐ 데이터를 전부 수집한 시점의 '현재 시간(KST)'을 추가로 반환합니다.
        fetch_time = datetime.now(KST)
        return indices, fund_flow, sector_data, stock_perf, fetch_time

    with st.spinner("🌐 글로벌 시장 데이터 수집 중... (최초 30초, 이후 5분 캐시)"):
        indices, fund_flow, sector_data, stock_perf, fetch_time = get_all_home_data()

   # ⭐ 사이드바에 만들어둔 빈 공간에 업데이트 시간을 렌더링합니다. (다음 업데이트 = 현재 + 5분)
    next_update = fetch_time + timedelta(seconds=300)
    
    # ⬇️ 여기서부터 덮어씌워 주세요 ⬇️
    fetch_time_str = fetch_time.strftime('%H:%M:%S')
    next_update_str = next_update.strftime('%H:%M:%S')
    next_update_iso = next_update.isoformat() # ⭐ 자바스크립트로 시간을 넘기기 위한 변수 추가

    clock_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;700&display=swap');
        body {{ margin: 0; padding: 0; background-color: transparent; font-family: 'Noto Sans KR', sans-serif; }}
        .time-panel {{ background:#1a1e2a; border:1px solid #2a3040; border-radius:8px; padding:12px; }}
        .title {{ font-size:11px; color:#94a3b8; margin-bottom:8px; font-weight:400; }}
        .row {{ font-size:12px; color:#f0f2f8; display:flex; justify-content:space-between; margin-bottom:5px; align-items:center; }}
        .clock {{ color:#60a5fa; font-family:'DM Mono', monospace; font-weight:700; font-size:13px; }}
        .fetch {{ color:#4ade80; font-family:'DM Mono', monospace; font-weight:700; font-size:13px; }}
        .next {{ color:#fbbf24; font-family:'DM Mono', monospace; font-weight:700; font-size:13px; }}
    </style>
    <div class="time-panel">
        <div class="title">🕒 시간 정보 & 데이터 자동 갱신</div>
        <div class="row">
            <span>현재 시간</span>
            <span class="clock" id="live-clock">--:--:--</span>
        </div>
        <div class="row">
            <span>마지막 업데이트</span>
            <span class="fetch">{fetch_time_str}</span>
        </div>
        <div class="row" style="margin-bottom:0;">
            <span>다음 업데이트</span>
            <span class="next">{next_update_str}</span>
        </div>
    </div>
    <script>
        // 파이썬에서 넘겨준 '다음 업데이트' 시간을 자바스크립트가 인식합니다.
        const targetTime = new Date("{next_update_iso}").getTime();

        function updateTime() {{
            const now = new Date();
            // 한국 시간(KST) 계산
            const kstOffset = 9 * 60; 
            const localOffset = now.getTimezoneOffset();
            const kstTime = new Date(now.getTime() + (kstOffset + localOffset) * 60000);
            
            const hh = String(kstTime.getHours()).padStart(2, '0');
            const mm = String(kstTime.getMinutes()).padStart(2, '0');
            const ss = String(kstTime.getSeconds()).padStart(2, '0');
            document.getElementById('live-clock').innerText = hh + ':' + mm + ':' + ss;

            // ⭐ 핵심 로직: 현재 시간이 '다음 업데이트' 시간에 도달하면 화면 자동 새로고침!
            // (파이썬 캐시가 확실히 만료되도록 목표 시간보다 3초 뒤에 새로고침을 실행합니다)
            if (now.getTime() >= targetTime + 3000) {{
                window.parent.location.reload();
            }}
        }}
        setInterval(updateTime, 1000);
        updateTime();
    </script>
    """
    
    with refresh_placeholder:
        components.html(clock_html, height=120)
    # ⬆️ 여기까지 덮어씌워 주세요 ⬆️

    # 1. 글로벌 투자자금 흐름
    st.markdown("## 💰 글로벌 투자자금 흐름")
    st.caption("주요 ETF 거래대금 추세로 자금 증감 추정")
    if fund_flow:
        cols=st.columns(len(fund_flow))
        for col,(name,data) in zip(cols,fund_flow.items()):
            vc=data["vol_chg"]; pc=data["price_chg"]
            if vc>5 and pc>0: status,sc="자금 유입 ↑","#4ade80"
            elif vc<-5 and pc<0: status,sc="자금 유출 ↓","#f87171"
            elif vc>5 and pc<0: status,sc="매도 급증 ⚠","#fb923c"
            else: status,sc="보합","#94a3b8"
            vc_c="#4ade80" if vc>0 else "#f87171"; pc_c="#4ade80" if pc>0 else "#f87171"
            col.markdown(f"""<div class='hover-card' style='background:#151820;border:1px solid #1e2130;border-radius:10px;padding:14px 12px;'>
  <div style='font-size:11px;color:#4a5060;margin-bottom:6px;'>{name}</div>
  <div style='font-size:13px;font-weight:700;color:{sc};margin-bottom:8px;'>{status}</div>
  <div style='font-size:11px;color:#5a6070;'>거래대금 <span style='color:{vc_c};'>{"▲" if vc>0 else "▼"}{abs(vc):.1f}%</span></div>
  <div style='font-size:11px;color:#5a6070;'>수익률 <span style='color:{pc_c};'>{"▲" if pc>0 else "▼"}{abs(pc):.1f}%</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 2. 자금이 몰리는 섹터 Top 5
    st.markdown("## 🏭 자금이 몰리는 섹터 Top 5")
    st.caption("미국 섹터 ETF 1주 수익률 기준 · 순위 변동 표시")
    if sector_data:
        sorted_sectors=sorted(sector_data.items(),key=lambda x:x[1]["ret1w"],reverse=True)
        top5=sorted_sectors[:5]; worst1=sorted_sectors[-1]
        max_ret=max(abs(v["ret1w"]) for _,v in sorted_sectors) or 1
        rank1m_list=sorted(sector_data.items(),key=lambda x:x[1]["ret1m"],reverse=True)
        for rank,(name,data) in enumerate(top5,1):
            ret1w=data["ret1w"]; ret1m=data["ret1m"]
            bar_w=int(abs(ret1w)/max_ret*100); bar_color="#4ade80" if ret1w>0 else "#f87171"
            rank1m=next((i+1 for i,(n,_) in enumerate(rank1m_list) if n==name),0)
            rank_diff=rank1m-rank
            if rank_diff>2:   momentum,m_color=f"🚀 급상승 +{rank_diff}계단","#4ade80"
            elif rank_diff>0: momentum,m_color=f"↗ 상승 +{rank_diff}계단","#86efac"
            elif rank_diff==0: momentum,m_color="→ 유지","#94a3b8"
            elif rank_diff>-3: momentum,m_color=f"↘ 하락 {rank_diff}계단","#fb923c"
            else: momentum,m_color=f"📉 급락 {rank_diff}계단","#f87171"
            rank_emoji=["🥇","🥈","🥉","4️⃣","5️⃣"][rank-1]
            stocks_info=data.get("stocks",{}); leaders=stocks_info.get("leader",[]); darks=stocks_info.get("dark",[])
            def badge(t,dark=False):
                p=stock_perf.get(t,0); c="#4ade80" if p>0 else "#f87171"; s="▲" if p>0 else "▼"
                b="#fbbf24" if dark else "#1e2130"
                return f"<span style='background:#151820;border:1px solid {b};border-radius:6px;padding:3px 8px;font-size:11px;color:{c};margin-right:4px;'>{t} {s}{abs(p):.1f}%</span>"
            lb="".join([badge(s) for s in leaders]); db="".join([badge(s,True) for s in darks])
            st.markdown(f"""<div class='hover-card' style='background:#151820;border:1px solid #1e2130;border-radius:12px;padding:16px 20px;margin-bottom:10px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
    <div><span style='font-size:16px;'>{rank_emoji}</span><span style='font-size:15px;font-weight:700;color:#f0f2f8;margin-left:8px;'>{name}</span><span style='font-size:12px;color:{m_color};margin-left:12px;'>{momentum}</span></div>
    <div style='text-align:right;'><span style='font-size:14px;font-weight:700;color:{bar_color};font-family:DM Mono,monospace;'>{"▲" if ret1w>0 else "▼"}{abs(ret1w):.2f}%</span><span style='font-size:11px;color:#4a5060;margin-left:8px;'>1주 / 1달 {"▲" if ret1m>0 else "▼"}{abs(ret1m):.1f}%</span></div>
  </div>
  <div style='background:#1a1e2a;border-radius:4px;height:6px;margin-bottom:12px;'><div style='background:{bar_color};height:6px;border-radius:4px;width:{bar_w}%;'></div></div>
  <div style='margin-bottom:6px;'><span style='font-size:10px;color:#4a5060;margin-right:8px;'>👑 선두</span>{lb}</div>
  <div><span style='font-size:10px;color:#fbbf24;margin-right:8px;'>⚡ 다크호스</span>{db}</div>
</div>""", unsafe_allow_html=True)
        nw,dw=worst1; rw=dw["ret1w"]
        st.markdown(f"<div style='background:#1a0d0d;border:1px solid #4a1a1a;border-radius:10px;padding:12px 16px;font-size:12px;color:#f87171;'>📉 자금 이탈 섹터: <b>{nw}</b> — {rw:.2f}% (1주)</div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # 3. 글로벌 시장 현황 (맨 마지막으로 이동)
    st.markdown("## 🌍 글로벌 시장 현황")
    st.caption("yfinance 기준 · 5분 자동 갱신")
    icons = {"코스피":"🇰🇷","코스닥":"📊","나스닥":"🇺🇸","S&P500":"🗽","달러/원":"💵","VIX":"😨","금":"🥇","원유":"🛢️"}
    if indices:
        cols = st.columns(len(indices))
        for col,(name,data) in zip(cols,indices.items()):
            chg=data["chg"]; color="#4ade80" if chg>=0 else "#f87171"; sign="▲" if chg>=0 else "▼"
            col.markdown(f"""<div class='hover-card' style='background:#151820;border:1px solid #1e2130;border-radius:10px;padding:14px 10px;text-align:center;'>
  <div style='font-size:18px;'>{icons.get(name,"📈")}</div>
  <div style='font-size:10px;color:#4a5060;margin:4px 0;'>{name}</div>
  <div style='font-size:16px;font-weight:700;color:#f0f2f8;font-family:DM Mono,monospace;'>{data["cur"]:,.1f}</div>
  <div style='font-size:12px;color:{color};font-weight:600;'>{sign}{abs(chg):.2f}%</div>
</div>""", unsafe_allow_html=True)

    if indices:
        vix=indices.get("VIX",{}).get("cur",20); usd_chg=indices.get("달러/원",{}).get("chg",0)
        nasdaq_chg=indices.get("나스닥",{}).get("chg",0); gold_chg=indices.get("금",{}).get("chg",0)
        fg_score=max(0,min(100,int(100-(vix-10)*2.5)))
        if fg_score>=75:   fg_label,fg_color="극도의 탐욕 😍","#4ade80"
        elif fg_score>=55: fg_label,fg_color="탐욕 😊","#86efac"
        elif fg_score>=45: fg_label,fg_color="중립 😐","#fbbf24"
        elif fg_score>=25: fg_label,fg_color="공포 😨","#fb923c"
        else:              fg_label,fg_color="극도의 공포 😱","#f87171"
        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2=st.columns([1,2])
        with c1:
            st.markdown(f"""<div class='hover-card' style='background:#151820;border:1px solid #1e2130;border-radius:12px;padding:20px;text-align:center;'>
  <div style='font-size:11px;color:#4a5060;margin-bottom:8px;'>공포 & 탐욕 지수</div>
  <div style='font-size:48px;font-weight:800;color:{fg_color};font-family:DM Mono,monospace;'>{fg_score}</div>
  <div style='font-size:14px;color:{fg_color};margin-top:4px;font-weight:600;'>{fg_label}</div>
  <div style='font-size:10px;color:#3a4050;margin-top:8px;'>VIX {vix:.1f} 기반</div>
</div>""", unsafe_allow_html=True)
        with c2:
            comments=[]
            if gold_chg>0.5 and usd_chg>0.3: comments.append("🔴 **안전자산 동반 강세** — 위험 회피, 주식 비중 축소 고려")
            elif gold_chg<-0.3 and nasdaq_chg>0.5: comments.append("🟢 **위험자산 선호** — 성장주 유리")
            if usd_chg>0.5: comments.append("💵 **원화 약세** → 외국인 매도 압력, 수출 대형주 주목")
            elif usd_chg<-0.5: comments.append("💵 **원화 강세** → 외국인 유입 기대")
            if nasdaq_chg>1: comments.append("🇺🇸 **나스닥 강세** → 다음 거래일 코스닥 상승 기대")
            elif nasdaq_chg<-1: comments.append("🇺🇸 **나스닥 약세** → 다음 거래일 코스닥 하락 주의")
            if not comments: comments.append("😐 특별한 시그널 없음")
            st.markdown("**💬 시장 해석**")
            for c in comments: st.markdown(f"• {c}")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#2a3040;font-size:13px;padding:12px 0;'>엘리어트 파동 × 일목균형표 × 멀티타임프레임 교차검증 플랫폼 · 사이드바에서 <b style='color:#4ade80;'>종목 분석</b> 또는 <b style='color:#4ade80;'>Top5 추천</b> 선택</div>", unsafe_allow_html=True)

# ══════════════════════════════════════
# 개별 종목 분석
# ══════════════════════════════════════
elif st.session_state.mode == "analyze":
    ticker = search_ticker(ticker_input)
    with st.spinner(f"{ticker} 일봉+주봉 데이터 수집 중..."):
        df_d = get_stock_data(ticker, period, interval)
        df_w = get_weekly_data(ticker)

    if df_d is None or len(df_d) < 30:
        st.error(f"'{ticker}' 데이터를 불러올 수 없습니다.")
    else:
        df_d = calc_indicators(df_d)
        ind  = summarize(df_d)
        cv   = cross_validate(df_d, df_w)

        # 주봉 요약
        w_summary = None
        if df_w is not None and len(df_w) >= 26:
            try:
                df_w = calc_indicators(df_w)
                w_summary = summarize(df_w)
            except: pass

        # ── 신뢰도 대형 배너
        conf = cv["confidence"]
        conf_color = cv["verdict_color"]
        bar_pct = conf
        st.markdown(f"""
<div style='background:#151820;border:2px solid {conf_color};border-radius:14px;padding:20px 28px;margin-bottom:20px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <div style='font-size:11px;color:#4a5060;letter-spacing:0.15em;margin-bottom:4px;'>교차검증 신뢰도 ({cv["buy_count"]}/5 기법 매수 신호)</div>
      <div style='font-size:28px;font-weight:800;color:{conf_color};font-family:DM Mono,monospace;'>{conf}%</div>
    </div>
    <div style='font-size:22px;font-weight:800;color:{conf_color};'>{cv["verdict"]}</div>
  </div>
  <div style='background:#1a1e2a;border-radius:4px;height:8px;margin-top:12px;'>
    <div style='background:{conf_color};height:8px;border-radius:4px;width:{bar_pct}%;transition:width 0.5s;'></div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── 5가지 기법 신호 카드
        st.markdown("#### 📊 5가지 기법 교차검증")
        sig_cols = st.columns(5)
        direction_colors = {"매수":"#4ade80","강한매수":"#4ade80","추세추종매수":"#86efac",
                            "관망→매수준비":"#fbbf24","관망":"#94a3b8","관망/매도검토":"#f87171",
                            "데이터부족":"#475569","계산오류":"#475569","대기":"#94a3b8","참고":"#94a3b8"}
        for col, sig in zip(sig_cols, cv["signals"]):
            icon, name = sig[0].split(" ",1)
            direction = sig[1]; detail = sig[2]; pts = sig[3]
            dc = direction_colors.get(direction, "#94a3b8")
            col.markdown(f"""<div class='hover-card' style='background:#151820;border:1px solid #1e2130;border-left:3px solid {dc};border-radius:8px;padding:12px;height:140px;overflow:hidden;'>
  <div style='font-size:16px;margin-bottom:4px;'>{icon}</div>
  <div style='font-size:10px;color:#4a5060;margin-bottom:6px;'>{name}</div>
  <div style='font-size:12px;font-weight:700;color:{dc};margin-bottom:6px;'>{direction}</div>
  <div style='font-size:10px;color:#5a6070;line-height:1.4;'>{detail[:60]}{"..." if len(detail)>60 else ""}</div>
  <div style='font-size:10px;color:#3a4050;margin-top:6px;'>점수: {pts}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 지표 카드
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        vr = ind["vol"]/ind["vol_ma20"] if ind["vol_ma20"] else 1
        al = "정배열" if ind["ma5"]>ind["ma20"]>ind["ma60"] else "역배열" if ind["ma5"]<ind["ma20"]<ind["ma60"] else "혼조"
        with c1: st.metric("현재가",    f"{ind['close']:,.0f}",    f"{ind['change_pct']:+.2f}%")
        with c2: st.metric("RSI",       f"{ind['rsi']:.1f}",       ">70과매수 / <30과매도")
        with c3: st.metric("StochRSI K",f"{ind['stoch_k']:.1f}",   ">80과매수 / <20과매도")
        with c4: st.metric("전환선",    f"{ind['tenkan']:,.0f}",    "일목 단기 추세")
        with c5: st.metric("기준선",    f"{ind['kijun']:,.0f}",     "일목 중기 추세")
        with c6: st.metric("ATR",       f"{ind['atr']:,.0f}",       "변동성 기준")

        # 다이버전스
        if cv["divs"]:
            for d_type, d_msg in cv["divs"]:
                color = "#fbbf24" if "약세" in d_type else "#4ade80"
                st.markdown(f"<div style='background:#1a1a0d;border:1px solid {color};border-radius:8px;padding:8px 16px;margin:4px 0;color:{color};font-size:13px;'>🔔 <b>{d_type}</b> — {d_msg}</div>", unsafe_allow_html=True)

        # 피보나치 요약바
        try:
            ret, ext, sh, sl = calc_fibonacci(df_d)
            st.markdown(f"<div style='background:#151820;border:1px solid #1e2130;border-radius:8px;padding:10px 16px;font-size:12px;color:#94a3b8;margin:4px 0;'>📐 <b>피보나치</b> — 고점 <b style='color:#f87171;'>{sh:,.0f}</b> / 저점 <b style='color:#4ade80;'>{sl:,.0f}</b> | 38.2% <b style='color:#fbbf24;'>{ret['38.2%']:,.0f}</b> | 61.8% <b style='color:#4ade80;'>{ret['61.8%']:,.0f}</b> | 161.8% 목표 <b style='color:#60a5fa;'>{ext['161.8%']:,.0f}</b></div>", unsafe_allow_html=True)
            fib_info = f"고점:{sh:,.0f} / 저점:{sl:,.0f}\n되돌림: " + " | ".join([f"{k}={v:,.0f}" for k,v in list(ret.items())[1:5]]) + "\n확장: " + " | ".join([f"{k}={v:,.0f}" for k,v in ext.items()])
        except:
            fib_info = "피보나치 계산 불가"

        # 차트
        fig = build_chart(df_d, ticker, show_fib=show_fib, show_ichi=show_ichi)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # 주봉 요약 (멀티타임프레임)
        if w_summary:
            w_al = "정배열" if w_summary["ma5"]>w_summary["ma20"]>w_summary["ma60"] else "역배열" if w_summary["ma5"]<w_summary["ma20"]<w_summary["ma60"] else "혼조"
            mtf_match = "✅ 일치" if (ind["ma5"]>ind["ma20"]) == (w_summary["ma5"]>w_summary["ma20"]) else "⚠️ 불일치"
            st.markdown(f"<div style='background:#151820;border:1px solid #1e2130;border-radius:8px;padding:10px 16px;font-size:12px;color:#94a3b8;'>📅 <b>주봉</b> — {w_al} | RSI {w_summary['rsi']:.1f} | 전환선 {w_summary['tenkan']:,.0f} / 기준선 {w_summary['kijun']:,.0f} | 일봉방향 {mtf_match}</div>", unsafe_allow_html=True)

        # AI 분석
        st.markdown("---")
        st.markdown("#### 🤖 AI 종합 분석 리포트")
        if not api_key:
            st.info("사이드바에 Groq API Key를 입력하면 AI 분석이 활성화됩니다.")
        else:
            prompt = build_prompt(ticker, ind, period_label, cv, fib_info, w_summary)
            box = st.empty(); full = ""
            try:
                client = Groq(api_key=api_key)
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role":"user","content":prompt}], stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full += delta
                    box.markdown(f'<div class="analysis-box">{full}▌</div>', unsafe_allow_html=True)
                box.markdown(f'<div class="analysis-box">{full}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI 분석 오류: {e}")

        with st.expander("📋 최근 20일 데이터"):
            show = df_d[["Open","High","Low","Close","Volume","RSI","StochRSI_K","MACD","ATR","Tenkan","Kijun"]].tail(20).copy()
            show.index = show.index.strftime("%Y-%m-%d")
            st.dataframe(show, use_container_width=True)

# ══════════════════════════════════════
# 스크리닝 Top5
# ══════════════════════════════════════
elif st.session_state.mode == "screen":
    st.markdown("## 🏆 교차검증 기반 매수 Top 5")
    st.caption("엘리어트 × 일목균형표 × 멀티타임프레임 × 피보나치 × 다이버전스 종합 신뢰도")

    if st.session_state.screen_result is None:
        total=len(KR_UNIVERSE); prog_bar=st.progress(0); status=st.empty(); results=[]
        for i,(name,ticker) in enumerate(KR_UNIVERSE.items()):
            prog_bar.progress((i+1)/total)
            status.markdown(f"🔍 **{name}** ({ticker}) 교차검증 중... `{i+1} / {total}`")
            r = score_stock(name, ticker)
            if r and r["score"] > 0: results.append(r)
        prog_bar.progress(1.0)
        status.markdown(f"✅ 완료! {len(results)}개 종목 분석 → 신뢰도 순 Top 5 선정")
        results.sort(key=lambda x: x["score"], reverse=True)
        st.session_state.screen_result = results[:5]
        st.rerun()
    else:
        if st.button("🔄 다시 스크리닝"): st.session_state.screen_result=None; st.rerun()
        rank_emoji=["🥇","🥈","🥉","4️⃣","5️⃣"]
        for i,r in enumerate(st.session_state.screen_result):
            conf=r["score"]; conf_color=r["verdict_color"]
            st.markdown(f"""<div class='hover-card' style='background:#151820;border:2px solid {conf_color};border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
    <div><span style='font-size:18px;'>{rank_emoji[i]}</span><span style='font-size:16px;font-weight:700;color:#f0f2f8;margin-left:8px;'>{r["name"]} ({r["ticker"]})</span></div>
    <div><span style='font-size:16px;font-weight:800;color:{conf_color};font-family:DM Mono,monospace;'>{conf}%</span><span style='font-size:13px;color:{conf_color};margin-left:8px;'>{r["verdict"]}</span></div>
  </div>
  <div style='background:#1a1e2a;border-radius:4px;height:6px;margin-bottom:12px;'><div style='background:{conf_color};height:6px;border-radius:4px;width:{conf}%;'></div></div>
</div>""", unsafe_allow_html=True)
            c1,c2,c3,c4,c5=st.columns(5)
            with c1: st.metric("현재가",f"{r['cur']:,.0f}")
            with c2: st.metric("신뢰도",f"{conf}%")
            with c3: st.metric("매수 신호",f"{r['buy_count']}/5")
            with c4: st.metric("매수 목표가",f"{r['target_buy']:,.0f}")
            with c5: st.metric("목표 매도가",f"{r['target_sell']:,.0f}")
            for sig in r["signals"]:
                dc={"매수":"🟢","추세추종매수":"🟢","관망→매수준비":"🟡","관망":"⚪","관망/매도검토":"🔴"}.get(sig[1],"⚪")
                st.markdown(f"<span style='font-size:12px;'>{dc} **{sig[0]}**: {sig[1]} — {sig[2]}</span>", unsafe_allow_html=True)
            st.plotly_chart(build_screening_chart(r), use_container_width=True, config={"displayModeBar":False})
            if i<4: st.markdown("---")
