import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq

st.set_page_config(page_title="차트 분석 어드바이저", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp { background-color: #0d0f14; }
[data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #1e2130; }
div.stButton > button { background: #4ade80; color: #0d1a0f; border: none; border-radius: 8px; font-weight: 700; font-size: 14px; padding: 10px 24px; width: 100%; margin-bottom: 8px; }
div.stButton > button:hover { background: #22c55e; }
.analysis-box { background: #111620; border: 1px solid #1e3040; border-left: 3px solid #4ade80; border-radius: 10px; padding: 20px 24px; font-size: 14px; line-height: 1.9; color: #c8d0e0; white-space: pre-wrap; }
.home-btn { position: fixed; top: 14px; left: 260px; z-index: 999; }
</style>
""", unsafe_allow_html=True)

KR_UNIVERSE = {
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS", "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS", "기아": "000270.KS",
    "셀트리온": "068270.KS", "카카오": "035720.KS",
    "네이버": "035420.KS", "LG화학": "051910.KS",
    "삼성SDI": "006400.KS", "POSCO홀딩스": "005490.KS",
    "KB금융": "105560.KS", "신한지주": "055550.KS",
    "하나금융": "086790.KS", "현대모비스": "012330.KS",
    "SK텔레콤": "017670.KS", "LG전자": "066570.KS",
    "크래프톤": "259960.KS", "엔씨소프트": "036570.KS",
    "한화에어로스페이스": "012450.KS", "HD현대중공업": "329180.KS",
    "HPSP": "403870.KQ", "에코프로": "086520.KQ",
    "에코프로비엠": "247540.KQ", "리노공업": "058470.KQ",
    "알테오젠": "196170.KQ", "HLB": "028300.KQ",
    "클래시스": "214150.KQ", "레인보우로보틱스": "277810.KQ",
    "이오테크닉스": "039030.KQ", "파크시스템스": "140860.KQ",
    "원익IPS": "240810.KQ", "피에스케이": "319660.KQ",
    "덕산네오룩스": "213580.KQ", "루닛": "328130.KQ",
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

def get_stock_data(ticker, period):
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        return raw.dropna(how="all")
    except:
        return None

def calc_indicators(df):
    close = df["Close"].squeeze()
    # 이동평균
    df["MA5"]  = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()
    # 볼린저밴드
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2*bb_std
    df["BB_lower"] = bb_mid - 2*bb_std
    df["BB_mid"]   = bb_mid
    df["BB_width"] = (df["BB_upper"]-df["BB_lower"])/bb_mid
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100/(1+rs)
    # 스토캐스틱 RSI
    rsi = df["RSI"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    df["StochRSI"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    df["StochRSI_K"] = df["StochRSI"].rolling(3).mean() * 100
    df["StochRSI_D"] = df["StochRSI_K"].rolling(3).mean()
    # MACD
    ema12 = close.ewm(span=12,adjust=False).mean()
    ema26 = close.ewm(span=26,adjust=False).mean()
    df["MACD"]        = ema12-ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_hist"]   = df["MACD"]-df["MACD_signal"]
    # ATR
    high = df["High"].squeeze()
    low  = df["Low"].squeeze()
    tr   = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    # 거래량
    df["Vol_MA20"] = df["Volume"].squeeze().rolling(20).mean()
    return df

def detect_divergence(df):
    """RSI 다이버전스 감지 (최근 30봉 기준)"""
    close = df["Close"].squeeze().values[-30:]
    rsi   = df["RSI"].values[-30:]
    results = []
    # 약세 다이버전스: 가격 고점 상승 + RSI 고점 하락
    price_highs = [i for i in range(1,len(close)-1) if close[i] > close[i-1] and close[i] > close[i+1]]
    if len(price_highs) >= 2:
        p1, p2 = price_highs[-2], price_highs[-1]
        if close[p2] > close[p1] and rsi[p2] < rsi[p1]:
            results.append(("약세 다이버전스", "⚠️ 가격 신고점 + RSI 하락 → 5파 완성 또는 반전 가능"))
    # 강세 다이버전스: 가격 저점 하락 + RSI 저점 상승
    price_lows = [i for i in range(1,len(close)-1) if close[i] < close[i-1] and close[i] < close[i+1]]
    if len(price_lows) >= 2:
        p1, p2 = price_lows[-2], price_lows[-1]
        if close[p2] < close[p1] and rsi[p2] > rsi[p1]:
            results.append(("강세 다이버전스", "✅ 가격 신저점 + RSI 상승 → 2파/4파 완료, 반등 가능"))
    return results

def calc_fibonacci(df):
    """최근 주요 고점/저점 기준 피보나치 레벨 계산"""
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    recent = 60
    h_idx = high.iloc[-recent:].idxmax()
    l_idx = low.iloc[-recent:].idxmin()
    swing_high = float(high[h_idx])
    swing_low  = float(low[l_idx])
    diff = swing_high - swing_low
    # 되돌림 레벨
    retracement = {
        "0%":    swing_high,
        "23.6%": swing_high - diff * 0.236,
        "38.2%": swing_high - diff * 0.382,
        "50%":   swing_high - diff * 0.500,
        "61.8%": swing_high - diff * 0.618,
        "78.6%": swing_high - diff * 0.786,
        "100%":  swing_low,
    }
    # 확장 레벨 (목표가)
    extension = {
        "127.2%": swing_low + diff * 1.272,
        "161.8%": swing_low + diff * 1.618,
        "200%":   swing_low + diff * 2.000,
        "261.8%": swing_low + diff * 2.618,
    }
    return retracement, extension, swing_high, swing_low, h_idx, l_idx

def build_chart(df, ticker, show_fib=True):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.18, 0.14],
        vertical_spacing=0.02
    )
    close = df["Close"].squeeze()
    # 캔들
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(), close=close, name="주가",
        increasing_fillcolor="#4ade80", increasing_line_color="#4ade80",
        decreasing_fillcolor="#f87171", decreasing_line_color="#f87171",
    ), row=1, col=1)
    # 이동평균
    for ma, color in [("MA5","#60a5fa"),("MA20","#fbbf24"),("MA60","#a78bfa"),("MA120","#fb923c")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma].squeeze(), name=ma, line=dict(color=color, width=1.2)), row=1, col=1)
    # 볼린저밴드
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"].squeeze(), line=dict(color="#94a3b8", width=0.8, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"].squeeze(), line=dict(color="#94a3b8", width=0.8, dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,0.05)", showlegend=False), row=1, col=1)

    # 피보나치 레벨
    if show_fib:
        try:
            ret, ext, sh, sl, h_idx, l_idx = calc_fibonacci(df)
            fib_colors = {"0%":"#f87171","23.6%":"#fb923c","38.2%":"#fbbf24","50%":"#a3e635","61.8%":"#4ade80","78.6%":"#34d399","100%":"#60a5fa"}
            for label, val in ret.items():
                fig.add_hline(y=val, line_dash="dot", line_color=fib_colors.get(label,"#ffffff"),
                              line_width=0.8, annotation_text=f" Fib {label} {val:,.0f}",
                              annotation_position="right", annotation_font_size=9,
                              annotation_font_color=fib_colors.get(label,"#ffffff"), row=1, col=1)
        except:
            pass

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(), name="RSI", line=dict(color="#c084fc", width=1.5)), row=2, col=1)
    for lvl, clr in [(70,"#f87171"),(50,"#475569"),(30,"#4ade80")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, line_width=0.8, row=2, col=1)

    # 스토캐스틱 RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["StochRSI_K"].squeeze(), name="StochK", line=dict(color="#38bdf8", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["StochRSI_D"].squeeze(), name="StochD", line=dict(color="#f472b6", width=1.2)), row=3, col=1)
    for lvl, clr in [(80,"#f87171"),(20,"#4ade80")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, line_width=0.8, row=3, col=1)

    # MACD
    hist_vals  = df["MACD_hist"].squeeze().fillna(0).tolist()
    bar_colors = ["#4ade80" if v>=0 else "#f87171" for v in hist_vals]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"].squeeze(), name="MACD Hist", marker_color=bar_colors, opacity=0.8), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(), name="MACD", line=dict(color="#60a5fa", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"].squeeze(), name="Signal", line=dict(color="#fbbf24", width=1.2)), row=4, col=1)

    fig.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(color="#5a6070", size=10),
        xaxis_rangeslider_visible=False, height=720,
        margin=dict(l=0, r=100, t=30, b=0),
        legend=dict(bgcolor="#111318", bordercolor="#1e2130", borderwidth=1, orientation="h", x=0, y=1.02)
    )
    for row in [1,2,3,4]:
        fig.update_xaxes(gridcolor="#1a1e2a", row=row, col=1)
        fig.update_yaxes(gridcolor="#1a1e2a", row=row, col=1)
    return fig

def build_screening_chart(r):
    df = r["df"]
    close = df["Close"].squeeze()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(), low=df["Low"].squeeze(), close=close, name=r["name"], increasing_fillcolor="#4ade80", increasing_line_color="#4ade80", decreasing_fillcolor="#f87171", decreasing_line_color="#f87171"))
    for span,color,label in [(20,"#fbbf24","MA20"),(60,"#a78bfa","MA60")]:
        fig.add_trace(go.Scatter(x=df.index, y=close.rolling(span).mean(), name=label, line=dict(color=color, width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid+2*bb_std, line=dict(color="#94a3b8", width=0.8, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid-2*bb_std, line=dict(color="#94a3b8", width=0.8, dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,0.05)", showlegend=False))
    # 피보나치
    try:
        ret, ext, sh, sl, h_idx, l_idx = calc_fibonacci(df)
        for label, val in list(ret.items())[2:5]:
            fig.add_hline(y=val, line_dash="dot", line_color="#4ade80", line_width=0.7,
                          annotation_text=f" Fib {label}", annotation_font_size=9, annotation_font_color="#4ade80")
    except:
        pass
    fig.add_hline(y=r["target_buy"], line_color="#4ade80", line_width=2, line_dash="dash",
                  annotation_text=f"  매수 목표가 {r['target_buy']:,.0f}", annotation_position="right", annotation_font_color="#4ade80")
    fig.add_hline(y=r["target_sell"], line_color="#f87171", line_width=2, line_dash="dash",
                  annotation_text=f"  목표 매도가 {r['target_sell']:,.0f}", annotation_position="right", annotation_font_color="#f87171")
    fig.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font=dict(color="#5a6070", size=11),
                      xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=130,t=10,b=0),
                      legend=dict(bgcolor="#111318", bordercolor="#1e2130", borderwidth=1, orientation="h", x=0, y=1.08))
    fig.update_xaxes(gridcolor="#1a1e2a")
    fig.update_yaxes(gridcolor="#1a1e2a")
    return fig

def score_stock(name, ticker):
    try:
        raw = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        if raw is None or len(raw) < 60: return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        df = raw.dropna(how="all")
        df = calc_indicators(df)
        close = df["Close"].squeeze()
        score = 0
        signals = []
        ma5  = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        cur  = float(close.iloc[-1])
        rsi  = float(df["RSI"].iloc[-1])
        hist = float(df["MACD_hist"].iloc[-1])
        hist_prev = float(df["MACD_hist"].iloc[-2])
        stoch_k = float(df["StochRSI_K"].iloc[-1])
        bb_lower = float(df["BB_lower"].iloc[-1])
        bb_upper = float(df["BB_upper"].iloc[-1])
        bb_width = float(df["BB_width"].iloc[-1])
        vol  = float(df["Volume"].squeeze().iloc[-1])
        vol_ma20 = float(df["Vol_MA20"].iloc[-1])
        vol_ratio = vol/vol_ma20 if vol_ma20 else 1
        atr = float(df["ATR"].iloc[-1])

        # RSI 과매도
        if rsi < 35:
            score += 30; signals.append(f"RSI 과매도({rsi:.0f}) — 2파/4파 완료 가능")
        elif rsi < 45:
            score += 15; signals.append(f"RSI 저점권({rsi:.0f}) — 반등 준비")
        # 스토캐스틱 RSI 과매도
        if stoch_k < 20:
            score += 15; signals.append(f"StochRSI 과매도({stoch_k:.0f}) — 단기 반전 신호")
        # MACD 골든크로스
        if hist > 0 and hist_prev <= 0:
            score += 25; signals.append("MACD 골든크로스 — 3파/5파 시작 신호")
        elif hist > 0 and hist > hist_prev:
            score += 10; signals.append("MACD 히스토그램 상승 중")
        # 볼린저 하단
        if cur <= bb_lower*1.02:
            score += 20; signals.append("볼린저 하단 근접 — 과매도 반등 구간")
        elif cur <= bb_lower*1.05:
            score += 10
        # 이동평균 배열
        if ma5 > ma20 and ma20 < ma60:
            score += 15; signals.append("단기 반등 + 중기 눌림 — 3파 준비")
        elif ma5 > ma20 > ma60:
            score += 10; signals.append("정배열 — 상승 추세")
        # 거래량
        if vol_ratio > 2.0:
            score += 10; signals.append(f"거래량 급증({vol_ratio:.1f}x) — 세력 유입 가능")
        # 밴드 수축
        if bb_width < 0.08:
            score += 10; signals.append("밴드 수축 — 변동성 폭발 임박")
        # 다이버전스
        divs = detect_divergence(df)
        for d_type, d_msg in divs:
            if d_type == "강세 다이버전스":
                score += 20; signals.append(d_msg)
        # 피보나치 지지
        try:
            ret, _, _, _, _, _ = calc_fibonacci(df)
            fib618 = ret["61.8%"]
            fib50  = ret["50%"]
            if fib618*0.99 <= cur <= fib618*1.01:
                score += 15; signals.append(f"피보나치 61.8% 지지({fib618:,.0f}) — 핵심 되돌림 구간")
            elif fib50*0.99 <= cur <= fib50*1.01:
                score += 10; signals.append(f"피보나치 50% 지지({fib50:,.0f})")
        except:
            pass

        target_buy  = round(max(bb_lower, cur*0.97), 0)
        target_sell = round(cur + atr*3, 0)  # ATR 기반 목표가
        return dict(ticker=ticker, name=name, score=score, signals=signals,
                    cur=cur, rsi=rsi, macd_hist=hist,
                    bb_lower=bb_lower, bb_upper=bb_upper,
                    target_buy=target_buy, target_sell=target_sell, df=df, atr=atr)
    except:
        return None

def summarize(df):
    l, p = df.iloc[-1], df.iloc[-2]
    def f(col): return float(np.squeeze(l[col]))
    close = f("Close")
    return dict(
        close=close, change_pct=(close/float(np.squeeze(p["Close"]))-1)*100,
        rsi=f("RSI"), macd=f("MACD"), macd_sig=f("MACD_signal"), macd_hist=f("MACD_hist"),
        stoch_k=f("StochRSI_K"), stoch_d=f("StochRSI_D"),
        ma5=f("MA5"), ma20=f("MA20"), ma60=f("MA60"), ma120=f("MA120"),
        bb_upper=f("BB_upper"), bb_lower=f("BB_lower"), bb_width=f("BB_width"),
        atr=f("ATR"), vol=f("Volume"), vol_ma20=f("Vol_MA20"),
        high_52w=float(df["High"].squeeze().rolling(252).max().iloc[-1]),
        low_52w=float(df["Low"].squeeze().rolling(252).min().iloc[-1]),
    )

def build_prompt(ticker, ind, period_label, fib_info, div_info):
    vr = ind["vol"]/ind["vol_ma20"] if ind["vol_ma20"] else 1
    al = "정배열" if ind["ma5"]>ind["ma20"]>ind["ma60"] else "역배열" if ind["ma5"]<ind["ma20"]<ind["ma60"] else "혼조"
    div_text = "\n".join([f"- {d[0]}: {d[1]}" for d in div_info]) if div_info else "- 다이버전스 없음"
    return f"""당신은 엘리어트 파동 이론 전문가이자 기술적 분석가입니다.
주식 차트는 모든 시장 정보가 이미 반영되어 있다는 전제 하에, 아래 기술적 지표만으로 {ticker}를 분석해주세요.

=== 현재 지표 ({period_label}) ===
현재가: {ind['close']:,.0f} ({ind['change_pct']:+.2f}%)
52주 고가: {ind['high_52w']:,.0f} / 저가: {ind['low_52w']:,.0f}
이동평균: MA5={ind['ma5']:,.0f} MA20={ind['ma20']:,.0f} MA60={ind['ma60']:,.0f} MA120={ind['ma120']:,.0f} [{al}]
RSI(14): {ind['rsi']:.1f} | 스토캐스틱RSI K: {ind['stoch_k']:.1f} D: {ind['stoch_d']:.1f}
MACD Hist: {ind['macd_hist']:+.4f}
볼린저밴드: 상단={ind['bb_upper']:,.0f} 하단={ind['bb_lower']:,.0f} 밴드폭={ind['bb_width']:.3f}
ATR(14): {ind['atr']:,.0f}
거래량: 20일 평균 대비 {vr:.1f}배

=== 피보나치 레벨 ===
{fib_info}

=== 다이버전스 감지 ===
{div_text}

=== 엘리어트 파동 중심 분석 요청 ===
1. 🌊 현재 파동 위치 추정
   - 현재 어느 파동(1~5파 또는 A-B-C 조정파)에 있는지 추정
   - 근거 (이동평균, RSI, MACD, 거래량 패턴)

2. 📐 피보나치 분석
   - 현재가가 어느 피보나치 구간에 위치하는지
   - 다음 파동의 목표가 (피보나치 확장 기준)

3. ⚡ 모멘텀 분석
   - RSI, 스토캐스틱RSI, MACD 종합
   - 다이버전스 해석

4. 📦 거래량 분석
   - 파동별 거래량 패턴과 현재 거래량 비교

5. 🎯 매매 전략
   - 매수 / 매도 / 관망 중 하나 명시
   - 구체적 진입가, 손절가, 목표가 (ATR={ind['atr']:,.0f} 참고)
   - 파동 무효화 조건 (이 가격 이하로 내려가면 분석 틀림)

6. ⚠️ 리스크
   - 현재 분석이 틀릴 수 있는 시나리오
"""

# ── 세션 초기화
if "mode" not in st.session_state:
    st.session_state.mode = "home"
if "screen_result" not in st.session_state:
    st.session_state.screen_result = None

# ── 사이드바
with st.sidebar:
    st.markdown("## 📈 차트 분석기")
    st.markdown("---")
    ticker_input = st.text_input("종목 코드", value="005930.KS", help="삼성전자, HPSP, 애플 등 이름도 가능")
    period_map   = {"1개월":"1mo","3개월":"3mo","6개월":"6mo","1년":"1y","2년":"2y"}
    period_label = st.selectbox("분석 기간", list(period_map.keys()), index=2)
    period       = period_map[period_label]
    show_fib     = st.toggle("피보나치 레벨 표시", value=True)
    api_key      = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="console.groq.com 무료 발급")
    st.markdown("---")
    if st.button("🏠 홈으로", use_container_width=True):
        st.session_state.mode = "home"
        st.rerun()
    if st.button("🔍 종목 분석", use_container_width=True):
        st.session_state.mode = "analyze"
        st.rerun()
    if st.button("🏆 매수 Top5 추천", use_container_width=True):
        st.session_state.mode = "screen"
        st.session_state.screen_result = None
        st.rerun()
    st.markdown("<div style='font-size:11px;color:#2a3040;margin-top:12px;line-height:1.8;'>삼성전자 005930.KS<br>SK하이닉스 000660.KS<br>HPSP 403870.KQ<br>애플 AAPL / 엔비디아 NVDA</div>", unsafe_allow_html=True)

# ══════════════════════════════════════
# 홈
# ══════════════════════════════════════
if st.session_state.mode == "home":

    # ── 데이터 수집 함수들
    @st.cache_data(ttl=300)
    def get_all_home_data():
        # 1. 글로벌 주요 지수 + 안전자산
        index_tickers = {
            "코스피":"^KS11","코스닥":"^KQ11","나스닥":"^IXIC",
            "S&P500":"^GSPC","달러/원":"KRW=X","VIX":"^VIX",
            "금":"GC=F","원유":"CL=F",
        }
        # 2. 글로벌 투자자금 대용 (대형 ETF 거래대금 추세)
        fund_tickers = {
            "SPY(미국전체)":"SPY","QQQ(나스닥100)":"QQQ",
            "EEM(신흥국)":"EEM","IEF(미국채10Y)":"IEF",
        }
        # 3. 미국 섹터 ETF 11개
        sector_tickers = {
            "기술(XLK)":"XLK","헬스케어(XLV)":"XLV","금융(XLF)":"XLF",
            "에너지(XLE)":"XLE","산업재(XLI)":"XLI","소비재(XLY)":"XLY",
            "필수소비재(XLP)":"XLP","유틸리티(XLU)":"XLU","부동산(XLRE)":"XLRE",
            "소재(XLB)":"XLB","통신(XLC)":"XLC",
        }
        # 섹터별 대표 종목 (선두 + 다크호스)
        sector_stocks = {
            "XLK": {"leader":["NVDA","MSFT","AAPL"], "dark":["PLTR","ARM","SMCI"]},
            "XLV": {"leader":["LLY","UNH","JNJ"],    "dark":["RXRX","TVTX","NUVL"]},
            "XLF": {"leader":["BRK-B","JPM","V"],    "dark":["HOOD","SOFI","AFRM"]},
            "XLE": {"leader":["XOM","CVX","SLB"],    "dark":["SM","CIVI","MGY"]},
            "XLI": {"leader":["GE","CAT","HON"],     "dark":["KTOS","HII","DRS"]},
            "XLY": {"leader":["AMZN","TSLA","MCD"],  "dark":["RIVN","LCID","NKLA"]},
            "XLP": {"leader":["WMT","PG","KO"],      "dark":["COTY","SFM","GO"]},
            "XLU": {"leader":["NEE","DUK","SO"],     "dark":["VST","NRG","AES"]},
            "XLRE":{"leader":["PLD","AMT","EQIX"],   "dark":["IIPR","COLD","REXR"]},
            "XLB": {"leader":["LIN","APD","SHW"],    "dark":["MP","ALTM","CTRA"]},
            "XLC": {"leader":["META","GOOGL","DIS"],  "dark":["RDDT","SNAP","PINS"]},
        }

        def fetch(ticker, period="1mo"):
            try:
                df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                if df is None or df.empty: return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df.dropna(how="all")
            except: return None

        # 지수 데이터
        indices = {}
        for name, t in index_tickers.items():
            df = fetch(t, "5d")
            if df is None: continue
            close = df["Close"].squeeze().dropna()
            if len(close) < 2: continue
            cur  = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            indices[name] = {"cur":cur, "chg":(cur/prev-1)*100, "ticker":t}

        # 글로벌 자금 흐름 (ETF 거래대금 = 가격×거래량 추세)
        fund_flow = {}
        for name, t in fund_tickers.items():
            df = fetch(t, "1mo")
            if df is None: continue
            close = df["Close"].squeeze().dropna()
            vol   = df["Volume"].squeeze().dropna()
            # 거래대금 (억달러)
            turnover = (close * vol / 1e8).dropna()
            if len(turnover) < 10: continue
            recent_avg = float(turnover.iloc[-5:].mean())
            prev_avg   = float(turnover.iloc[-15:-5].mean())
            chg_pct    = (recent_avg/prev_avg-1)*100 if prev_avg else 0
            price_chg  = float((close.iloc[-1]/close.iloc[0]-1)*100)
            fund_flow[name] = {
                "recent_vol": recent_avg,
                "vol_chg": chg_pct,
                "price_chg": price_chg,
            }

        # 섹터 수익률 + 거래대금 (1주, 1개월)
        sector_data = {}
        for name, t in sector_tickers.items():
            df1w = fetch(t, "5d")
            df1m = fetch(t, "1mo")
            if df1w is None or df1m is None: continue
            c1w  = df1w["Close"].squeeze().dropna()
            c1m  = df1m["Close"].squeeze().dropna()
            v1w  = df1w["Volume"].squeeze().dropna()
            if len(c1w)<2 or len(c1m)<2: continue
            ret1w = float((c1w.iloc[-1]/c1w.iloc[0]-1)*100)
            ret1m = float((c1m.iloc[-1]/c1m.iloc[0]-1)*100)
            # 거래대금 증가율
            vol_recent = float((df1w["Close"].squeeze() * df1w["Volume"].squeeze()).iloc[-3:].mean())
            sector_data[name] = {
                "ticker": t,
                "ret1w": ret1w,
                "ret1m": ret1m,
                "stocks": sector_stocks.get(t, {}),
            }

        # 섹터 종목 수익률 (다크호스 확인용)
        stock_perf = {}
        all_stocks = set()
        for v in sector_stocks.values():
            all_stocks.update(v.get("leader",[]))
            all_stocks.update(v.get("dark",[]))
        for s in list(all_stocks)[:30]:  # 너무 많으면 느리니 30개 제한
            df = fetch(s, "1mo")
            if df is None: continue
            close = df["Close"].squeeze().dropna()
            if len(close) < 2: continue
            stock_perf[s] = float((close.iloc[-1]/close.iloc[0]-1)*100)

        return indices, fund_flow, sector_data, stock_perf

    with st.spinner("🌐 글로벌 시장 데이터 수집 중... (최초 30초, 이후 5분 캐시)"):
        indices, fund_flow, sector_data, stock_perf = get_all_home_data()

    # ══ 섹션 1: 글로벌 지수 현황 ══
    st.markdown("## 🌍 글로벌 시장 현황")
    st.caption("yfinance 기준 · 5분 자동 갱신")

    icons = {"코스피":"🇰🇷","코스닥":"📊","나스닥":"🇺🇸","S&P500":"🗽","달러/원":"💵","VIX":"😨","금":"🥇","원유":"🛢️"}
    if indices:
        cols = st.columns(len(indices))
        for col, (name, data) in zip(cols, indices.items()):
            chg   = data["chg"]
            color = "#4ade80" if chg >= 0 else "#f87171"
            sign  = "▲" if chg >= 0 else "▼"
            col.markdown(f"""
<div style='background:#151820;border:1px solid #1e2130;border-radius:10px;padding:14px 10px;text-align:center;'>
  <div style='font-size:18px;'>{icons.get(name,"📈")}</div>
  <div style='font-size:10px;color:#4a5060;margin:4px 0;'>{name}</div>
  <div style='font-size:16px;font-weight:700;color:#f0f2f8;font-family:DM Mono,monospace;'>{data["cur"]:,.1f}</div>
  <div style='font-size:12px;color:{color};font-weight:600;'>{sign}{abs(chg):.2f}%</div>
</div>""", unsafe_allow_html=True)

    # VIX 기반 공포탐욕 + 시장 해석
    if indices:
        vix = indices.get("VIX",{}).get("cur", 20)
        usd_chg = indices.get("달러/원",{}).get("chg", 0)
        nasdaq_chg = indices.get("나스닥",{}).get("chg", 0)
        gold_chg = indices.get("금",{}).get("chg", 0)

        # 공포탐욕 점수 (0~100)
        fg_score = max(0, min(100, int(100 - (vix - 10) * 2.5)))
        if fg_score >= 75:   fg_label, fg_color = "극도의 탐욕 😍", "#4ade80"
        elif fg_score >= 55: fg_label, fg_color = "탐욕 😊", "#86efac"
        elif fg_score >= 45: fg_label, fg_color = "중립 😐", "#fbbf24"
        elif fg_score >= 25: fg_label, fg_color = "공포 😨", "#fb923c"
        else:                fg_label, fg_color = "극도의 공포 😱", "#f87171"

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
<div style='background:#151820;border:1px solid #1e2130;border-radius:12px;padding:20px;text-align:center;'>
  <div style='font-size:12px;color:#4a5060;margin-bottom:8px;letter-spacing:0.1em;'>공포 & 탐욕 지수</div>
  <div style='font-size:48px;font-weight:800;color:{fg_color};font-family:DM Mono,monospace;'>{fg_score}</div>
  <div style='font-size:14px;color:{fg_color};margin-top:4px;font-weight:600;'>{fg_label}</div>
  <div style='font-size:10px;color:#3a4050;margin-top:8px;'>VIX {vix:.1f} 기반</div>
</div>""", unsafe_allow_html=True)
        with col2:
            comments = []
            if gold_chg > 0.5 and usd_chg > 0.3:
                comments.append("🔴 **안전자산 동반 강세** — 위험 회피 심리, 주식 비중 축소 고려")
            elif gold_chg < -0.3 and nasdaq_chg > 0.5:
                comments.append("🟢 **위험자산 선호** — 금 약세 + 나스닥 강세, 성장주 유리")
            if usd_chg > 0.5:
                comments.append("💵 **원화 약세** → 외국인 매도 압력, 수출 대형주 주목")
            elif usd_chg < -0.5:
                comments.append("💵 **원화 강세** → 외국인 유입 기대, 내수·IT 주목")
            if nasdaq_chg > 1:
                comments.append("🇺🇸 **나스닥 강세** → 다음 거래일 코스닥 상승 기대")
            elif nasdaq_chg < -1:
                comments.append("🇺🇸 **나스닥 약세** → 다음 거래일 코스닥 하락 주의")
            if not comments:
                comments.append("😐 특별한 시그널 없음 — 관망 유지")
            st.markdown("**💬 시장 해석**")
            for c in comments:
                st.markdown(f"• {c}")

    st.markdown("---")

    # ══ 섹션 2: 글로벌 투자자금 흐름 ══
    st.markdown("## 💰 글로벌 투자자금 흐름")
    st.caption("주요 ETF 거래대금 추세로 자금 증감 추정 (최근 5일 vs 이전 10일 평균)")

    if fund_flow:
        cols = st.columns(len(fund_flow))
        for col, (name, data) in zip(cols, fund_flow.items()):
            vol_chg   = data["vol_chg"]
            price_chg = data["price_chg"]
            v_color = "#4ade80" if vol_chg > 0 else "#f87171"
            p_color = "#4ade80" if price_chg > 0 else "#f87171"
            v_sign  = "▲" if vol_chg > 0 else "▼"
            p_sign  = "▲" if price_chg > 0 else "▼"
            # 자금 상태 판단
            if vol_chg > 5 and price_chg > 0:
                status, s_color = "자금 유입 ↑", "#4ade80"
            elif vol_chg < -5 and price_chg < 0:
                status, s_color = "자금 유출 ↓", "#f87171"
            elif vol_chg > 5 and price_chg < 0:
                status, s_color = "매도 급증 ⚠", "#fb923c"
            else:
                status, s_color = "보합", "#94a3b8"
            col.markdown(f"""
<div style='background:#151820;border:1px solid #1e2130;border-radius:10px;padding:14px 12px;'>
  <div style='font-size:11px;color:#4a5060;margin-bottom:6px;'>{name}</div>
  <div style='font-size:13px;font-weight:700;color:{s_color};margin-bottom:8px;'>{status}</div>
  <div style='font-size:11px;color:#5a6070;'>거래대금 {v_sign}<span style='color:{v_color};'>{abs(vol_chg):.1f}%</span></div>
  <div style='font-size:11px;color:#5a6070;'>수익률 {p_sign}<span style='color:{p_color};'>{abs(price_chg):.1f}%</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ══ 섹션 3: 돈이 몰리는 섹터 Top5 ══
    st.markdown("## 🏭 자금이 몰리는 섹터 Top 5")
    st.caption("미국 섹터 ETF 1주 수익률 기준 · 순위 변동 표시")

    if sector_data:
        # 1주 수익률 기준 정렬
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]["ret1w"], reverse=True)
        top5 = sorted_sectors[:5]
        worst1 = sorted_sectors[-1]
        max_ret = max(abs(v["ret1w"]) for _, v in sorted_sectors) or 1

        for rank, (name, data) in enumerate(top5, 1):
            ret1w = data["ret1w"]
            ret1m = data["ret1m"]
            bar_w = int(abs(ret1w) / max_ret * 100)
            bar_color = "#4ade80" if ret1w > 0 else "#f87171"
            # 1주 vs 1개월 순위 비교로 모멘텀 파악
            rank1m_list = sorted(sector_data.items(), key=lambda x: x[1]["ret1m"], reverse=True)
            rank1m = next((i+1 for i, (n,_) in enumerate(rank1m_list) if n==name), 0)
            rank_diff = rank1m - rank  # 양수 = 최근 상승, 음수 = 최근 하락
            if rank_diff > 2:   momentum, m_color = f"🚀 급상승 +{rank_diff}계단", "#4ade80"
            elif rank_diff > 0: momentum, m_color = f"↗ 상승 +{rank_diff}계단", "#86efac"
            elif rank_diff == 0: momentum, m_color = "→ 유지", "#94a3b8"
            elif rank_diff > -3: momentum, m_color = f"↘ 하락 {rank_diff}계단", "#fb923c"
            else:               momentum, m_color = f"📉 급락 {rank_diff}계단", "#f87171"

            rank_emoji = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank-1]

            # 섹터 내 종목
            stocks_info = data.get("stocks", {})
            leaders  = stocks_info.get("leader", [])
            darks    = stocks_info.get("dark", [])

            def stock_badge(ticker, is_dark=False):
                perf = stock_perf.get(ticker, 0)
                color = "#4ade80" if perf > 0 else "#f87171"
                sign  = "▲" if perf > 0 else "▼"
                border = "#fbbf24" if is_dark else "#1e2130"
                return f"<span style='background:#151820;border:1px solid {border};border-radius:6px;padding:3px 8px;font-size:11px;color:{color};margin-right:4px;'>{ticker} {sign}{abs(perf):.1f}%</span>"

            leader_badges = "".join([stock_badge(s) for s in leaders])
            dark_badges   = "".join([stock_badge(s, True) for s in darks])

            st.markdown(f"""
<div style='background:#151820;border:1px solid #1e2130;border-radius:12px;padding:16px 20px;margin-bottom:10px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
    <div>
      <span style='font-size:16px;'>{rank_emoji}</span>
      <span style='font-size:15px;font-weight:700;color:#f0f2f8;margin-left:8px;'>{name}</span>
      <span style='font-size:12px;color:{m_color};margin-left:12px;'>{momentum}</span>
    </div>
    <div style='text-align:right;'>
      <span style='font-size:14px;font-weight:700;color:{bar_color};font-family:DM Mono,monospace;'>{"▲" if ret1w>0 else "▼"}{abs(ret1w):.2f}%</span>
      <span style='font-size:11px;color:#4a5060;margin-left:8px;'>1주 / 1개월 {"▲" if ret1m>0 else "▼"}{abs(ret1m):.1f}%</span>
    </div>
  </div>
  <div style='background:#1a1e2a;border-radius:4px;height:6px;margin-bottom:12px;'>
    <div style='background:{bar_color};height:6px;border-radius:4px;width:{bar_w}%;'></div>
  </div>
  <div style='margin-bottom:6px;'>
    <span style='font-size:10px;color:#4a5060;margin-right:8px;'>👑 선두</span>{leader_badges}
  </div>
  <div>
    <span style='font-size:10px;color:#fbbf24;margin-right:8px;'>⚡ 다크호스</span>{dark_badges}
  </div>
</div>""", unsafe_allow_html=True)

        # 꼴찌 섹터 (자금 이탈)
        name_w, data_w = worst1
        ret_w = data_w["ret1w"]
        st.markdown(f"<div style='background:#1a0d0d;border:1px solid #4a1a1a;border-radius:10px;padding:12px 16px;font-size:12px;color:#f87171;'>📉 자금 이탈 섹터: <b>{name_w}</b> — {ret_w:.2f}% (1주)</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style='text-align:center;color:#2a3040;font-size:13px;padding:12px 0;'>
엘리어트 파동 중심 기술적 분석 플랫폼 · 사이드바에서 <b style='color:#4ade80;'>종목 분석</b> 또는 <b style='color:#4ade80;'>Top5 추천</b> 선택
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════
# 개별 종목 분석
# ══════════════════════════════════════
elif st.session_state.mode == "analyze":
    ticker = search_ticker(ticker_input)
    with st.spinner(f"{ticker} 데이터 수집 중..."):
        df = get_stock_data(ticker, period)
    if df is None or len(df) < 30:
        st.error(f"'{ticker}' 데이터를 불러올 수 없습니다.")
    else:
        df  = calc_indicators(df)
        ind = summarize(df)
        divs = detect_divergence(df)

        # 피보나치 계산
        try:
            ret, ext, sh, sl, h_idx, l_idx = calc_fibonacci(df)
            fib_info = f"스윙 고점: {sh:,.0f} / 스윙 저점: {sl:,.0f}\n"
            fib_info += "되돌림: " + " | ".join([f"{k}={v:,.0f}" for k,v in list(ret.items())[1:5]]) + "\n"
            fib_info += "확장 목표: " + " | ".join([f"{k}={v:,.0f}" for k,v in ext.items()])
        except:
            fib_info = "피보나치 계산 불가"

        # 지표 카드
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        vr = ind["vol"]/ind["vol_ma20"] if ind["vol_ma20"] else 1
        al = "정배열" if ind["ma5"]>ind["ma20"]>ind["ma60"] else "역배열" if ind["ma5"]<ind["ma20"]<ind["ma60"] else "혼조"
        with c1: st.metric("현재가", f"{ind['close']:,.0f}", f"{ind['change_pct']:+.2f}%")
        with c2: st.metric("RSI", f"{ind['rsi']:.1f}", "과매수>70 / 과매도<30")
        with c3: st.metric("StochRSI K", f"{ind['stoch_k']:.1f}", "과매수>80 / 과매도<20")
        with c4: st.metric("MACD Hist", f"{ind['macd_hist']:+.4f}")
        with c5: st.metric("ATR", f"{ind['atr']:,.0f}", "변동성 기준")
        with c6: st.metric("MA 배열", al)

        # 다이버전스 배지
        if divs:
            for d_type, d_msg in divs:
                color = "#fbbf24" if "약세" in d_type else "#4ade80"
                st.markdown(f"<div style='background:#1a1a0d;border:1px solid {color};border-radius:8px;padding:8px 16px;margin:4px 0;color:{color};font-size:13px;'>🔔 <b>{d_type}</b> — {d_msg}</div>", unsafe_allow_html=True)

        # 피보나치 요약
        try:
            st.markdown(f"<div style='background:#151820;border:1px solid #1e2130;border-radius:8px;padding:10px 16px;font-size:12px;color:#94a3b8;margin:4px 0;'>📐 <b>피보나치</b> — 스윙 고점 <b style='color:#f87171;'>{sh:,.0f}</b> / 저점 <b style='color:#4ade80;'>{sl:,.0f}</b> | 61.8% 지지 <b style='color:#fbbf24;'>{ret['61.8%']:,.0f}</b> | 161.8% 목표 <b style='color:#60a5fa;'>{ext['161.8%']:,.0f}</b></div>", unsafe_allow_html=True)
        except:
            pass

        # 차트
        fig = build_chart(df, ticker, show_fib=show_fib)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # AI 분석
        st.markdown("---")
        st.markdown("#### 🌊 엘리어트 파동 AI 분석 리포트")
        if not api_key:
            st.info("사이드바에 Groq API Key를 입력하면 AI 분석이 활성화됩니다.")
        else:
            prompt = build_prompt(ticker, ind, period_label, fib_info, divs)
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
            show = df[["Open","High","Low","Close","Volume","RSI","StochRSI_K","MACD","ATR"]].tail(20).copy()
            show.index = show.index.strftime("%Y-%m-%d")
            st.dataframe(show, use_container_width=True)

# ══════════════════════════════════════
# 스크리닝 Top5
# ══════════════════════════════════════
elif st.session_state.mode == "screen":
    st.markdown("## 🏆 엘리어트 파동 기준 매수 Top 5")
    st.caption("RSI 과매도 · StochRSI · MACD 골든크로스 · 피보나치 지지 · 강세 다이버전스 종합 점수")

    if st.session_state.screen_result is None:
        total    = len(KR_UNIVERSE)
        prog_bar = st.progress(0)
        status   = st.empty()
        results  = []
        for i, (name, ticker) in enumerate(KR_UNIVERSE.items()):
            prog_bar.progress((i+1)/total)
            status.markdown(f"🔍 **{name}** ({ticker}) 분석 중... `{i+1} / {total}`")
            r = score_stock(name, ticker)
            if r and r["score"] > 0:
                results.append(r)
        prog_bar.progress(1.0)
        status.markdown(f"✅ 완료! {len(results)}개 종목 분석 → Top 5 선정")
        results.sort(key=lambda x: x["score"], reverse=True)
        st.session_state.screen_result = results[:5]
        st.rerun()
    else:
        if st.button("🔄 다시 스크리닝"):
            st.session_state.screen_result = None
            st.rerun()
        rank_emoji = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        for i, r in enumerate(st.session_state.screen_result):
            st.markdown(f"### {rank_emoji[i]} {r['name']} ({r['ticker']}) — {r['score']}점")
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: st.metric("현재가", f"{r['cur']:,.0f}")
            with c2: st.metric("RSI", f"{r['rsi']:.1f}")
            with c3: st.metric("ATR", f"{r['atr']:,.0f}")
            with c4: st.metric("매수 목표가", f"{r['target_buy']:,.0f}")
            with c5: st.metric("목표 매도가", f"{r['target_sell']:,.0f}", delta=f"ATR×3")
            for sig in r["signals"]:
                st.markdown(f"• {sig}")
            st.plotly_chart(build_screening_chart(r), use_container_width=True, config={"displayModeBar": False})
            if i < 4:
                st.markdown("---")
