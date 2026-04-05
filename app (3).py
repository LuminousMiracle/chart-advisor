
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

KR_NAMES = {v: k for k, v in KR_UNIVERSE.items()}

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
        if raw is None or raw.empty: return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        return raw.dropna(how="all")
    except:
        return None

def calc_indicators(df):
    close = df["Close"].squeeze()
    df["MA5"]  = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
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
    ema12 = close.ewm(span=12,adjust=False).mean()
    ema26 = close.ewm(span=26,adjust=False).mean()
    df["MACD"]        = ema12-ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_hist"]   = df["MACD"]-df["MACD_signal"]
    df["Vol_MA20"]    = df["Volume"].squeeze().rolling(20).mean()
    return df

def build_chart(df, ticker):
    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.55,0.25,0.20],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"].squeeze(),high=df["High"].squeeze(),low=df["Low"].squeeze(),close=df["Close"].squeeze(),name="주가",increasing_fillcolor="#4ade80",increasing_line_color="#4ade80",decreasing_fillcolor="#f87171",decreasing_line_color="#f87171"),row=1,col=1)
    for ma,color in [("MA5","#60a5fa"),("MA20","#fbbf24"),("MA60","#a78bfa")]:
        fig.add_trace(go.Scatter(x=df.index,y=df[ma].squeeze(),name=ma,line=dict(color=color,width=1.3)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_upper"].squeeze(),line=dict(color="#94a3b8",width=0.8,dash="dot"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_lower"].squeeze(),line=dict(color="#94a3b8",width=0.8,dash="dot"),fill="tonexty",fillcolor="rgba(148,163,184,0.05)",showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["RSI"].squeeze(),name="RSI",line=dict(color="#c084fc",width=1.5)),row=2,col=1)
    for lvl,clr in [(70,"#f87171"),(30,"#4ade80"),(50,"#475569")]:
        fig.add_hline(y=lvl,line_dash="dot",line_color=clr,line_width=0.8,row=2,col=1)
    hist_vals = df["MACD_hist"].squeeze().fillna(0).tolist()
    bar_colors = ["#4ade80" if v>=0 else "#f87171" for v in hist_vals]
    fig.add_trace(go.Bar(x=df.index,y=df["MACD_hist"].squeeze(),name="MACD Hist",marker_color=bar_colors,opacity=0.8),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["MACD"].squeeze(),name="MACD",line=dict(color="#60a5fa",width=1.2)),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["MACD_signal"].squeeze(),name="Signal",line=dict(color="#fbbf24",width=1.2)),row=3,col=1)
    fig.update_layout(paper_bgcolor="#0d0f14",plot_bgcolor="#0d0f14",font=dict(color="#5a6070",size=11),xaxis_rangeslider_visible=False,height=620,margin=dict(l=0,r=0,t=30,b=0),legend=dict(bgcolor="#111318",bordercolor="#1e2130",borderwidth=1,orientation="h",x=0,y=1.02))
    for row in [1,2,3]:
        fig.update_xaxes(gridcolor="#1a1e2a",row=row,col=1)
        fig.update_yaxes(gridcolor="#1a1e2a",row=row,col=1)
    return fig

def build_screening_chart(r):
    df = r["df"]
    close = df["Close"].squeeze()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"].squeeze(),high=df["High"].squeeze(),low=df["Low"].squeeze(),close=close,name=r["name"],increasing_fillcolor="#4ade80",increasing_line_color="#4ade80",decreasing_fillcolor="#f87171",decreasing_line_color="#f87171"))
    for span,color,label in [(20,"#fbbf24","MA20"),(60,"#a78bfa","MA60")]:
        fig.add_trace(go.Scatter(x=df.index,y=close.rolling(span).mean(),name=label,line=dict(color=color,width=1.2)))
    fig.add_trace(go.Scatter(x=df.index,y=bb_mid+2*bb_std,line=dict(color="#94a3b8",width=0.8,dash="dot"),showlegend=False))
    fig.add_trace(go.Scatter(x=df.index,y=bb_mid-2*bb_std,line=dict(color="#94a3b8",width=0.8,dash="dot"),fill="tonexty",fillcolor="rgba(148,163,184,0.05)",showlegend=False))
    fig.add_hline(y=r["target_buy"],line_color="#4ade80",line_width=2,line_dash="dash",annotation_text=f"  매수 목표가 {r['target_buy']:,.0f}",annotation_position="right",annotation_font_color="#4ade80")
    fig.add_hline(y=r["target_sell"],line_color="#f87171",line_width=2,line_dash="dash",annotation_text=f"  목표 매도가 {r['target_sell']:,.0f}",annotation_position="right",annotation_font_color="#f87171")
    fig.update_layout(paper_bgcolor="#0d0f14",plot_bgcolor="#0d0f14",font=dict(color="#5a6070",size=11),xaxis_rangeslider_visible=False,height=400,margin=dict(l=0,r=130,t=10,b=0),legend=dict(bgcolor="#111318",bordercolor="#1e2130",borderwidth=1,orientation="h",x=0,y=1.08))
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
        close = df["Close"].squeeze()
        score = 0
        signals = []
        ma5  = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        cur  = float(close.iloc[-1])
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100-100/(1+rs)).iloc[-1])
        ema12 = close.ewm(span=12,adjust=False).mean()
        ema26 = close.ewm(span=26,adjust=False).mean()
        macd  = ema12-ema26
        sig   = macd.ewm(span=9,adjust=False).mean()
        hist      = float((macd-sig).iloc[-1])
        hist_prev = float((macd-sig).iloc[-2])
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_lower = float((bb_mid-2*bb_std).iloc[-1])
        bb_upper = float((bb_mid+2*bb_std).iloc[-1])
        bb_width = (bb_upper-bb_lower)/float(bb_mid.iloc[-1])
        vol      = float(df["Volume"].squeeze().iloc[-1])
        vol_ma20 = float(df["Volume"].squeeze().rolling(20).mean().iloc[-1])
        vol_ratio = vol/vol_ma20 if vol_ma20 else 1
        if rsi < 35:
            score += 30; signals.append(f"RSI 과매도({rsi:.0f}) — 2파/4파 완료 가능")
        elif rsi < 45:
            score += 15; signals.append(f"RSI 저점권({rsi:.0f}) — 반등 준비")
        if hist > 0 and hist_prev <= 0:
            score += 25; signals.append("MACD 골든크로스 — 3파/5파 시작 신호")
        elif hist > 0 and hist > hist_prev:
            score += 10; signals.append("MACD 히스토그램 상승 중")
        if cur <= bb_lower*1.02:
            score += 20; signals.append("볼린저 하단 근접 — 과매도 반등 구간")
        elif cur <= bb_lower*1.05:
            score += 10
        if ma5 > ma20 and ma20 < ma60:
            score += 15; signals.append("단기 반등 + 중기 눌림 — 3파 준비")
        elif ma5 > ma20 > ma60:
            score += 10; signals.append("정배열 — 상승 추세 유지")
        if vol_ratio > 2.0:
            score += 10; signals.append(f"거래량 급증({vol_ratio:.1f}x) — 세력 유입 가능")
        if bb_width < 0.08:
            score += 10; signals.append("밴드 수축 — 변동성 폭발 임박")
        target_buy  = round(max(bb_lower, cur*0.97), 0)
        target_sell = round(cur*1.1, 0)
        return dict(ticker=ticker,name=name,score=score,signals=signals,cur=cur,rsi=rsi,macd_hist=hist,bb_lower=bb_lower,bb_upper=bb_upper,target_buy=target_buy,target_sell=target_sell,df=df)
    except:
        return None

def summarize(df):
    l, p = df.iloc[-1], df.iloc[-2]
    def f(col): return float(np.squeeze(l[col]))
    close = f("Close")
    return dict(close=close,change_pct=(close/float(np.squeeze(p["Close"]))-1)*100,rsi=f("RSI"),macd=f("MACD"),macd_sig=f("MACD_signal"),macd_hist=f("MACD_hist"),ma5=f("MA5"),ma20=f("MA20"),ma60=f("MA60"),bb_upper=f("BB_upper"),bb_lower=f("BB_lower"),bb_width=f("BB_width"),vol=f("Volume"),vol_ma20=f("Vol_MA20"),high_52w=float(df["High"].squeeze().rolling(252).max().iloc[-1]),low_52w=float(df["Low"].squeeze().rolling(252).min().iloc[-1]))

def build_prompt(ticker, ind, period_label):
    vr = ind["vol"]/ind["vol_ma20"] if ind["vol_ma20"] else 1
    al = "정배열" if ind["ma5"]>ind["ma20"]>ind["ma60"] else "역배열" if ind["ma5"]<ind["ma20"]<ind["ma60"] else "혼조"
    return f"""당신은 주식 기술적 분석 전문가입니다. {ticker} 종목을 분석하여 매수/매도/관망 의견을 한국어로 제시해주세요.

=== 최신 지표 ({period_label}) ===
현재가: {ind["close"]:,.0f} ({ind["change_pct"]:+.2f}%)
52주 고가: {ind["high_52w"]:,.0f} / 저가: {ind["low_52w"]:,.0f}
이동평균: MA5={ind["ma5"]:,.0f} MA20={ind["ma20"]:,.0f} MA60={ind["ma60"]:,.0f} [{al}]
RSI(14): {ind["rsi"]:.1f}
MACD Hist: {ind["macd_hist"]:+.4f}
볼린저밴드: 상단={ind["bb_upper"]:,.0f} 하단={ind["bb_lower"]:,.0f} 밴드폭={ind["bb_width"]:.3f}
거래량: 20일 평균 대비 {vr:.1f}배

=== 분석 항목 ===
1. 추세 분석 (이동평균 배열)
2. 모멘텀 분석 (RSI, MACD)
3. 볼린저밴드 분석
4. 거래량 분석
5. 엘리어트 파동 위치 추정
6. 최종 의견 (매수/매도/관망 명시 + 지지/저항선 + 리스크)"""

# ── 세션 초기화
if "mode" not in st.session_state:
    st.session_state.mode = "home"
if "screen_result" not in st.session_state:
    st.session_state.screen_result = None

# ── 사이드바
with st.sidebar:
    st.markdown("## 📈 차트 분석기")
    st.markdown("---")
    ticker_input = st.text_input("종목 코드", value="005930.KS", help="삼성전자, HPSP, 애플, NVDA 등 이름도 가능")
    period_map   = {"1개월":"1mo","3개월":"3mo","6개월":"6mo","1년":"1y","2년":"2y"}
    period_label = st.selectbox("분석 기간", list(period_map.keys()), index=2)
    period       = period_map[period_label]
    api_key      = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="console.groq.com 무료 발급")
    st.markdown("---")

    if st.button("🔍 분석 시작", use_container_width=True):
        st.session_state.mode = "analyze"
        st.rerun()

    if st.button("🏆 매수 Top5 추천", use_container_width=True):
        st.session_state.mode = "screen"
        st.rerun()

    st.markdown("<div style='font-size:11px;color:#2a3040;margin-top:12px;line-height:1.8;'>삼성전자 005930.KS<br>SK하이닉스 000660.KS<br>HPSP 403870.KQ<br>애플 AAPL / 엔비디아 NVDA</div>", unsafe_allow_html=True)

# ── 홈
if st.session_state.mode == "home":
    st.markdown("<div style='text-align:center;padding:100px 0;color:#2a3040;'><div style='font-size:52px;'>📈</div><div style='font-size:18px;color:#3a4050;margin-top:16px;'>종목 분석 또는 Top5 추천을 선택하세요</div></div>", unsafe_allow_html=True)

# ── 개별 종목 분석
elif st.session_state.mode == "analyze":
    ticker = search_ticker(ticker_input)
    with st.spinner(f"{ticker} 데이터 수집 중..."):
        df = get_stock_data(ticker, period)
    if df is None or len(df) < 30:
        st.error(f"'{ticker}' 데이터를 불러올 수 없습니다.")
    else:
        df  = calc_indicators(df)
        ind = summarize(df)
        c1,c2,c3,c4,c5 = st.columns(5)
        vr = ind["vol"]/ind["vol_ma20"] if ind["vol_ma20"] else 1
        al = "정배열" if ind["ma5"]>ind["ma20"]>ind["ma60"] else "역배열" if ind["ma5"]<ind["ma20"]<ind["ma60"] else "혼조"
        with c1: st.metric("현재가",f"{ind['close']:,.0f}",f"{ind['change_pct']:+.2f}%")
        with c2: st.metric("RSI",f"{ind['rsi']:.1f}","과매수>70/과매도<30")
        with c3: st.metric("MACD Hist",f"{ind['macd_hist']:+.4f}")
        with c4: st.metric("거래량",f"{vr:.1f}x","20일 평균 대비")
        with c5: st.metric("MA 배열",al)
        st.plotly_chart(build_chart(df, ticker), use_container_width=True, config={"displayModeBar":False})
        st.markdown("---")
        st.markdown("#### 🤖 AI 기술적 분석 리포트")
        if not api_key:
            st.info("사이드바에 Groq API Key를 입력하면 AI 분석이 활성화됩니다.")
        else:
            prompt = build_prompt(ticker, ind, period_label)
            box = st.empty(); full = ""
            try:
                client = Groq(api_key=api_key)
                stream = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"user","content":prompt}],stream=True)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full += delta
                    box.markdown(f'<div class="analysis-box">{full}▌</div>',unsafe_allow_html=True)
                box.markdown(f'<div class="analysis-box">{full}</div>',unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI 분석 오류: {e}")
        with st.expander("📋 최근 20일 데이터"):
            show = df[["Open","High","Low","Close","Volume","RSI","MACD"]].tail(20).copy()
            show.index = show.index.strftime("%Y-%m-%d")
            st.dataframe(show, use_container_width=True)

# ── 스크리닝 Top5
elif st.session_state.mode == "screen":
    st.markdown("## 🏆 엘리어트 파동 기준 매수 Top 5")
    st.caption("RSI 과매도 반등 · MACD 골든크로스 · 볼린저 하단 근접 · 거래량 급증 종합 점수")

    if st.session_state.screen_result is None:
        total    = len(KR_UNIVERSE)
        prog_bar = st.progress(0)
        status   = st.empty()
        results  = []

        for i, (name, ticker) in enumerate(KR_UNIVERSE.items()):
            pct = (i+1)/total
            prog_bar.progress(pct)
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
        if st.button("🔄 다시 스크리닝", use_container_width=False):
            st.session_state.screen_result = None
            st.rerun()

        rank_emoji = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        for i, r in enumerate(st.session_state.screen_result):
            st.markdown(f"### {rank_emoji[i]} {r['name']} ({r['ticker']})")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("현재가", f"{r['cur']:,.0f}")
            with c2: st.metric("RSI", f"{r['rsi']:.1f}")
            with c3: st.metric("매수 목표가", f"{r['target_buy']:,.0f}", delta="진입가")
            with c4: st.metric("목표 매도가", f"{r['target_sell']:,.0f}", delta="+10%")
            for sig in r["signals"]:
                st.markdown(f"• {sig}")
            st.plotly_chart(build_screening_chart(r), use_container_width=True, config={"displayModeBar":False})
            if i < 4:
                st.markdown("---")
