"""
MarketWhisper AI Agent App (Prompt Engineering CS4680 Final Project)
=================================================

This is the *full* project version of MarketWhisper. It is designed to:

- Fetch **live market data** (via `yfinance`)
- Pull **fresh external context** (via Reddit search + optional NewsAPI)
- Feed that external context into the LLM so it can reason about **Q3 earnings,
  current news, social buzz**, etc. despite its training cutoff
- Act as a multi-domain **AI Agent**:
    * Finance Agent (plans + executes finance-related tasks)
    * File System Agent (plans + executes file operations safely)
    * General Chat Agent (optionally finance-aware, using live context)
- Provide a full **GUI** (Streamlit) + logging and safety checks

Core Project Requirements Mapping
---------------------------------
LLM Integration Module
- Uses OpenAI Python SDK (client.chat.completions.create)
- Centralized helper with retry + simple backoff for rate limits
- JSON and text response helpers

Action Interpreter / Executor
- Finance Agent:
    * Plans actions like "fetch_stock", "compare_stocks", "build_portfolio"
      from natural language using structured JSON output from the LLM.
    * Executes actions using Python + yfinance and feeds results back.
- File System Agent:
    * Summarizes a directory
    * Plans file ops ("mkdir", "move", "delete_soft") via LLM
    * Validates + safely executes (no path traversal; soft-delete only)

User Interface
- Streamlit app with multiple tabs:
    * Single Stock Explorer
    * Finance Agent
    * File System Agent
    * General AI Chat (with external data)
    * Action Logs

Error Handling & Safety
- Try/catch around API calls
- Simple retry/backoff for rate limiting
- Path validation for file operations
- Dry-run mode + explicit confirmation for destructive operations
- Soft delete instead of hard delete
- Logs all actions to JSONL and to in-memory session log

External Data Sources for Fresh Context
---------------------------------------
To partially bypass model training cutoffs for *finance-related* questions,
this app pulls in **live external data** and passes it into the LLM:

1. yfinance
   - Recent OHLC data
   - Built-in `news` attribute (if available)
   - Basic company info, market cap, etc.

2. Reddit (no API key required)
   - Uses public JSON search: https://www.reddit.com/search.json
   - Fetches recent posts for a ticker or keyword (e.g., "TSLA Q3 earnings")
   - Extracts titles + snippets as "social sentiment" context

3. Optional: NewsAPI.org (if you configure an API key)
   - Set `NEWSAPI_KEY` in your environment
   - App will call `https://newsapi.org/v2/everything` for the ticker/keyword
   - Extracts fresh article headlines and descriptions

All of that context is **injected into the LLM prompt** for:
- Single Stock Explorer narrative
- Finance Agent tasks
- General AI Chat (when asked about earnings/news/etc.)

Nothing here is financial advice. This is an educational demo.
"""

# ======================================================================================
# Imports and Global Configuration
# ======================================================================================

import os
import sys
import re
import json
import time
import math
import shutil
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# OpenAI client (support new + old styles)
try:
    from openai import OpenAI  # type: ignore
    _OPENAI_CLIENT_CLASS_AVAILABLE = True
except Exception:  # pragma: no cover
    _OPENAI_CLIENT_CLASS_AVAILABLE = False
    import openai  # type: ignore

# ======================================================================================
# App Constants
# ======================================================================================

APP_NAME: str = "MarketWhisper: AI Financial Storyteller & Agent"
APP_VERSION: str = "1.2.0"

ROOT_DIR: Path = Path(os.getcwd()).resolve()
LOG_DIR: Path = ROOT_DIR / "agent_logs"
LOG_DIR.mkdir(exist_ok=True)

AGENT_LOG_FILE: Path = LOG_DIR / "agent_actions.jsonl"
FILE_AGENT_TRASH_DIR_NAME: str = ".file_agent_trash"

DEFAULT_FAST_MODEL: str = "gpt-4o-mini"
DEFAULT_DETAILED_MODEL: str = "gpt-4o-mini"

MAX_FILE_SUMMARY_ITEMS: int = 250

# External news config
NEWSAPI_KEY: Optional[str] = os.environ.get("NEWSAPI_KEY")  # optional
REDDIT_USER_AGENT: str = "MarketWhisperBot/1.0 (https://github.com/yourusername/marketwhisper)"

# ======================================================================================
# OpenAI Client Setup
# ======================================================================================

OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please ensure the OPENAI_API_KEY environment variable is set.")
    st.stop()

if _OPENAI_CLIENT_CLASS_AVAILABLE:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:  # pragma: no cover
    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # type: ignore

# ======================================================================================
# Utility Functions: Logging, Paths, JSON
# ======================================================================================


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating it if necessary."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    """
    Append a JSON record to a `.jsonl` log file.
    """
    try:
        p = Path(path)
        with p.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        print(f"[append_jsonl] Failed to log record: {e}", file=sys.stderr)


def load_jsonl(path: Union[str, Path], max_lines: int = 5000) -> List[Dict[str, Any]]:
    """
    Load up to `max_lines` JSON records from a .jsonl file.
    """
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return records

    try:
        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    # Skip malformed line
                    continue
    except Exception as e:
        print(f"[load_jsonl] Failed to read log file: {e}", file=sys.stderr)

    return records


def safe_join(base: Union[str, Path], *parts: str) -> Path:
    """
    Safely join a base directory with additional path parts, preventing path traversal.
    """
    base_path = Path(base).resolve()
    final_path = base_path
    for part in parts:
        final_path = final_path / part

    final_path = final_path.resolve()
    try:
        final_path.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Attempted path traversal outside base directory: {final_path}")

    return final_path


def format_exception(e: BaseException) -> str:
    """Format an exception and traceback as a short string."""
    return f"{e.__class__.__name__}: {str(e)}"


def short_uid(prefix: str = "id") -> str:
    """
    Small unique-ish ID for logging / action IDs.
    """
    ts = int(time.time())
    rand = int((time.time() - ts) * 10000)
    return f"{prefix}-{ts}-{rand}"


# ======================================================================================
# LLM Helper Functions (with basic retry / backoff)
# ======================================================================================


def _openai_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    response_format: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    base_backoff: float = 1.0,
):
    """
    Low-level wrapper for client.chat.completions.create with simple retry/backoff.

    Retries on:
    - Probable rate-limit (HTTP 429)
    - Probable server errors (HTTP 5xx / "temporarily unavailable")

    Does NOT silently swallow errors: if it fails after retries,
    it re-raises the last exception.
    """
    backoff = base_backoff
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            completion = client.chat.completions.create(**kwargs)
            return completion

        except Exception as e:
            last_error = e
            msg = str(e).lower()
            is_rate_limit = "rate limit" in msg or "429" in msg
            is_server_error = "500" in msg or "503" in msg or "temporarily unavailable" in msg

            if attempt < max_retries - 1 and (is_rate_limit or is_server_error):
                time.sleep(backoff)
                backoff *= 2
                continue
            break

    if last_error:
        raise last_error
    raise RuntimeError("Unknown error calling OpenAI API")


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_FAST_MODEL,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Call the LLM and ask for a JSON object using `response_format={"type": "json_object"}`.
    """
    completion = _openai_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
        return data
    except Exception as e:
        # Surface the actual content in logs for debugging, but not in UI
        print(f"[call_llm_json] Failed to parse JSON. Raw content:\n{content}", file=sys.stderr)
        raise RuntimeError(f"Failed to parse JSON from LLM: {format_exception(e)}")


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_FAST_MODEL,
    max_tokens: int = 800,
    temperature: float = 0.7,
) -> str:
    """
    Call the LLM and get back plain text.
    """
    completion = _openai_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=None,
    )
    return completion.choices[0].message.content.strip()


# ======================================================================================
# External Data Fetchers (Reddit + Optional NewsAPI + yfinance news merge)
# ======================================================================================

def fetch_reddit_posts(
    query: str,
    limit: int = 10,
    sort: str = "new",
    timeout: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Fetch a small set of Reddit posts using the public search JSON endpoint.
    This does **not** require an API key, but is rate-limited and best-effort.

    Returns a list of dicts with keys:
        - "title"
        - "subreddit"
        - "score"
        - "num_comments"
        - "url"
        - "created_utc"
    """
    results: List[Dict[str, Any]] = []
    try:
        url = "https://www.reddit.com/search.json"
        params = {"q": query, "limit": limit, "sort": sort, "restrict_sr": False}
        headers = {"User-Agent": REDDIT_USER_AGENT}
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return []

        data = resp.json()
        children = data.get("data", {}).get("children", [])
        for child in children:
            post = child.get("data", {})
            results.append(
                {
                    "title": post.get("title", ""),
                    "subreddit": post.get("subreddit", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "url": "https://www.reddit.com" + post.get("permalink", ""),
                    "created_utc": post.get("created_utc"),
                }
            )
    except Exception as e:
        print(f"[fetch_reddit_posts] Error: {format_exception(e)}", file=sys.stderr)
        return []

    return results


def fetch_newsapi_articles(
    query: str,
    language: str = "en",
    page_size: int = 10,
    timeout: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Fetch recent news articles using NewsAPI, **if** NEWSAPI_KEY is set.

    Returns a list of dicts:
        - "title"
        - "description"
        - "url"
        - "published_at"
        - "source"
    """
    if not NEWSAPI_KEY:
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": NEWSAPI_KEY,
        }
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 200:
            print(
                f"[fetch_newsapi_articles] Non-200 status {resp.status_code}: {resp.text[:200]}",
                file=sys.stderr,
            )
            return []

        data = resp.json()
        articles_raw = data.get("articles", [])
        articles: List[Dict[str, Any]] = []
        for a in articles_raw:
            articles.append(
                {
                    "title": a.get("title") or "",
                    "description": a.get("description") or "",
                    "url": a.get("url") or "",
                    "published_at": a.get("publishedAt") or "",
                    "source": (a.get("source") or {}).get("name") or "",
                }
            )
        return articles
    except Exception as e:
        print(f"[fetch_newsapi_articles] Error: {format_exception(e)}", file=sys.stderr)
        return []


def merge_news_sources(
    ticker: str,
    yf_news: List[Dict[str, Any]],
    newsapi_articles: List[Dict[str, Any]],
    reddit_posts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combine news from yfinance, NewsAPI, and Reddit into a single structure.

    Returns:
        {
            "headlines": [ ... ],
            "reddit_summaries": [ ... ],
            "newsapi_summaries": [ ... ]
        }
    """
    headlines: List[str] = []
    for item in (yf_news or [])[:10]:
        title = item.get("title")
        if title:
            headlines.append(title)

    newsapi_summaries: List[str] = []
    for a in newsapi_articles[:10]:
        title = a.get("title", "")
        desc = a.get("description", "")
        source = a.get("source", "")
        combined = f"{title} — {desc}".strip(" —")
        if source:
            combined += f" (Source: {source})"
        if combined:
            newsapi_summaries.append(combined)

    reddit_summaries: List[str] = []
    for p in reddit_posts[:10]:
        title = p.get("title", "")
        sub = p.get("subreddit", "")
        score = p.get("score", 0)
        comments = p.get("num_comments", 0)
        if title:
            reddit_summaries.append(
                f"[r/{sub}] {title} (score: {score}, comments: {comments})"
            )

    # Deduplicate headlines-ish
    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            key = s.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    return {
        "headlines": dedupe(headlines),
        "reddit_summaries": dedupe(reddit_summaries),
        "newsapi_summaries": dedupe(newsapi_summaries),
    }


def gather_external_market_context(
    ticker: str,
    extra_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper to gather external market context for a ticker.

    - Uses yfinance's `.news` as base
    - Augments with:
        * Reddit search
        * Optional NewsAPI search if available

    Returns a dict:
        {
            "combined_text": "...",
            "sources": {
                "yf_news": [...],
                "reddit_posts": [...],
                "newsapi_articles": [...]
            }
        }
    """
    ticker_str = ticker.upper().strip()

    # yfinance news
    stock = yf.Ticker(ticker_str)
    yf_news = getattr(stock, "news", []) or []

    # Query string (can incorporate an extra phrase like "Q3 earnings")
    if extra_query:
        query = f"{ticker_str} {extra_query}"
    else:
        query = ticker_str

    reddit_posts = fetch_reddit_posts(query=query, limit=10)
    newsapi_articles = fetch_newsapi_articles(query=query, page_size=10)

    merged = merge_news_sources(
        ticker=ticker_str,
        yf_news=yf_news,
        newsapi_articles=newsapi_articles,
        reddit_posts=reddit_posts,
    )

    # Build a text blob to feed into the LLM
    parts: List[str] = []

    if merged["headlines"]:
        parts.append("Market/financial headlines:")
        for h in merged["headlines"]:
            parts.append(f"- {h}")

    if merged["newsapi_summaries"]:
        parts.append("")
        parts.append("News articles (NewsAPI):")
        for s in merged["newsapi_summaries"]:
            parts.append(f"- {s}")

    if merged["reddit_summaries"]:
        parts.append("")
        parts.append("Reddit social sentiment / posts:")
        for s in merged["reddit_summaries"]:
            parts.append(f"- {s}")

    combined_text = "\n".join(parts).strip()

    return {
        "combined_text": combined_text,
        "sources": {
            "yf_news": yf_news,
            "reddit_posts": reddit_posts,
            "newsapi_articles": newsapi_articles,
        },
    }


# ======================================================================================
# Finance Domain Functions (yfinance + external context)
# ======================================================================================


def fetch_stock_data(ticker: str, period: str) -> Tuple[Optional[float], Any, Any, str, Dict[str, Any]]:
    """
    Fetch stock history, recommendations, news, and basic info using yfinance.

    NOTE (chart correctness):
    - We explicitly sort the history index by time.
    - We ensure it's a simple, single-level DateTimeIndex.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist is None or hist.empty:
            return None, None, None, "", {}

        # Ensure history is sorted and cleaned
        hist = hist.copy()
        hist.sort_index(inplace=True)

        # Just to be safe: drop any duplicated indices
        hist = hist[~hist.index.duplicated(keep="last")]

        recommendations = getattr(stock, "recommendations", None)
        news = getattr(stock, "news", []) or []
        info = getattr(stock, "info", {}) or {}

        price_change: Optional[float] = None
        if len(hist) > 1:
            last_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2]
            if prev_close != 0:
                price_change = ((last_close - prev_close) / prev_close) * 100.0
                price_change = round(price_change, 2)

        news_headlines: List[str] = [
            item.get("title", "No Title Available") for item in news[:5] if item.get("title")
        ]
        news_text = ". ".join(news_headlines)

        return price_change, hist, recommendations, news_text, info

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None, "", {}


def get_current_price(ticker: str) -> Optional[float]:
    """Get the latest close price for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist is None or hist.empty:
            return None
        hist = hist.copy()
        hist.sort_index(inplace=True)
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def mock_sentiment_score(ticker: str) -> Tuple[float, float]:
    """
    Deterministic mock sentiment score used as fallback.

    This is used when:
    - No usable external context is available, or
    - LLM sentiment analysis fails.
    """
    np.random.seed(hash(ticker) % (2**32))
    bullish = np.random.uniform(40, 80)
    bearish = 100 - bullish
    return round(bullish, 1), round(bearish, 1)


def analyze_sentiment_with_llm(
    ticker: str,
    base_news_text: str,
    external_context: Optional[str] = None,
) -> Tuple[float, float, str]:
    """
    Use the LLM to turn recent news + external context into a bullish/bearish score.

    - `base_news_text`: the yfinance headlines string
    - `external_context`: combined text from gather_external_market_context
    """
    combined_text = ""

    if base_news_text.strip():
        combined_text += "YFINANCE HEADLINES:\n" + base_news_text.strip() + "\n\n"

    if external_context and external_context.strip():
        combined_text += "EXTERNAL CONTEXT (NewsAPI/Reddit/etc.):\n" + external_context.strip()

    if not combined_text.strip():
        b, br = mock_sentiment_score(ticker)
        return b, br, (
            "Not enough recent news or external context to compute sentiment. "
            "Using a deterministic demo score instead."
        )

    system_prompt = """
You are a financial sentiment analyst.

You receive a bundle of recent market news and social media snippets
for a single stock. You must:
- Estimate bullish and bearish sentiment (0-100 each).
- Provide a short natural-language explanation.

Rules:
- Output JSON ONLY with keys: bullish, bearish, summary.
- bullish + bearish should be about 100 (does not have to be exact).
- Do not be overconfident; avoid 0 or 100.
- Focus on *short-term* sentiment (1–4 weeks).
"""

    user_prompt = f"""
Ticker: {ticker}

Combined news + social context:
{combined_text}

Return JSON only.
"""

    try:
        data = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=DEFAULT_FAST_MODEL,
            max_tokens=400,
            temperature=0.2,
        )
        bullish = float(data.get("bullish", 50.0))
        bearish = float(data.get("bearish", 50.0))
        summary = str(data.get("summary", "No summary provided.")).strip()
        bullish = max(0.0, min(100.0, bullish))
        bearish = max(0.0, min(100.0, bearish))
        return round(bullish, 1), round(bearish, 1), summary
    except Exception as e:
        st.warning(f"Sentiment analyzer fell back to deterministic mode: {format_exception(e)}")
        b, br = mock_sentiment_score(ticker)
        return b, br, "LLM sentiment failed; using a deterministic demo score instead."


def generate_narrative(
    ticker: str,
    price_change: Optional[float],
    bullish: float,
    bearish: float,
    base_news_text: str,
    recommendations_data: Any,
    external_context_text: Optional[str] = None,
) -> str:
    """
    Generate a market narrative using the LLM, enriched with external context.

    - `base_news_text`: from yfinance headlines
    - `external_context_text`: from gather_external_market_context (News + Reddit)
    """
    persona_prompt = """
You are a senior financial market analyst and storyteller.

You explain stock price movements, fundamental factors, analyst views,
and social sentiment in a clear, concise, factual manner.

Constraints:
- Avoid heavy jargon.
- Be unbiased and balanced.
- Highlight potential bubble risks AND sustained trend drivers.
- Never give direct investment advice. You may describe risks, catalysts,
  and scenarios instead.
"""

    few_shot_examples = """
Example 1:
Stock: AAPL
Price Change: +3.5%
Sentiment: 75% bullish
Recent News (sample): 'Apple announces new chip with improved battery life'
Analyst Recommendations (sample): Strong Buy: 5, Buy: 24, Hold: 14, Sell: 2, Strong Sell: 3
External Context (sample): Reddit chatter generally positive about performance and ecosystem lock-in.
Narrative: Apple’s stock increased by 3.5% following the announcement of its new chip promising longer battery life. Market sentiment is strongly bullish, reflecting optimism around new product demand and ecosystem stickiness. Analyst recommendations are predominantly positive (29 'Strong Buy' or 'Buy' vs. 5 'Sell' or 'Strong Sell'), reinforcing the optimistic medium-term outlook. However, the strong run-up may price in a lot of good news already, leaving the stock vulnerable to short-term pullbacks if execution or guidance disappoint.

Example 2:
Stock: TSLA
Price Change: -4.2%
Sentiment: 60% bearish
Recent News (sample): 'Tesla faces regulatory scrutiny in Europe'
Analyst Recommendations (sample): Strong Buy: 1, Buy: 5, Hold: 10, Sell: 8, Strong Sell: 4
External Context (sample): Reddit and Twitter show mixed reactions, with some long-term bulls but concerns about valuation and regulatory risk.
Narrative: Tesla's 4.2% decline reflects increased concern over regulatory challenges in Europe and uncertainty around demand in key markets. Sentiment tilts slightly bearish, as more analysts recommend 'Hold' or 'Sell' compared with outright 'Buy' ratings. Online discussion remains polarized: long-term supporters emphasize the company’s technology and brand strength, while skeptics focus on valuation and execution risks. Together, this suggests a more cautious short-term stance while the market waits for clearer signals on regulation and growth.
"""

    recommendation_summary = "Analyst Recommendations: "
    if recommendations_data is not None and hasattr(recommendations_data, "empty") and not recommendations_data.empty:
        latest_rec = recommendations_data.iloc[0]
        recommendation_summary += (
            f"Strong Buy: {latest_rec.get('strongBuy', 'N/A')}, "
            f"Buy: {latest_rec.get('buy', 'N/A')}, "
            f"Hold: {latest_rec.get('hold', 'N/A')}, "
            f"Sell: {latest_rec.get('sell', 'N/A')}, "
            f"Strong Sell: {latest_rec.get('strongSell', 'N/A')}"
        )
    else:
        recommendation_summary += "No recent analyst recommendations available."

    pc_str = "N/A"
    if price_change is not None:
        pc_str = f"{price_change:+.2f}%"

    extended_news = base_news_text or "(no yfinance headlines)"
    if external_context_text:
        extended_news += "\n\nExternal context (NewsAPI/Reddit/etc.):\n" + external_context_text

    user_prompt = f"""
{persona_prompt}

{few_shot_examples}

Now analyze this stock using the same tone, level of detail, and structure.

Stock: {ticker.upper()}
Price Change: {pc_str}
Sentiment: {bullish}% bullish, {bearish}% bearish
Recent News + Context:
{extended_news}
{recommendation_summary}

Write a concise narrative (6–10 sentences) that:
- explains recent price behavior in plain language,
- describes how sentiment and news might be connected,
- highlights 1–2 key risks and 1–2 potential catalysts,
- clearly avoids direct "buy/sell" recommendations.
"""

    try:
        narrative = call_llm_text(
            system_prompt=persona_prompt,
            user_prompt=user_prompt,
            model=DEFAULT_DETAILED_MODEL,
            max_tokens=650,
            temperature=0.65,
        )
        return narrative.strip()
    except Exception as e:
        st.error(f"Error generating narrative: {format_exception(e)}")
        return "Could not generate narrative due to an LLM error."


def compare_stocks(
    tickers: List[str],
    period: str = "6mo",
) -> Dict[str, Any]:
    """
    Fetch and compare multiple tickers over a given period.

    - Returns normalized price series (starting at 100)
    - Computes simple returns
    - Generates a short narrative using the LLM
    """
    results: Dict[str, pd.Series] = {}
    returns: Dict[str, float] = {}

    for t in tickers:
        price_change, hist, _, _, _ = fetch_stock_data(t, period)
        if hist is None or hist.empty:
            continue
        close = hist["Close"].copy()
        close = close.sort_index()
        normalized = close / close.iloc[0] * 100.0
        results[t] = normalized
        if price_change is not None:
            returns[t] = price_change

    if not results:
        return {
            "tickers": tickers,
            "period": period,
            "returns": {},
            "normalized_prices": pd.DataFrame(),
            "analysis": "Could not load any of the requested tickers.",
        }

    df_norm = pd.DataFrame(results)

    summary_text = " | ".join([f"{t}: {r:+.2f}%" for t, r in returns.items()])
    prompt = f"""
You are a portfolio analyst.

Here are approximate recent returns over the period '{period}':
{summary_text}

Explain, in a short paragraph, how these stocks performed relative to each other
and what a casual investor might notice. Avoid giving direct investment advice.
"""

    try:
        analysis = call_llm_text(
            system_prompt="You explain relative stock performance in a neutral, educational way.",
            user_prompt=prompt,
            model=DEFAULT_FAST_MODEL,
            max_tokens=400,
            temperature=0.6,
        )
    except Exception as e:
        st.warning(f"Could not generate comparison narrative: {format_exception(e)}")
        analysis = "Comparison narrative unavailable due to an LLM error."

    return {
        "tickers": tickers,
        "period": period,
        "returns": returns,
        "normalized_prices": df_norm,
        "analysis": analysis,
    }


def build_portfolio_allocation(
    tickers: List[str],
    capital: float,
    risk_level: str = "balanced",
) -> Dict[str, Any]:
    """
    Build a very simple equal-weight model portfolio across the given tickers.

    This is intentionally not "smart" – it is an educational example.
    """
    prices: Dict[str, float] = {}
    for t in tickers:
        price = get_current_price(t)
        if price is not None and price > 0:
            prices[t] = price

    if not prices:
        return {
            "capital": capital,
            "risk_level": risk_level,
            "allocations": [],
            "cash_left": capital,
            "explanation": "Could not fetch prices for any ticker.",
        }

    n = len(prices)
    equal_weight = 1.0 / n
    dollars_per_ticker = capital * equal_weight

    allocations: List[Dict[str, Any]] = []
    total_invested = 0.0

    for t, price in prices.items():
        shares = int(dollars_per_ticker // price)
        invested = shares * price
        total_invested += invested
        weight_pct = (invested / capital * 100.0) if capital > 0 else 0.0
        allocations.append(
            {
                "ticker": t,
                "price": round(price, 2),
                "shares": int(shares),
                "invested": round(invested, 2),
                "weight_pct": round(weight_pct, 2),
            }
        )

    cash_left = round(capital - total_invested, 2)

    description = "\n".join(
        [f"{a['ticker']}: {a['shares']} shares (~${a['invested']})" for a in allocations]
    )
    prompt = f"""
You are a portfolio explainer bot.

The user has a {risk_level} risk profile and {capital:.2f} USD of capital.
We built an equal-weight demo portfolio with these positions:
{description}
Uninvested cash: {cash_left:.2f} USD.

Explain in 4–6 sentences what this simple allocation is doing,
what kind of risk profile it roughly corresponds to, and 1–2 things
the user should pay attention to (without giving direct financial advice).
"""

    try:
        explanation = call_llm_text(
            system_prompt="You explain portfolios in friendly, non-advisory language.",
            user_prompt=prompt,
            model=DEFAULT_FAST_MODEL,
            max_tokens=350,
            temperature=0.7,
        )
    except Exception as e:
        st.warning(f"Could not generate portfolio explanation: {format_exception(e)}")
        explanation = (
            "This is a simple equal-weight portfolio across the selected tickers. "
            "It is for demonstration only and not financial advice."
        )

    return {
        "capital": capital,
        "risk_level": risk_level,
        "allocations": allocations,
        "cash_left": cash_left,
        "explanation": explanation,
    }


# ======================================================================================
# Finance Agent Planner & Executor
# ======================================================================================

def plan_finance_agent(user_goal: str) -> Dict[str, Any]:
    """
    Turn a natural-language goal into a JSON action plan for the finance domain.

    Available action types:

        1. "fetch_stock"
           params: {"ticker": "AAPL", "period": "6mo"}

        2. "compare_stocks"
           params: {"tickers": ["AAPL", "MSFT"], "period": "1y"}

        3. "build_portfolio"
           params: {"tickers": ["AAPL", "MSFT"], "capital": 10000, "risk_level": "balanced"}

        4. "external_news_only"
           params: {"ticker": "NVDA", "query": "Q3 earnings"}  # gather context only

    The planner is intentionally flexible: if the user mixes "analyze NVDA Q3 earnings"
    and "compare it with AMD", the agent can choose multiple actions.

    It returns ONLY JSON:
        {
          "plan": "short description",
          "actions": [
             {"id": "step1", "type": "...", "params": {...}},
             ...
          ]
        }
    """
    system_prompt = """
You are a planning agent for a stock-market assistant.

Your job:
- Read the user's goal.
- Decide what *structured* actions to run.
- Return a JSON plan with a small list of actions.

Available action types:

1. "fetch_stock"
   params: {"ticker": "AAPL", "period": "6mo"}  # period optional
   Use when the user wants an in-depth look at 1 stock.

2. "compare_stocks"
   params: {"tickers": ["AAPL", "MSFT"], "period": "1y"}  # period optional
   Use when the user explicitly asks to compare multiple stocks.

3. "build_portfolio"
   params: {"tickers": ["AAPL", "MSFT"], "capital": 10000, "risk_level": "conservative"|"balanced"|"aggressive"}
   Use when the user mentions an amount of money to allocate across multiple tickers.

4. "external_news_only"
   params: {"ticker": "NVDA", "query": "Q3 earnings"}  # query optional
   Use when the user mostly wants a narrative about *very recent news / earnings*,
   and not so much charts or portfolio math.

Return ONLY valid JSON with this structure:
{
  "plan": "short natural language description of what you will do",
  "actions": [
     {"id": "step1", "type": "...", "params": {...}},
     ...
  ]
}

If the request is clearly unrelated to finance/markets, return:
{"plan": "no_actions_needed", "actions": []}
"""

    user_prompt = f"User goal: {user_goal}"

    plan = call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=DEFAULT_FAST_MODEL,
        max_tokens=700,
        temperature=0.25,
    )
    return plan


def execute_finance_agent_actions(
    plan_json: Dict[str, Any],
    use_llm_sentiment: bool = True,
) -> List[Dict[str, Any]]:
    """
    Execute the finance actions produced by `plan_finance_agent`.
    """
    results: List[Dict[str, Any]] = []
    actions = plan_json.get("actions", []) if isinstance(plan_json, dict) else []

    if not isinstance(actions, list):
        actions = []

    for action in actions:
        atype = action.get("type")
        params = action.get("params", {}) or {}

        if atype == "fetch_stock":
            ticker = params.get("ticker")
            period = params.get("period", "6mo")
            if not ticker:
                results.append({"type": "error", "error": "fetch_stock action missing 'ticker'."})
                continue

            price_change, hist, recs, news_text, info = fetch_stock_data(ticker, period)
            if hist is None or hist.empty:
                results.append({"type": "error", "error": f"Could not load data for {ticker}."})
                continue

            external_ctx_obj = gather_external_market_context(ticker)
            external_ctx_text = external_ctx_obj.get("combined_text", "")

            if use_llm_sentiment:
                bullish, bearish, sentiment_summary = analyze_sentiment_with_llm(
                    ticker,
                    base_news_text=news_text or "",
                    external_context=external_ctx_text,
                )
            else:
                bullish, bearish = mock_sentiment_score(ticker)
                sentiment_summary = (
                    "Deterministic demo sentiment score (LLM sentiment disabled)."
                )

            narrative = generate_narrative(
                ticker=ticker,
                price_change=price_change,
                bullish=bullish,
                bearish=bearish,
                base_news_text=news_text,
                recommendations_data=recs,
                external_context_text=external_ctx_text,
            )

            results.append(
                {
                    "type": "fetch_stock",
                    "ticker": ticker,
                    "period": period,
                    "price_change": price_change,
                    "hist": hist,
                    "recommendations": recs,
                    "news_text": news_text,
                    "info": info,
                    "bullish": bullish,
                    "bearish": bearish,
                    "sentiment_summary": sentiment_summary,
                    "external_context": external_ctx_text,
                    "narrative": narrative,
                }
            )

        elif atype == "compare_stocks":
            tickers = params.get("tickers", [])
            period = params.get("period", "6mo")
            if not tickers or len(tickers) < 2:
                results.append({"type": "error", "error": "compare_stocks needs at least 2 tickers."})
                continue
            comp = compare_stocks(tickers, period)
            comp["type"] = "compare_stocks"
            results.append(comp)

        elif atype == "build_portfolio":
            tickers = params.get("tickers", [])
            try:
                capital = float(params.get("capital", 0))
            except Exception:
                capital = 0.0
            risk_level = params.get("risk_level", "balanced")
            if not tickers or capital <= 0:
                results.append(
                    {
                        "type": "error",
                        "error": "build_portfolio needs 'tickers' and a positive 'capital' amount.",
                    }
                )
                continue
            portfolio = build_portfolio_allocation(tickers, capital, risk_level)
            portfolio["type"] = "build_portfolio"
            results.append(portfolio)

        elif atype == "external_news_only":
            ticker = params.get("ticker")
            query = params.get("query") or "recent news"
            if not ticker:
                results.append({"type": "error", "error": "external_news_only requires 'ticker'."})
                continue

            # We still fetch some basic price info, but the focus is narrative.
            price_change, hist, recs, news_text, info = fetch_stock_data(ticker, "1mo")

            external_ctx_obj = gather_external_market_context(ticker, extra_query=query)
            external_ctx_text = external_ctx_obj.get("combined_text", "")

            if use_llm_sentiment:
                bullish, bearish, sentiment_summary = analyze_sentiment_with_llm(
                    ticker,
                    base_news_text=news_text or "",
                    external_context=external_ctx_text,
                )
            else:
                bullish, bearish = mock_sentiment_score(ticker)
                sentiment_summary = "Deterministic demo sentiment score (LLM sentiment disabled)."

            narrative = generate_narrative(
                ticker=ticker,
                price_change=price_change,
                bullish=bullish,
                bearish=bearish,
                base_news_text=news_text,
                recommendations_data=recs,
                external_context_text=external_ctx_text,
            )

            results.append(
                {
                    "type": "external_news_only",
                    "ticker": ticker,
                    "query": query,
                    "external_context": external_ctx_text,
                    "price_change": price_change,
                    "bullish": bullish,
                    "bearish": bearish,
                    "sentiment_summary": sentiment_summary,
                    "narrative": narrative,
                }
            )

        else:
            results.append(
                {
                    "type": "error",
                    "error": f"Unknown action type: {atype}",
                    "raw_action": action,
                }
            )

    return results


# ======================================================================================
# File System Agent: Summarization, Planning, Validation, Execution
# ======================================================================================

def summarize_directory(
    base_dir: Union[str, Path],
    max_items: int = MAX_FILE_SUMMARY_ITEMS,
    max_depth: int = 2,
) -> Dict[str, Any]:
    """
    Summarize the contents of a directory for the file agent.
    """
    base = Path(base_dir).resolve()
    if not base.exists() or not base.is_dir():
        raise ValueError(f"Base directory does not exist or is not a directory: {base}")

    items: List[Dict[str, Any]] = []
    truncated = False

    for root, dirs, files in os.walk(base):
        current = Path(root)
        try:
            rel = current.relative_to(base)
            depth = len(rel.parts)
        except ValueError:
            continue

        if depth > max_depth:
            dirs[:] = []
            continue

        # Folders
        for d in dirs:
            if len(items) >= max_items:
                truncated = True
                dirs[:] = []
                break
            path_rel = str((current / d).relative_to(base))
            items.append(
                {
                    "path": path_rel,
                    "type": "folder",
                    "size": None,
                    "depth": depth + 1,
                }
            )

        if truncated:
            break

        # Files
        for f in files:
            if len(items) >= max_items:
                truncated = True
                break
            p = current / f
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                size = None
            path_rel = str(p.relative_to(base))
            items.append(
                {
                    "path": path_rel,
                    "type": "file",
                    "size": size,
                    "depth": depth + 1,
                }
            )

        if truncated:
            break

    file_count = sum(1 for it in items if it["type"] == "file")
    folder_count = sum(1 for it in items if it["type"] == "folder")

    examples = items[: min(len(items), 12)]
    bullet_lines = []
    for it in examples:
        size_str = f"{it['size']} bytes" if (it["size"] is not None and it["type"] == "file") else "-"
        bullet_lines.append(f"- {it['type']}: {it['path']} (size: {size_str})")

    if truncated:
        bullet_lines.append("- (truncated summary; directory contains more items)")

    human_summary = textwrap.dedent(
        f"""
        Base directory: {base}
        Number of folders: {folder_count}
        Number of files: {file_count}

        Example entries:
        {os.linesep.join(bullet_lines)}
        """
    ).strip()

    return {
        "base_dir": str(base),
        "items": items,
        "truncated": truncated,
        "human_summary": human_summary,
    }


def plan_file_agent_actions(
    base_dir: Union[str, Path],
    user_goal: str,
    dir_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use the LLM to create a file-operation plan (JSON) for the File System Agent.
    """
    base_dir_str = str(Path(base_dir).resolve())
    human_summary = dir_summary.get("human_summary", "")

    system_prompt = f"""
You are a careful file-organization planning agent.

You are given:
- A base directory (the root within which you are allowed to operate).
- A short summary of files/folders under that base directory.
- A user goal describing how they want this directory organized.

Your job is to return a JSON plan with safe actions to reorganize files/folders.

SAFETY AND SCOPE:
- You MUST operate only inside the base directory.
- Use ONLY RELATIVE paths from the base directory. (Do NOT include the absolute base directory.)
- Do NOT use absolute paths.
- Do NOT reference any paths that are not mentioned or implied by the summary.
- Prefer moving files into folders instead of deleting them.
- For deletions, always use 'delete_soft' (moving into a trash folder) instead of permanent deletion.

Allowed action types:

1. "mkdir" (create folder)
   params: {{
     "path": "relative/path/to/new/folder"
   }}

2. "move" (move or rename a file/folder)
   params: {{
     "src": "existing/path/from/base",
     "dest": "new/path/from/base"
   }}

3. "delete_soft" (soft delete by moving to trash folder)
   params: {{
     "path": "existing/path/from/base"
   }}

JSON FORMAT:
Return ONLY valid JSON with this structure:

{{
  "plan": "short natural language description of what you will do",
  "actions": [
     {{"id": "step1", "type": "mkdir" | "move" | "delete_soft", "params": {{...}}}},
     ...
  ]
}}

If no changes are needed, return:
{{"plan": "no_changes_needed", "actions": []}}
"""

    user_prompt = f"""
Base directory (absolute path): {base_dir_str}

Directory summary:
{human_summary}

User goal:
{user_goal}

Remember:
- Use ONLY relative paths (relative to the base directory).
- Use only the allowed action types.
- Use 'delete_soft' instead of actual deletion.
Return JSON only.
"""

    plan = call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=DEFAULT_FAST_MODEL,
        max_tokens=800,
        temperature=0.25,
    )
    return plan


def validate_file_agent_actions(
    base_dir: Union[str, Path],
    plan: Dict[str, Any],
    dir_summary: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate the file-agent actions before execution.

    Ensures:
    - Paths are relative and inside base_dir
    - Sources exist when moving or soft-deleting
    """
    base = Path(base_dir).resolve()
    items = dir_summary.get("items", [])
    existing_paths = {it["path"] for it in items}

    actions = plan.get("actions", []) if isinstance(plan, dict) else []
    if not isinstance(actions, list):
        actions = []

    validated: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for idx, action in enumerate(actions, start=1):
        atype = action.get("type")
        params = action.get("params", {}) or {}
        aid = action.get("id") or short_uid("file-step")

        def add_warning(msg: str) -> None:
            warnings.append(f"[action {idx} / {aid}] {msg}")

        def resolve_rel(p: str) -> Optional[Path]:
            p = p.strip()
            if not p:
                add_warning("Empty path encountered.")
                return None
            if os.path.isabs(p):
                add_warning(f"Absolute path not allowed: {p}")
                return None
            try:
                return safe_join(base, p)
            except ValueError as ve:
                add_warning(str(ve))
                return None

        if atype == "mkdir":
            rel = str(params.get("path", "")).strip()
            if not rel:
                add_warning("mkdir missing 'path'")
                continue
            abs_path = resolve_rel(rel)
            if abs_path is None:
                continue
            validated.append(
                {
                    "id": aid,
                    "type": "mkdir",
                    "rel_path": rel,
                    "abs_path": str(abs_path),
                    "params": {"path": rel},
                }
            )

        elif atype == "move":
            src_rel = str(params.get("src", "")).strip()
            dest_rel = str(params.get("dest", "")).strip()
            if not src_rel or not dest_rel:
                add_warning("move requires both 'src' and 'dest'")
                continue
            src_abs = resolve_rel(src_rel)
            dest_abs = resolve_rel(dest_rel)
            if src_abs is None or dest_abs is None:
                continue
            if src_rel not in existing_paths and not src_abs.exists():
                add_warning(f"Source path does not exist: {src_rel}")
                continue
            validated.append(
                {
                    "id": aid,
                    "type": "move",
                    "rel_src": src_rel,
                    "rel_dest": dest_rel,
                    "abs_src": str(src_abs),
                    "abs_dest": str(dest_abs),
                    "params": {"src": src_rel, "dest": dest_rel},
                }
            )

        elif atype == "delete_soft":
            rel = str(params.get("path", "")).strip()
            if not rel:
                add_warning("delete_soft missing 'path'")
                continue
            abs_path = resolve_rel(rel)
            if abs_path is None:
                continue
            if rel not in existing_paths and not abs_path.exists():
                add_warning(f"Path to delete_soft does not exist: {rel}")
                continue
            validated.append(
                {
                    "id": aid,
                    "type": "delete_soft",
                    "rel_path": rel,
                    "abs_path": str(abs_path),
                    "params": {"path": rel},
                }
            )

        else:
            add_warning(f"Unknown action type: {atype!r}")
            continue

    return validated, warnings


def execute_file_agent_actions(
    base_dir: Union[str, Path],
    actions: List[Dict[str, Any]],
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Execute validated file-agent actions with safety (soft-delete, no path traversal).
    """
    base = Path(base_dir).resolve()
    trash_dir = ensure_dir(base / FILE_AGENT_TRASH_DIR_NAME)

    results: List[Dict[str, Any]] = []

    for action in actions:
        atype = action["type"]
        aid = action.get("id", short_uid("file-step"))
        result: Dict[str, Any] = {"id": aid, "type": atype, "status": "pending"}

        try:
            if atype == "mkdir":
                abs_path = Path(action["abs_path"])
                if dry_run:
                    result["status"] = "ok (dry-run)"
                    result["message"] = f"Would create folder: {abs_path}"
                else:
                    abs_path.mkdir(parents=True, exist_ok=True)
                    result["status"] = "ok"
                    result["message"] = f"Created folder: {abs_path}"

            elif atype == "move":
                abs_src = Path(action["abs_src"])
                abs_dest = Path(action["abs_dest"])
                if dry_run:
                    result["status"] = "ok (dry-run)"
                    result["message"] = f"Would move {abs_src} -> {abs_dest}"
                else:
                    abs_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(abs_src), str(abs_dest))
                    result["status"] = "ok"
                    result["message"] = f"Moved {abs_src} -> {abs_dest}"

            elif atype == "delete_soft":
                abs_path = Path(action["abs_path"])
                if dry_run:
                    result["status"] = "ok (dry-run)"
                    result["message"] = f"Would soft-delete: {abs_path}"
                else:
                    rel = abs_path.relative_to(base)
                    trash_target = trash_dir / rel
                    trash_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(abs_path), str(trash_target))
                    result["status"] = "ok"
                    result["message"] = f"Soft-deleted (moved to trash): {abs_path}"

            else:
                result["status"] = "error"
                result["message"] = f"Unknown action type at execution: {atype}"

        except Exception as e:
            result["status"] = "error"
            result["message"] = format_exception(e)

        results.append(result)

    return results


# ======================================================================================
# General AI Chat: Finance-aware, with External Context
# ======================================================================================

TICKER_STOPWORDS = {
    "I", "A", "AN", "THE", "AND", "OR", "BUT", "FOR", "YOU",
    "ARE", "IS", "AM", "ON", "OFF", "UP", "AS", "IT", "TO",
    "DO", "BY", "OF", "IN", "ETF", "CEO", "CFO", "EPS",
}


def detect_ticker_candidates(text: str, max_candidates: int = 3) -> List[str]:
    """
    Very simple heuristic to detect ticker-like tokens (e.g., AAPL, TSLA).
    """
    candidates = re.findall(r"\b[A-Z]{1,5}\b", text)
    unique: List[str] = []
    for c in candidates:
        if c in TICKER_STOPWORDS:
            continue
        if c not in unique:
            unique.append(c)
        if len(unique) >= max_candidates:
            break
    return unique


def build_finance_context_for_chat(
    user_prompt: str,
    use_external_feeds: bool,
    period: str = "6mo",
) -> str:
    """
    If the user is asking about earnings or news for specific tickers,
    fetch some data and news via yfinance + external feeds to give the LLM
    extra context.
    """
    lowered = user_prompt.lower()
    keywords = ["earnings", "q1", "q2", "q3", "q4", "results", "report", "guidance", "dividend", "news"]
    if not any(k in lowered for k in keywords):
        return ""

    tickers = detect_ticker_candidates(user_prompt, max_candidates=2)
    if not tickers:
        return ""

    sections: List[str] = []
    for t in tickers:
        try:
            price_change, hist, recs, news_text, info = fetch_stock_data(t, period)
            if hist is None or hist.empty:
                continue
            last_close = hist["Close"].iloc[-1]
            last_close = float(last_close) if last_close is not None else None

            pc_str = "N/A"
            if price_change is not None:
                pc_str = f"{price_change:+.2f}%"

            company_name = info.get("shortName") or info.get("longName") or t

            section_lines = [
                f"TICKER: {t}",
                f"Company: {company_name}",
                f"Approx. last close: {last_close:.2f}" if last_close is not None else "Approx. last close: N/A",
                f"Recent price change over last bar: {pc_str}",
            ]

            headlines = []
            if news_text:
                for h in news_text.split(". "):
                    h = h.strip()
                    if h:
                        headlines.append(h)
                        if len(headlines) >= 5:
                            break

            if headlines:
                section_lines.append("Recent headlines (yfinance):")
                for h in headlines:
                    section_lines.append(f"- {h}")
            else:
                section_lines.append("Recent headlines (yfinance): (none found or not available)")

            if use_external_feeds:
                ext_ctx_obj = gather_external_market_context(t, extra_query="earnings")
                ext_text = ext_ctx_obj.get("combined_text", "")
                if ext_text:
                    section_lines.append("")
                    section_lines.append("External context (NewsAPI/Reddit):")
                    for line in ext_text.splitlines():
                        section_lines.append(line)

            sections.append("\n".join(section_lines))
        except Exception as e:
            print(f"[build_finance_context_for_chat] Error on {t}: {format_exception(e)}", file=sys.stderr)
            continue

    if not sections:
        return ""

    return "\n\n" + "\n\n".join(sections)


def general_ai_chat(
    prompt: str,
    detail_level: str = "Medium",
    history: Optional[List[Dict[str, str]]] = None,
    enable_finance_tools: bool = True,
    enable_external_feeds: bool = True,
) -> str:
    """
    Enhanced general chat:
    - Supports conversation history
    - Configurable detail level (Short / Medium / Deep dive)
    - Optionally uses finance tools (yfinance + Reddit/NewsAPI) for ticker-related
      news/earnings questions
    - Uses a richer prompt template to encourage structured answers

    This does NOT magically give the LLM real-time awareness of everything,
    but it *does* feed fresh context into the prompt for finance topics.
    """
    detail_level_map = {
        "Short": 350,
        "Medium": 800,
        "Deep dive": 1400,
    }
    max_tokens = detail_level_map.get(detail_level, 800)

    finance_context = ""
    if enable_finance_tools:
        finance_context = build_finance_context_for_chat(
            prompt,
            use_external_feeds=enable_external_feeds,
            period="6mo",
        )

    system_prompt = textwrap.dedent("""
        You are an advanced assistant with strong skills in:
        - financial markets and corporate earnings
        - software engineering and debugging
        - explaining complex ideas at different depth levels (short, medium, deep-dive)
        - summarizing news and documents concisely

        Rules:
        - Think through problems step by step internally, but **do not** reveal chain-of-thought.
        - Present only the final explanation, structured with:
            * short sections with headings (when appropriate)
            * bullet points for lists
            * examples or analogies when they help understanding
        - If you reference specific numeric info from the provided external data,
          mention that it is approximate and may be time-limited.
        - If you are unsure about current real-world facts (like very recent events),
          say that explicitly instead of guessing.
        - Avoid giving financial, legal, or medical advice. You may provide educational
          information and general guidance only.
    """).strip()

    if finance_context:
        user_prompt = textwrap.dedent(f"""
        User question:
        {prompt}

        Additional financial data feed (from external providers like yfinance, Reddit, NewsAPI):
        {finance_context}

        Use this data feed where helpful, but do not overstate certainty.
        """).strip()
    else:
        user_prompt = prompt

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-10:]:
            if "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_prompt})

    completion = _openai_chat_completion(
        messages=messages,
        model=DEFAULT_DETAILED_MODEL,
        max_tokens=max_tokens,
        temperature=0.7,
        response_format=None,
    )
    answer = completion.choices[0].message.content.strip()
    return answer


# ======================================================================================
# Streamlit UI: Setup & Session State
# ======================================================================================

st.set_page_config(
    layout="wide",
    page_title=APP_NAME,
)

with st.sidebar:
    st.markdown(f"## ⚙️ Settings ({APP_VERSION})")

    analysis_depth = st.selectbox("Finance analysis depth", ["Quick", "Detailed"], index=1)

    use_llm_sentiment = st.checkbox(
        "Use LLM-based sentiment scoring",
        value=True,
        help="If disabled, a deterministic demo score will be used for bullish/bearish sentiment.",
    )

    enable_external_feeds_global = st.checkbox(
        "Use external feeds (Reddit + optional NewsAPI) where supported",
        value=True,
        help="If OFF, app will ignore Reddit/NewsAPI and rely on yfinance only.",
    )

    show_raw_finance_plan = st.checkbox(
        "Show raw finance agent plan (JSON)",
        value=False,
        help="Displays the low-level JSON plan returned by the finance agent.",
    )

    show_raw_file_plan = st.checkbox(
        "Show raw file agent plan (JSON)",
        value=False,
        help="Displays the low-level JSON plan returned by the file agent.",
    )

    st.markdown("---")
    st.markdown("### API Status")
    if OPENAI_API_KEY:
        st.success("✅ OpenAI API key loaded")
    else:
        st.error("❌ No API key (set OPENAI_API_KEY)")

    if enable_external_feeds_global:
        st.caption(
            "External feeds enabled (Reddit search; NewsAPI if `NEWSAPI_KEY` is set). "
            "All requests are best-effort and may fail silently under heavy rate-limits."
        )

    st.markdown("---")
    st.caption(
        "This application is for educational purposes only.\n"
        "**Nothing here is financial advice.**"
    )

# Session state defaults
if "finance_analysis_done" not in st.session_state:
    st.session_state.finance_analysis_done = False

if "finance_stock_info" not in st.session_state:
    st.session_state.finance_stock_info = None

if "finance_agent_last_plan" not in st.session_state:
    st.session_state.finance_agent_last_plan = None

if "finance_agent_last_results" not in st.session_state:
    st.session_state.finance_agent_last_results = None

if "file_agent_last_summary" not in st.session_state:
    st.session_state.file_agent_last_summary = None

if "file_agent_last_plan" not in st.session_state:
    st.session_state.file_agent_last_plan = None

if "file_agent_last_validated" not in st.session_state:
    st.session_state.file_agent_last_validated = None

if "file_agent_last_exec_results" not in st.session_state:
    st.session_state.file_agent_last_exec_results = None

if "action_log" not in st.session_state:
    st.session_state.action_log = []

if "general_chat_history" not in st.session_state:
    st.session_state.general_chat_history: List[Dict[str, str]] = []

if "last_chat_response" not in st.session_state:
    st.session_state.last_chat_response = None

# ======================================================================================
# Tabs
# ======================================================================================

st.title(APP_NAME)
st.subheader("LLM-driven narratives, live external context, and multi-agent automation.")

tab_finance, tab_finance_agent, tab_file_agent, tab_chat, tab_logs = st.tabs(
    [
        "📈 Single Stock Explorer",
        "🤖 Finance Agent",
        "📂 File System Agent",
        "💬 General AI Chat",
        "📝 Action Logs",
    ]
)

# ======================================================================================
# Tab 4: General AI Chat (Richer, Finance-aware, External Feeds)
# ======================================================================================

with tab_chat:
    st.subheader("General-purpose Chat with External Market Context")

    chat_detail_level = st.selectbox(
        "Answer detail level",
        ["Short", "Medium", "Deep dive"],
        index=1,
        help="Controls how long and detailed the answers will be.",
    )

    chat_enable_finance_tools = st.checkbox(
        "Use finance tools (tickers, prices, earnings, etc.) when helpful",
        value=True,
        help="If enabled, the assistant will look for ticker symbols and fetch basic price/news data.",
    )

    chat_enable_external_feeds = st.checkbox(
        "Pull external feeds (Reddit + optional NewsAPI) for finance questions",
        value=enable_external_feeds_global,
        help="If ON, the assistant may fetch Reddit posts and/or NewsAPI articles related to your question.",
    )

    chat_prompt = st.text_area(
        "Ask me anything (finance, market events, Q3 earnings, coding, general topics):",
        height=140,
        placeholder="Examples:\n"
                    "- Explain NVIDIA's Q3 earnings using recent news.\n"
                    "- Summarize what happened to TSLA last week.\n"
                    "- Help me debug this Python error.\n"
                    "- Compare AAPL and MSFT over the last year.",
    )

    col_chat_buttons = st.columns(3)
    with col_chat_buttons[0]:
        chat_run = st.button("Send ✉️")
    with col_chat_buttons[1]:
        chat_clear = st.button("Clear chat history")
    with col_chat_buttons[2]:
        chat_clear_last = st.button("Clear last reply")

    if chat_clear:
        st.session_state.general_chat_history = []
        st.session_state.last_chat_response = None

    if chat_clear_last and st.session_state.general_chat_history:
        # Pop last assistant + possibly last user
        if st.session_state.general_chat_history[-1]["role"] == "assistant":
            st.session_state.general_chat_history.pop()
        if st.session_state.general_chat_history and st.session_state.general_chat_history[-1]["role"] == "user":
            st.session_state.general_chat_history.pop()
        st.session_state.last_chat_response = None

    if chat_run and chat_prompt.strip():
        history = st.session_state.general_chat_history

        # Append user message to history
        history.append({"role": "user", "content": chat_prompt.strip()})

        with st.spinner("Talking to the LLM with live market context..."):
            try:
                response = general_ai_chat(
                    prompt=chat_prompt.strip(),
                    detail_level=chat_detail_level,
                    history=history,
                    enable_finance_tools=chat_enable_finance_tools,
                    enable_external_feeds=chat_enable_external_feeds and enable_external_feeds_global,
                )
            except Exception as e:
                response = (
                    "Could not generate a response due to an LLM error. "
                    f"Internal error: {format_exception(e)}"
                )

        history.append({"role": "assistant", "content": response})
        st.session_state.general_chat_history = history
        st.session_state.last_chat_response = response

        # Log chat interaction
        log_entry = {
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "agent": "chat",
            "detail_level": chat_detail_level,
            "finance_tools_used": chat_enable_finance_tools,
            "external_feeds_used": chat_enable_external_feeds and enable_external_feeds_global,
            "user_message": chat_prompt.strip(),
            "assistant_reply_preview": response[:300],
        }
        st.session_state.action_log.append(log_entry)
        append_jsonl(AGENT_LOG_FILE, log_entry)

    # Render conversation
    if st.session_state.general_chat_history:
        st.markdown("### Conversation")
        for msg in st.session_state.general_chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Assistant:** {msg['content']}")
    else:
        st.info("Start a conversation by typing a question or request above.")

# ======================================================================================
# Tab 1: Single Stock Explorer (with external context and fixed charts)
# ======================================================================================

with tab_finance:
    st.subheader("Single Stock Explorer")

    col1, col2 = st.columns(2)

    with col1:
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"]
        selected = st.selectbox("Select a popular stock ticker:", popular_stocks)
        custom_ticker = st.text_input(
            "Or type any other ticker (overrides selection above):",
            value="",
            placeholder="e.g. NFLX",
        )
        ticker_input = (custom_ticker or selected).upper().strip()

    with col2:
        time_period = st.selectbox(
            "Select chart time period:",
            ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"),
            index=2,
        )

    analyze_col, clear_col = st.columns([1, 1])

    with analyze_col:
        analyze_clicked = st.button("🔍 Analyze Stock", type="primary")
    with clear_col:
        clear_clicked = st.button("❌ Clear Analysis")

    if clear_clicked:
        st.session_state.finance_analysis_done = False
        st.session_state.finance_stock_info = None
        st.rerun()

    if analyze_clicked:
        if not ticker_input:
            st.warning("Please enter or select a stock ticker.")
        else:
            with st.spinner(f"Fetching data and generating narrative for {ticker_input}..."):
                price_change, hist, recommendations, news_text, info = fetch_stock_data(
                    ticker_input, time_period
                )

                if hist is not None and not hist.empty:
                    # External context (Reddit + NewsAPI + additional yfinance news)
                    external_ctx_obj = (
                        gather_external_market_context(ticker_input)
                        if enable_external_feeds_global
                        else {"combined_text": "", "sources": {}}
                    )
                    external_ctx_text = external_ctx_obj.get("combined_text", "")

                    if use_llm_sentiment:
                        bullish, bearish, sentiment_summary = analyze_sentiment_with_llm(
                            ticker_input,
                            base_news_text=news_text or "",
                            external_context=external_ctx_text,
                        )
                    else:
                        bullish, bearish = mock_sentiment_score(ticker_input)
                        sentiment_summary = (
                            "Deterministic demo sentiment score (LLM sentiment disabled)."
                        )

                    narrative = generate_narrative(
                        ticker=ticker_input,
                        price_change=price_change,
                        bullish=bullish,
                        bearish=bearish,
                        base_news_text=news_text,
                        recommendations_data=recommendations,
                        external_context_text=external_ctx_text,
                    )

                    st.session_state.finance_analysis_done = True
                    st.session_state.finance_stock_info = {
                        "ticker": ticker_input,
                        "price_change": price_change,
                        "bullish": bullish,
                        "bearish": bearish,
                        "sentiment_summary": sentiment_summary,
                        "narrative": narrative,
                        "hist": hist,
                        "recommendations": recommendations,
                        "news_text": news_text,
                        "info": info,
                        "period": time_period,
                        "external_context": external_ctx_text,
                    }

                    # Log this single-stock analysis as an action
                    log_entry = {
                        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "agent": "finance-single",
                        "ticker": ticker_input,
                        "period": time_period,
                        "price_change": price_change,
                        "bullish": bullish,
                        "bearish": bearish,
                        "external_feeds_used": enable_external_feeds_global,
                    }
                    st.session_state.action_log.append(log_entry)
                    append_jsonl(AGENT_LOG_FILE, log_entry)

                    st.rerun()
                else:
                    st.error("No price history found for that ticker/time period.")

    if st.session_state.finance_analysis_done and st.session_state.finance_stock_info:
        info_data = st.session_state.finance_stock_info
        ticker = info_data["ticker"]
        period = info_data.get("period", time_period)

        st.markdown(f"### Analysis for `{ticker}`")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            pc = info_data["price_change"]
            pc_str = f"{pc:+.2f}%" if pc is not None else "N/A"
            st.metric(label="Price Change (Last Bar)", value=pc_str)
        with metric_col2:
            st.metric(
                label="Sentiment",
                value=f"📈 {info_data['bullish']}% Bullish\n📉 {info_data['bearish']}% Bearish",
            )
        with metric_col3:
            st.caption("Sentiment explanation:")
            st.write(info_data.get("sentiment_summary", ""))

        st.markdown("#### Additional Stock Information")
        info_buttons_col1, info_buttons_col2, info_buttons_col3 = st.columns(3)

        with info_buttons_col1:
            if st.button("Show Market Cap"):
                market_cap = info_data["info"].get("marketCap")
                if market_cap is not None:
                    st.write(f"Market Cap: ${market_cap:,.0f}")
                else:
                    st.write("Market Cap: N/A")

        with info_buttons_col2:
            if st.button("Show Dividend Yield"):
                dividend_yield = info_data["info"].get("dividendYield")
                if dividend_yield is not None:
                    st.write(f"Dividend Yield: {dividend_yield:.2%}")
                else:
                    st.write("Dividend Yield: N/A")

        with info_buttons_col3:
            if st.button("Show P/E Ratio"):
                pe_ratio = info_data["info"].get("trailingPE")
                if pe_ratio is not None:
                    st.write(f"P/E Ratio: {pe_ratio:.2f}")
                else:
                    st.write("P/E Ratio: N/A")

        st.markdown("#### Market Narrative")
        st.write(info_data["narrative"])

        st.markdown(f"#### {ticker} Price Chart ({period})")

        # Ensure chart is using time index correctly
        hist = info_data["hist"].copy()
        hist.sort_index(inplace=True)
        close_series = hist["Close"].rename(ticker)
        st.line_chart(close_series)

        st.markdown("#### Analyst Recommendations (Latest)")
        recommendations = info_data["recommendations"]
        if recommendations is not None and not recommendations.empty:
            latest_rec = recommendations.iloc[0]
            st.write(
                f"Strong Buy: {latest_rec.get('strongBuy', 'N/A')}, "
                f"Buy: {latest_rec.get('buy', 'N/A')}, "
                f"Hold: {latest_rec.get('hold', 'N/A')}, "
                f"Sell: {latest_rec.get('sell', 'N/A')}, "
                f"Strong Sell: {latest_rec.get('strongSell', 'N/A')}"
            )
        else:
            st.write("No recent analyst recommendations available.")

        st.markdown("#### Recent News Headlines (yfinance)")
        news_text = info_data["news_text"]
        if news_text:
            for headline in news_text.split(". "):
                h = headline.strip()
                if h:
                    st.write(f"- {h}")
        else:
            st.write("No recent news available from yfinance.")

        if info_data.get("external_context"):
            st.markdown("#### External Context (Reddit + optional NewsAPI)")
            with st.expander("Show external context used for the narrative"):
                st.text(info_data["external_context"])

# ======================================================================================
# Tab 2: Finance Agent
# ======================================================================================

with tab_finance_agent:
    st.subheader("Finance Agent: Natural-language Task Runner")

    st.markdown(
        "Describe a task and let the agent plan and execute it using live market data "
        "and external context.\n\n"
        "**Examples:**\n"
        "- *\"Compare TSLA and F with performance over the last year\"*\n"
        "- *\"Build a $5,000 balanced portfolio using AAPL, MSFT, and NVDA\"*\n"
        "- *\"Analyze NVDA Q3 earnings\"* (the agent may use external news feeds)\n"
    )

    agent_goal = st.text_area(
        "What do you want the finance agent to do?",
        height=120,
        placeholder="e.g. Build a $10,000 aggressive portfolio with TSLA, NVDA, and AMD",
    )

    agent_col1, agent_col2 = st.columns([1, 1])
    with agent_col1:
        run_agent = st.button("🚀 Plan & Execute Finance Task", type="primary")
    with agent_col2:
        clear_agent = st.button("🧹 Clear Finance Agent Output")

    if clear_agent:
        st.session_state.finance_agent_last_plan = None
        st.session_state.finance_agent_last_results = None
        st.rerun()

    if run_agent and agent_goal.strip():
        with st.spinner("Letting the finance agent plan your task..."):
            try:
                plan = plan_finance_agent(agent_goal.strip())
            except Exception as e:
                st.error(f"Error calling planner: {format_exception(e)}")
                plan = {"plan": "error", "actions": [], "error": str(e)}

            st.session_state.finance_agent_last_plan = plan
            try:
                results = execute_finance_agent_actions(plan, use_llm_sentiment=use_llm_sentiment)
            except Exception as e:
                st.error(f"Error executing finance agent actions: {format_exception(e)}")
                results = [{"type": "error", "error": str(e)}]

            st.session_state.finance_agent_last_results = results

            log_entry = {
                "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "agent": "finance-agent",
                "goal": agent_goal.strip(),
                "plan_summary": plan.get("plan", ""),
                "actions": [a.get("type", "unknown") for a in plan.get("actions", [])]
                if isinstance(plan, dict)
                else [],
            }
            st.session_state.action_log.append(log_entry)
            append_jsonl(AGENT_LOG_FILE, log_entry)

    plan = st.session_state.finance_agent_last_plan
    results = st.session_state.finance_agent_last_results

    if plan:
        st.markdown("### Finance Agent Plan")
        st.write(plan.get("plan", "No plan description provided."))

        if show_raw_finance_plan:
            st.markdown("##### Raw Plan (JSON)")
            st.json(plan)

    if results:
        st.markdown("### Finance Agent Results")

        for idx, res in enumerate(results, start=1):
            rtype = res.get("type")
            st.markdown(f"#### Step {idx}: `{rtype}`")

            if rtype == "fetch_stock":
                ticker = res["ticker"]
                period = res["period"]
                col_a, col_b = st.columns(2)
                with col_a:
                    pc = res["price_change"]
                    pc_str = f"{pc:+.2f}%" if pc is not None else "N/A"
                    st.metric(
                        label=f"{ticker} Price Change ({period})",
                        value=pc_str,
                    )
                with col_b:
                    st.metric(
                        label="Sentiment",
                        value=f"📈 {res['bullish']}% Bullish / 📉 {res['bearish']}% Bearish",
                    )
                    st.caption(res.get("sentiment_summary", ""))

                hist = res["hist"].copy()
                hist.sort_index(inplace=True)
                st.line_chart(hist["Close"].rename(ticker))

                if analysis_depth == "Detailed":
                    st.markdown("**Narrative:**")
                    st.write(res["narrative"])

                if res.get("external_context"):
                    with st.expander("External context used (Reddit + optional NewsAPI)"):
                        st.text(res["external_context"])

            elif rtype == "compare_stocks":
                tickers = res.get("tickers", [])
                st.write(f"Comparing: {', '.join(tickers)} over {res.get('period')}")
                returns = res.get("returns", {})
                if returns:
                    pretty = {t: f"{r:+.2f}%" for t, r in returns.items()}
                    st.write(pretty)
                df_norm = res.get("normalized_prices", pd.DataFrame())
                if not df_norm.empty:
                    st.line_chart(df_norm)
                st.write(res.get("analysis", ""))

            elif rtype == "build_portfolio":
                st.write(f"Capital: ${res.get('capital', 0):,.2f}")
                st.write(f"Risk level: {res.get('risk_level')}")
                allocations = res.get("allocations", [])
                if allocations:
                    df_alloc = pd.DataFrame(allocations)
                    st.dataframe(df_alloc, use_container_width=True)
                st.write(f"Unallocated cash: ${res.get('cash_left', 0):,.2f}")
                st.write(res.get("explanation", ""))

            elif rtype == "external_news_only":
                ticker = res.get("ticker")
                query = res.get("query")
                st.write(f"External news-focused analysis for {ticker} (query: {query})")
                st.metric(
                    label="Sentiment",
                    value=f"📈 {res['bullish']}% Bullish / 📉 {res['bearish']}% Bearish",
                )
                st.caption(res.get("sentiment_summary", ""))
                st.write(res.get("narrative", "No narrative."))
                if res.get("external_context"):
                    with st.expander("External context used"):
                        st.text(res["external_context"])

            elif rtype == "error":
                st.error(res.get("error", "Unknown error"))
                if "raw_action" in res:
                    with st.expander("Show raw action"):
                        st.json(res["raw_action"])

            else:
                st.warning(f"Unrecognized result type: {rtype}")
                with st.expander("Show raw result"):
                    st.json(res)

# ======================================================================================
# Tab 3: File System Agent
# ======================================================================================

with tab_file_agent:
    st.subheader("File System Agent: Local File Organizer")

    st.markdown(
        "This agent operates on directories in your Colab/Streamlit environment.\n\n"
        "**Capabilities:**\n"
        "- Summarize a directory and show files/folders.\n"
        "- Use an LLM to plan moves, renames, and folder creation.\n"
        "- Soft-delete files by moving them into a trash folder.\n"
        "- Validate and execute the plan with safety checks and explicit confirmation.\n\n"
        "**Important:** The agent only works inside the base directory you specify. "
        "Paths are checked so the agent cannot escape that directory."
    )

    default_base_dir = str(ROOT_DIR)
    base_dir_input = st.text_input(
        "Base directory (the agent's sandbox root):",
        value=default_base_dir,
        help="The file agent will only operate inside this directory.",
    )

    col_scan, col_clear = st.columns([1, 1])
    with col_scan:
        scan_clicked = st.button("🔎 Scan Directory")
    with col_clear:
        clear_file_agent = st.button("🧹 Clear File Agent State")

    if clear_file_agent:
        st.session_state.file_agent_last_summary = None
        st.session_state.file_agent_last_plan = None
        st.session_state.file_agent_last_validated = None
        st.session_state.file_agent_last_exec_results = None
        st.rerun()

    if scan_clicked:
        try:
            with st.spinner("Summarizing directory contents..."):
                summary = summarize_directory(base_dir_input)
                st.session_state.file_agent_last_summary = summary
        except Exception as e:
            st.error(f"Error summarizing directory: {format_exception(e)}")

    summary = st.session_state.file_agent_last_summary

    if summary:
        st.markdown("### Directory Summary")
        st.write(f"**Base directory:** `{summary['base_dir']}`")
        st.write(summary["human_summary"])

        items_df = pd.DataFrame(summary["items"])
        if not items_df.empty:
            st.markdown("#### Items (subset)")
            st.dataframe(items_df.head(200), use_container_width=True)
        else:
            st.info("No items found under this directory (within the configured depth).")

        st.markdown("---")
        st.markdown("### Plan File Operations with the Agent")

        file_agent_goal = st.text_area(
            "Describe how you want to organize this directory:",
            height=120,
            placeholder="e.g. Group PDFs into a 'docs' folder, images into 'images', and move old logs into 'archive'.",
        )

        col_plan, col_validate = st.columns([1, 1])
        with col_plan:
            plan_file_agent = st.button("🧠 Generate File-Agent Plan", type="primary")
        with col_validate:
            validate_file_agent_btn = st.button("✅ Validate Plan")

        if plan_file_agent and file_agent_goal.strip():
            with st.spinner("Asking the LLM to design a file-operation plan..."):
                try:
                    plan = plan_file_agent_actions(
                        base_dir=summary["base_dir"],
                        user_goal=file_agent_goal.strip(),
                        dir_summary=summary,
                    )
                except Exception as e:
                    st.error(f"Error generating file-agent plan: {format_exception(e)}")
                    plan = {"plan": "error", "actions": [], "error": str(e)}

                st.session_state.file_agent_last_plan = plan
                st.session_state.file_agent_last_validated = None
                st.session_state.file_agent_last_exec_results = None

                log_entry = {
                    "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "agent": "file-agent",
                    "goal": file_agent_goal.strip(),
                    "plan_summary": plan.get("plan", ""),
                    "actions": [a.get("type", "unknown") for a in plan.get("actions", [])]
                    if isinstance(plan, dict)
                    else [],
                }
                st.session_state.action_log.append(log_entry)
                append_jsonl(AGENT_LOG_FILE, log_entry)

        plan = st.session_state.file_agent_last_plan

        if plan:
            st.markdown("### File-Agent Plan")
            st.write(plan.get("plan", "No plan description provided."))

            if show_raw_file_plan:
                st.markdown("##### Raw File-Agent Plan (JSON)")
                st.json(plan)

        if validate_file_agent_btn and plan:
            with st.spinner("Validating plan and checking paths..."):
                try:
                    validated_actions, warnings = validate_file_agent_actions(
                        base_dir=summary["base_dir"],
                        plan=plan,
                        dir_summary=summary,
                    )
                except Exception as e:
                    st.error(f"Error validating file-agent plan: {format_exception(e)}")
                    validated_actions, warnings = [], []

                st.session_state.file_agent_last_validated = validated_actions

                if warnings:
                    st.markdown("#### Validation Warnings")
                    for w in warnings:
                        st.warning(w)
                else:
                    st.success("No validation warnings; actions look structurally safe.")

        validated_actions = st.session_state.file_agent_last_validated

        if validated_actions:
            st.markdown("### Validated Actions")

            readable_rows = []
            for a in validated_actions:
                if a["type"] == "mkdir":
                    readable_rows.append(
                        {
                            "id": a["id"],
                            "type": "mkdir",
                            "details": f"Create folder '{a['rel_path']}'",
                        }
                    )
                elif a["type"] == "move":
                    readable_rows.append(
                        {
                            "id": a["id"],
                            "type": "move",
                            "details": f"Move '{a['rel_src']}' -> '{a['rel_dest']}'",
                        }
                    )
                elif a["type"] == "delete_soft":
                    readable_rows.append(
                        {
                            "id": a["id"],
                            "type": "delete_soft",
                            "details": f"Soft-delete '{a['rel_path']}' (move to trash)",
                        }
                    )
                else:
                    readable_rows.append(
                        {
                            "id": a["id"],
                            "type": a["type"],
                            "details": "Unknown action type",
                        }
                    )

            readable_df = pd.DataFrame(readable_rows)
            st.dataframe(readable_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### Execute File-Agent Plan")

            col_exec1, col_exec2 = st.columns([1, 1])
            with col_exec1:
                confirm_checkbox = st.checkbox(
                    "I understand this will modify files/folders inside the base directory.",
                    value=False,
                )
            with col_exec2:
                dry_run_mode = st.checkbox(
                    "Dry-run mode (simulate only, no actual changes)",
                    value=True,
                    help="Recommended to preview actions before actual execution.",
                )

            exec_clicked = st.button("⚙️ Execute Plan")

            if exec_clicked:
                if not confirm_checkbox and not dry_run_mode:
                    st.error(
                        "You must confirm that you understand the plan will modify files/folders "
                        "if you turn off dry-run mode."
                    )
                else:
                    with st.spinner(
                        "Executing file-agent actions (respecting dry-run mode setting)..."
                    ):
                        try:
                            exec_results = execute_file_agent_actions(
                                base_dir=summary["base_dir"],
                                actions=validated_actions,
                                dry_run=dry_run_mode,
                            )
                        except Exception as e:
                            st.error(f"Error executing file-agent actions: {format_exception(e)}")
                            exec_results = []

                        st.session_state.file_agent_last_exec_results = exec_results

                        exec_log_entry = {
                            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            "agent": "file-agent-exec",
                            "goal": file_agent_goal.strip() if file_agent_goal else "",
                            "plan_summary": plan.get("plan", ""),
                            "dry_run": dry_run_mode,
                            "exec_results": exec_results,
                        }
                        st.session_state.action_log.append(exec_log_entry)
                        append_jsonl(AGENT_LOG_FILE, exec_log_entry)

            exec_results = st.session_state.file_agent_last_exec_results
            if exec_results:
                st.markdown("#### Execution Results")
                for r in exec_results:
                    status = r.get("status", "unknown")
                    msg = r.get("message", "")
                    if status.startswith("ok"):
                        st.success(f"[{r['type']}] {msg}")
                    elif status == "pending":
                        st.info(f"[{r['type']}] {msg}")
                    else:
                        st.error(f"[{r['type']}] {msg}")

                if not dry_run_mode:
                    st.info(
                        "If you executed the plan (non-dry-run), you may click **Scan Directory** "
                        "again to see the updated structure."
                    )

    else:
        st.info(
            "Scan a directory first to see its contents and allow the file agent "
            "to plan operations."
        )

# ======================================================================================
# Tab 5: Action Logs
# ======================================================================================

with tab_logs:
    st.subheader("Agent Action Logs")

    st.markdown(
        "This section shows actions taken by the Finance Agent, File Agent, General Chat, "
        "and Single Stock Explorer.\n\n"
        "- The **session log** shows actions logged in this Streamlit session.\n"
        "- The **persistent log** reads from `agent_logs/agent_actions.jsonl` (if present)."
    )

    st.markdown("### Session Log (In-Memory)")
    if st.session_state.action_log:
        df_session_log = pd.DataFrame(st.session_state.action_log)
        st.dataframe(df_session_log, use_container_width=True)
    else:
        st.write("No agent actions logged in this session yet.")

    st.markdown("---")
    st.markdown("### Persistent Log (agent_logs/agent_actions.jsonl)")

    persistent_records = load_jsonl(AGENT_LOG_FILE, max_lines=2000)
    if persistent_records:
        df_persist = pd.DataFrame(persistent_records)
        st.dataframe(df_persist, use_container_width=True)
    else:
        st.write("No persistent logs found yet.")

    st.caption(
        "Logs are stored as JSONL for easy post-hoc analysis or debugging. "
        "You can download the `agent_logs` directory from your Colab workspace."
    )

# ======================================================================================
# Footer
# ======================================================================================

st.markdown("---")
st.info(
    "This application demonstrates a full AI Agent pipeline for CS4680 (previously 4990):\n"
    "- LLM integration module with retry + JSON/text response helpers\n"
    "- Action interpreter & executor (Finance and File System agents)\n"
    "- External data ingestion from yfinance, Reddit, and optional NewsAPI\n"
    "- GUI interface with feedback and error messages\n"
    "- Safety checks (path validation, confirmations, dry-run mode)\n"
    "- Logging of all actions for auditability\n\n"
    "Nothing here is financial advice. Use this as a learning tool."
)

