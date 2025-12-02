# MarketWhisper: AI Financial Storyteller & Multi-Agent System

Final project for 4680, Prompt Engineering (Formally known as CS4990) – AI Agent Project

---

## Problem

Retail investors are flooded with raw charts, noisy headlines, and fragmented social media sentiment. Most tools:

- Show **price movements** but not *why* they happened.  
- Don’t connect **news, earnings, analyst opinions, and social buzz** into a coherent story.  
- Don’t leverage modern AI agents to *take actions* like comparing tickers, planning portfolios, or organizing data.

**MarketWhisper** addresses this by combining LLMs, live market data, external news/social sources, and dedicated agents to turn messy information into clear narratives and concrete actions.

---

## Objectives

- Aggregate **stock prices, company info, financial news, analyst recommendations, and social sentiment**.
- Use **prompt engineering** to generate clear, human-readable narratives and explanations.
- Implement **AI agents** that:
  - Plan tasks in JSON using LLMs.
  - Execute those tasks via Python tools (finance + file system).
- Provide an **interactive GUI** (Streamlit) for:
  - Single-stock analysis and narratives.
  - Natural-language finance tasks.
  - General-purpose chat with external context.
  - File-system organization.
- Demonstrate **error handling, safety checks, and logging** for all agent actions.

---

## High-Level Features

### 1. Single Stock Explorer (📈)
- Fetches historical price data via **yfinance**.
- Pulls **recent headlines** from yfinance.
- Optionally enriches context with:
  - **Reddit** search (public JSON endpoint).
  - **NewsAPI.org** articles (if `NEWSAPI_KEY` is set).
- Computes price changes and uses:
  - **LLM-based sentiment analysis** (bullish/bearish scores + explanation).
  - A **persona-based financial analyst** to generate a concise narrative.
- Displays:
  - Price change, sentiment metrics.
  - Market Cap, Dividend Yield, and P/E ratio (on demand).
  - Price chart for selected period.
  - Analyst recommendations and news headlines.

### 2. Finance Agent (🤖)
Natural-language finance assistant that plans and executes tasks using the LLM as a planner:

Supported action types:
- `fetch_stock` – In-depth analysis of a single ticker.
- `compare_stocks` – Multi-ticker comparison with normalized charts + LLM summary.
- `build_portfolio` – Simple equal-weight portfolio allocation with LLM explanation.
- `external_news_only` – News/earnings-focused narrative using external feeds.

The agent:
- Uses a **JSON-only planning prompt** to convert user goals into a structured `plan + actions`.
- Executes actions with Python (yfinance, external context) and returns rich visualizations and narratives.
- Can show the **raw JSON plan** for transparency.

### 3. General AI Chat (💬)
- Full-feature chat interface with:
  - Configurable **detail level** (Short / Medium / Deep dive).
  - Optional finance tools (tickers, prices, earnings).
  - Optional external feeds (Reddit + NewsAPI).
- Detects ticker symbols and, for earnings/news questions:
  - Pulls fresh market data and headlines.
  - Injects this context into the prompt so the LLM can discuss **recent events** despite training cutoffs.
- Handles **any topic** (coding, explanations, general Q&A), with guardrails against advice.

### 4. File System Agent (📂)
Local AI file organizer that operates **only inside a user-defined base directory**:

- **Summarizes** a directory (files, folders, sizes, depths).
- Uses an LLM planner to produce a JSON plan with actions:
  - `mkdir` – create folders.
  - `move` – move/rename files/folders.
  - `delete_soft` – *soft-delete* by moving items into a `.file_agent_trash` directory.
- Validates actions:
  - Ensures **relative paths** only.
  - Prevents **path traversal** outside the base directory.
  - Checks that sources exist before moving/deleting.
- Executes plan in:
  - **Dry-run mode** (simulated; no changes).
  - Real mode (with explicit user confirmation).
- Shows human-readable descriptions of each action and its result.

### 5. Action Logs (📝)
- Every agent interaction (finance, chat, file-system, single-stock) is logged:
  - In-memory session log.
  - Persistent **JSONL log**: `agent_logs/agent_actions.jsonl`.
- Logs include:
  - Timestamp, agent name, user goal, actions, and status.
- Useful for debugging, auditability, and satisfying assignment’s **logging requirement**.

---

## How This Meets 4680 Agent Requirements

**LLM Integration Module**
- Uses the **OpenAI Python SDK** (`gpt-4o-mini` variants).
- Central helper functions:
  - `call_llm_text` – structured text outputs.
  - `call_llm_json` – enforced JSON outputs via `response_format={"type": "json_object"}`.
- Implements **retry/backoff** logic for rate limits and transient errors.

**Action Interpreter / Executor**
- **Finance Agent Planner**:
  - Converts free-form goals into a JSON action plan with `type` + `params`.
  - Available actions: `fetch_stock`, `compare_stocks`, `build_portfolio`, `external_news_only`.
- **Finance Agent Executor**:
  - Reads the plan and performs concrete operations with Python (yfinance, Reddit/NewsAPI context).
- **File System Agent Planner**:
  - Takes directory summary + user goal and outputs JSON actions: `mkdir`, `move`, `delete_soft`.
- **File System Agent Executor**:
  - Validates and executes actions on disk (with soft delete and path safety).

**User Interface**
- Full **GUI via Streamlit**, with multiple tabs:
  - Single Stock Explorer.
  - Finance Agent.
  - File System Agent.
  - General Chat.
  - Action Logs.
- Users can:
  - Input natural-language requests.
  - See charts, tables, narratives, sentiments.
  - Review plans and execution results.
  - Inspect logs for transparency.

**Error Handling & Safety**
- Try/catch wrappers around:
  - OpenAI calls (with retries).
  - Network requests (Reddit, NewsAPI, yfinance).
  - File operations (permissions, missing files).
- Safety features:
  - **Path validation** using `safe_join` (no escaping base directory).
  - **Soft delete** only (no permanent deletions).
  - **Dry-run mode** + explicit confirmation for destructive operations.
  - LLM prompts that restrict tool usage and demand JSON-only output.
- Logged errors and warnings in UI and JSONL logs.

---

## Technical Overview

### Data Handling

- **yfinance**
  - Historical OHLC data per ticker and period.
  - Company info (market cap, dividend yield, P/E, etc.).
  - Built-in news feed (`Ticker.news`) for headlines.
- **Reddit**
  - Public `search.json` endpoint (no API key).
  - Fetches recent posts related to ticker/earnings.
  - Used as a proxy for **social sentiment**.
- **NewsAPI (optional)**
  - If `NEWSAPI_KEY` is provided, pulls recent articles:
    - Title, description, source, URL.

All sources are merged into a unified **external context** string that is fed into the LLM.

### Sentiment Analysis

- Primary: **LLM-based sentiment scoring** via `analyze_sentiment_with_llm`:
  - Input: combined yfinance headlines + Reddit/NewsAPI context.
  - Output: `bullish`, `bearish`, and summary text.
- Fallback: deterministic mock function (`mock_sentiment_score`) when LLM calls fail or no data is available.

### Narrative Generation

- Uses a **persona**: senior financial market analyst and storyteller.
- Includes **few-shot examples** defining:
  - Tone (balanced, factual).
  - Structure (analysis, risks, catalysts).
- The agent passes:
  - Price change.
  - Sentiment.
  - News + external context.
  - Analyst recommendation summary.
- Result: 6–10 sentence narrative explaining **what happened and why** (without giving direct financial advice).

### User Interface

- Built with **Streamlit**:
  - Inputs: text fields, select boxes, buttons.
  - Outputs: charts (`st.line_chart`), dataframes, metrics, expandable sections.
- Multi-tab layout groups functionality into logical domains:
  - Finance vs. file system vs. general chat vs. logs.

### APIs & Deployment

- OpenAI:
  - Reads `OPENAI_API_KEY` from environment.
  - Uses `client.chat.completions.create` for models like `gpt-4o-mini`.
- NewsAPI (optional):
  - Reads `NEWSAPI_KEY` from environment.
- Intended for:
  - Local/Colab development.
  - Streamlit + localtunnel or similar for sharing demos.

---

## Setup and Usage

1. Clone the Repository

```bash
git clone https://github.com/Agekyan/CS4680.git
cd CS4680
```

2. Install Dependencies

```bash
pip install streamlit yfinance openai numpy pandas requests
```

(If running in Colab, you may already have some of these preinstalled.)

3. Set Environment Variables

```bash
export OPENAI_API_KEY="your_openai_key_here"

# Optional: for external newsfeed
export NEWSAPI_KEY="your_newsapi_key_here"
```

4. Run the Application

```bash
streamlit run app.py
```

If using Google Colab:

- Use the provided %%writefile app.py pattern to write the Streamlit app to a local file in the notebook.
- Use a tool like localtunnel or ngrok from the Colab environment to expose the Streamlit app on a public URL (commands vary by tool). The repository includes example Colab snippets demonstrating how to write the file and start localtunnel.

---

## Prompt Engineering Techniques

Persona Prompts

- Finance analyst persona for narratives.
- Portfolio explainer for allocation summaries.
- File-organization planner for file agent actions.

Few-Shot Examples

- Example narratives for AAPL and TSLA define:
  - Expected structure.
  - Level of detail.
  - How to integrate price, news, sentiment, and analyst views.

JSON-Only Planner Outputs

- Finance Agent Planner:
  - Enforces JSON with plan + actions.
- File System Agent Planner:
  - Enforces JSON actions (mkdir, move, delete_soft) with relative paths.

Tool-Oriented Prompts

- The model is told exactly what actions are available and how to format them.
- This turns the LLM into a planner, not an executor.

Context Injection

- For stock and earnings questions:
  - Additional context from yfinance, Reddit, and NewsAPI is injected into the prompt.
- For chat:
  - Finance context (tickers, prices, headlines) is appended only when relevant.

---

## Roadmap / Future Work

- Integrate real-time sentiment from:
  - Full Reddit/Twitter APIs, with proper auth.
  - Order-book / volume analytics.
- Add:
  - Insider trading & institutional flow data via external APIs.
  - More advanced portfolio construction (risk models, diversification constraints).
- Extend File System Agent to:
  - Support tag-based organization.
  - Include optional summarization of documents before moving.
- Add user authentication and per-user profiles if deployed publicly.
- Deploy to a cloud host (Streamlit Cloud, Fly.io, etc.) for frictionless access.

---

## Links / Other Info

- Course: 4680, Prompt Engineering (Formally known as CS4990) – AI Agent Project
- Project Website / Live Demo: https://cs4680finalproject.streamlit.app/
- Demo Video: https://youtu.be/zWgaS4p8BRQ
- Reference Agent Video: https://youtu.be/wcS2QUXKeP4
- GitHub: https://github.com/Agekyan/CS4680
- Slides: https://docs.google.com/presentation/...
- Example / Reference code & repository: https://github.com/Agekyan/CS4680

Project Deliverables:

- GitHub repository with source code + this README.
- Google Slides presentation.
- 3–5 minute demo video walking through:
  - Problem & motivation.
  - Architecture & prompt design.
  - Live demo of each tab/agent.

(Update the Slides link above with your actual URLs as needed.)

---

## License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.

---

## Acknowledgments

- Streamlit – simple, powerful app framework.
- YFinance – convenient access to market data.
- OpenAI – GPT family of models powering narratives, planning, and chat.
- Reddit and NewsAPI – external feeds for news and sentiment context.
- CS4680 instructors and course materials for the AI Agent project framework and inspiration.
