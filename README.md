# 📊 AI Financial Analyst

🚀 **AI Financial Analyst** is an **Apify Actor** that provides **detailed stock market insights** using AI-powered **fundamental analysis, technical analysis, and sentiment analysis**. 

It leverages:
- 📈 **Yahoo Finance (`yfinance`)** for real-time financial data.
- 🤖 **OpenAI GPT models** for AI-driven market insights.
- 📰 **Alpha Vantage News Sentiment API** for sentiment analysis.
- ⚙ **Apify Actor** framework for automation.

---

## 🏆 **Features**
✔ **Technical Analysis** (SMA, EMA, RSI, MACD, Bollinger Bands, Supertrend)  
✔ **Fundamental Analysis** (P/E Ratio, EBITDA, Net Income, Debt-to-Equity)  
✔ **Sentiment Analysis** (News Headlines, Market Mood)  
✔ **AI-Generated Insights** (Bullish/Bearish Signals, Risks & Opportunities)  
✔ **Markdown Reports** (Auto-saved to Apify Key-Value Store)  

---

## 🚀 **How It Works**
1️⃣ **Fetch Stock Data** – Pulls financials, technical indicators & sentiment.  
2️⃣ **Perform Analysis** – Uses AI to analyze market conditions.  
3️⃣ **Generate Report** – Saves insights in a `report.md` file.  
4️⃣ **Store in Apify** – Saves report to Apify's Key-Value Store.  
5️⃣ **Token Billing** – Charges based on GPT token usage.  

---

## 📦 **Installation & Setup**
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ai-financialanalyst.git
cd ai-financialanalyst
```

### 2️⃣ Create a Virtual Environment (Optional)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set API Keys
Create a `.env` file and add your API keys:
```ini
OPENAI_API_KEY=your-openai-key
ALPHAVANTAGE_API_KEY=your-alpha-vantage-key
```

---

## 🎯 **How to Run Locally**
```bash
apify run --input-file=input.json
```
📜 **Example `input.json`**
```json
{
    "ticker": "AAPL"
}
```

---

## 🛠 **Project Structure**
```
ai-financialanalyst/
│── src/
│   ├── main.py         # Apify Actor entry point
│   ├── tools.py        # Financial, technical, sentiment analysis tools
│   ├── models.py       # Pydantic data models
│   ├── utils.py        # Helper functions
│   ├── report.md       # Generated stock analysis report
│── .venv/              # Virtual environment (optional)
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation
│── input.json          # Example input format
│── .env                # API keys (gitignore this file)
```

---

## 📊 **Example Report Output**
A sample AI-generated **Markdown Report**:
```markdown
# 📊 AI Financial Analyst Report

## 📌 **Stock Analysis Report**
- **Ticker Symbol:** `AAPL`
- **Generated by:** Apify AI Financial Analyst

---

## 🔍 **AI-Generated Insights**
- **Momentum:** AAPL has strong bullish momentum.
- **Risks:** Overbought RSI, potential short-term pullback.
- **Financial Stability:** Strong revenue & cash flow.
- **Bullish Signals:** Revenue growth, brand loyalty.
- **Bearish Signals:** Market saturation, competition.

---

## 📈 **Technical Analysis**
| Indicator | Value |
|-----------|------|
| **SMA (50)** | 180.25 |
| **RSI** | 72.3 (Overbought) |
| **MACD** | Bullish |

---

## 📰 **Sentiment Analysis**
- **Highest Sentiment:** `Positive`
- **Recent News:**
  - [Apple reports record iPhone sales](https://example.com/news1)
  - [AAPL stock hits all-time high](https://example.com/news2)
```
---


## 🚀 **Contributing**
We welcome contributions! Feel free to:
- **Open Issues** for bug reports or feature requests.
- **Submit Pull Requests** to improve the code.

---

## 📜 **License**
This project is licensed under the **MIT License**.
