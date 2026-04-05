# india-sector-intelligence
Interactive dashboard for Indian macroeconomic indicators — RBI, MOSPI, NSE data
# 🇮🇳 India Sector Intelligence Dashboard

An interactive macroeconomic intelligence dashboard built for sector-level 
analysis of the Indian economy — covering GDP, CPI, IIP, Repo Rate, INR/USD, 
and Trade Deficit from 2015–2024.

## 🔍 What This Does
- Tracks 6 key macro indicators across a 10-year window
- Compares revenue growth across 6 sectors (IT, Banking, Pharma, FMCG, Auto, Infra)
- Runs ADF stationarity tests and time series decomposition automatically
- Correlation heatmap across all macro variables
- Sector performance radar chart (latest 12-month average)
- Fully interactive — date range, indicator, and sector filters

## 🧠 Analytical Methods Used
- Augmented Dickey-Fuller (ADF) Test for stationarity
- Additive Time Series Decomposition (Trend + Seasonal + Residual)
- Pearson Correlation Matrix
- Descriptive Statistics with distributional analysis

## 🛠️ Tech Stack
`Python` `Streamlit` `Plotly` `Pandas` `NumPy` `Statsmodels` `Scikit-learn`

## 🚀 Run Locally
```bash
git clone https://github.com/yourusername/india-sector-intelligence.git
cd india-sector-intelligence
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 📁 Project Structure
india-sector-intelligence/
├── dashboard/
│   └── app.py          # Main Streamlit application
├── data/               # Data folder (extendable with live API feeds)
├── notebooks/          # EDA and modelling notebooks
├── requirements.txt
└── README.md
## 👩‍💻 Author
**Aarushi K Sharma** | Research Associate | MSc Business Statistics, VIT  
[LinkedIn](#) | aarushisharma0802@gmail.com
