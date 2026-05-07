# 📈 UPI Transaction Trend & Failure Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Data Science](https://img.shields.io/badge/Data_Science-Analytics-2ea44f?style=for-the-badge)

## 📖 1. Project Overview
This project presents an end-to-end Data Science and Machine Learning workflow analyzing Unified Payments Interface (UPI) transaction patterns. The primary objective is to investigate the underlying causes of transaction failures, visualize high-volume traffic flows, and deploy a predictive machine learning model within an interactive Streamlit dashboard. 

## 🎯 2. Problem Statement
In digital payment ecosystems, transaction failures disrupt user experience and indicate underlying system load or partner API downtime. The goal of this analysis is to identify temporal patterns, correlate transaction volumes against failure rates, and attempt algorithmic prediction of failures utilizing user-level metadata to optimize network reliability.

## 🛠️ 3. Tech Stack
- **Languages:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest, Logistic Regression)
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Interactive Application:** Streamlit

## 📊 4. Dataset Description
- **Domain:** Synthetic standard UPI logs capturing transactional metadata.
- **Volume:** ~250,000 algorithmic records.
- **Key Features:** Hour, Day of the Week, Transaction Type, Amount (INR), Device Type, Network Type, Transaction Status (SUCCESS/FAILED).

## 🔄 5. Project Workflow
1. **Data Cleaning & Preprocessing:** Parsed strings to datetime objects, engineered temporal columns natively (Day, Hour, Weekend Boolean), mapped target variables to binary integers, and executed robust one-hot encoding on categorical text parameters.
2. **Exploratory Data Analysis (EDA):** Aggregated metrics across macro (monthly) and micro (hourly) scopes, isolating distinct peaks in volume and failure rates.
3. **Machine Learning Pipeline:** Splitting matrices via stratification, establishing a transparent baseline utilizing Logistic Regression, and deploying a robust architecture via Random Forest handling massive class imbalance.
4. **Deployment:** Built an interactive, hot-reloading Streamlit UI featuring dual-axis Plotly components and real-time algorithmic inference.

## 💡 6. Key Insights
- **Temporal Volume Peaks:** General user traffic peaks gracefully around 19:00 (7:00 PM).
- **Temporal Failure Spikes:** Absolute failure rates spike aggressively at 6:00 AM, a distinct valley in organic user traffic.
- **Volume vs. Load:** High transaction volume does *not* correlate to higher failure rates. Spikes in failure are driven by scheduled backend infrastructure maintenance and banking reconciliations, not user-driven network crashes.

## 🤖 7. Machine Learning Approach
Two models were evaluated to predict the occurrence of a failed transaction:
1. **Logistic Regression (Baseline):** Delivered a deceptive 95% accuracy by blindly predicting the majority class (Success) across all vectors, achieving a Recall and Precision of 0%.
2. **Random Forest Classifier (Advanced):** Utilized robust `class_weight='balanced'` techniques across 100 parallel trees to mathematically combat class imbalance and actively isolate true failures.

## ⚠️ 8. Limitations & Learnings
**Critical Discovery:** Algorithmic architecture isolated the reality that base transaction metadata (Time, Amount, Device, OS) serves as an extremely weak mathematical proxy for network failures. 
The implementation of explicit feature engineering (`Is_Peak_Failure_Hour`, `Is_Weekend`) and synthetic class weighting yielded negligible statistical boosts. 
- **Learning:** Structural failures are inherently external (e.g., specific bank API timeouts or vendor routing loops). Commercial-grade prediction requires deep infrastructure-level data, proving conclusively that metadata alone cannot effectively preempt a transaction failure without extreme False Positives.

## 🖥️ 9. Dashboard Features
- **Dynamic KPIs:** Real-time extraction of Total Transactions, Success Rates, and identified peak network traffic markers.
- **Interactive Filters:** Side-bar slicing across Transaction Type, Merchant Category, and State boundaries.
- **Dual-Axis Intelligence:** Unified temporal correlations comparing hourly load vs failure percentages simultaneously.
- **Algorithmic Inference Sandbox (🔮):** A live prediction engine converting synthetic user profiles into one-hot vectors passed natively into the Random Forest `.pkl` blueprint, rendering probabilistic failure risk arrays (< 3% Low Risk, > 6% High Risk).

## 🚀 10. Future Improvements
- **Data Integration:** Append external routing status variables (e.g., `Bank_Downtime_Flag`, `Vendor_Timeout_Rate`).
- **Real-Time Data Streaming:** Transition the Streamlit ingest pipeline from static CSV parsing to API polling utilizing Apache Kafka for live dashboard synchronization.

## ⚙️ 11. How to Run the Project
1. Clone the repository:
   ```bash
   git clone git@github.com:sagar-202/Upi-Trend-Analysis.git
   ```
2. Navigate to the directory and activate your virtual environment:
   ```bash
   cd Upi-Trend-Analysis
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn plotly streamlit joblib
   ```
4. Execute the web application:
   ```bash
   streamlit run dashboard/app.py
   ```

## 👨‍💻 12. Author
**Project & Analysis:** Sagar Patgar  
**Focus:** Data Science, Machine Learning Pipeline Architecture, Product Analytics
