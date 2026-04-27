import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="UPI Analysis Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Styling ---
st.markdown("""
<style>
    .kpi-font {
        font-size: 20px !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("💸 UPI Transaction Trend & Failure Analysis Dashboard")
st.markdown("Explore nationwide transaction data, track failures, and understand network load metrics interactively.")

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    # Load dataset
    file_path = 'dataset/upi_transactions_2024.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Dataset not found at '{file_path}'. Make sure the CSV file is structured properly.")
        return pd.DataFrame()
    
    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['transaction_status'] = df['transaction_status'].str.upper()
    df['Failure'] = (df['transaction_status'] == 'FAILED').astype(int)
    
    return df

df = load_data()

if df.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Dashboard")

# Multiselect for Category filtering
tx_types = st.sidebar.multiselect(
    "1. Transaction Type",
    options=df["transaction type"].unique(),
    default=df["transaction type"].unique()
)

merchants = st.sidebar.multiselect(
    "2. Merchant Category",
    options=df["merchant_category"].unique(),
    default=df["merchant_category"].unique()
)

states = st.sidebar.multiselect(
    "3. Sender State",
    options=df["sender_state"].unique(),
    default=df["sender_state"].unique()
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the filters above to dive deeper into specific segments. The dashboard will automatically update.")

# Apply filters
filtered_df = df[
    (df["transaction type"].isin(tx_types)) &
    (df["merchant_category"].isin(merchants)) &
    (df["sender_state"].isin(states))
]

if filtered_df.empty:
    st.warning("No data available for the currently selected combination of filters.")
    st.stop()

# --- KPIs ---
st.markdown("### 📊 Overall Performance")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

# Calculate dynamic KPIs
total_tx = len(filtered_df)
total_failed = filtered_df['Failure'].sum()
failure_rate = (total_failed / total_tx) * 100 if total_tx > 0 else 0
success_rate = 100 - failure_rate

hourly_fail_calc = filtered_df.groupby('hour_of_day')['Failure'].mean() * 100
hourly_vol_calc = filtered_df.groupby('hour_of_day').size()

peak_fail_hour = hourly_fail_calc.idxmax() if not hourly_fail_calc.empty else "N/A"
peak_vol_hour = hourly_vol_calc.idxmax() if not hourly_vol_calc.empty else "N/A"

with kpi1:
    st.metric(label="Total Transactions", value=f"{total_tx:,}")
with kpi2:
    st.metric(label="Success Rate", value=f"{success_rate:.2f}%")
with kpi3:
    st.metric(label="Failure Rate", value=f"{failure_rate:.2f}%")
with kpi4:
    st.metric(label="Peak Traffic Hour", value=f"{peak_vol_hour}:00" if peak_vol_hour != "N/A" else "N/A")
with kpi5:
    st.metric(label="Peak Failure Hour", value=f"{peak_fail_hour}:00" if peak_fail_hour != "N/A" else "N/A")

st.markdown("---")

# --- Charts Layout: Top Row (Trends) ---
st.header("📈 Day/Month Trends")
col1, col2 = st.columns(2)

with col1:
    daily_vol = filtered_df.groupby('date').size().reset_index(name='Transactions')
    fig_daily = px.line(
        daily_vol, x='date', y='Transactions', 
        title='Daily Transaction Trend', line_shape='spline',
        color_discrete_sequence=['#1f77b4']
    )
    fig_daily.update_layout(xaxis_title="Date", yaxis_title="Total Transactions")
    st.plotly_chart(fig_daily, width="stretch")

with col2:
    monthly_vol = filtered_df.groupby('month').size().reset_index(name='Transactions')
    fig_monthly = px.bar(
        monthly_vol, x='month', y='Transactions', 
        title='Monthly Transaction Trend',
        color_discrete_sequence=['#ff7f0e']
    )
    # Lock the x-axis specifically to categorical months instead of continuous float
    fig_monthly.update_xaxes(type='category', title_text="Month (1-12)")
    fig_monthly.update_layout(yaxis_title="Total Transactions")
    st.plotly_chart(fig_monthly, width="stretch")


# --- Charts Layout: Bottom Row (Hourly Dual-Axis Flow) ---
st.header("⏰ Hourly Load & Failure Distribution (Dual-Axis)")

hourly_vol = filtered_df.groupby('hour_of_day').size().reset_index(name='Volume')
hourly_fail = filtered_df.groupby('hour_of_day')['Failure'].agg(['sum', 'count']).reset_index()
hourly_fail['Failure Rate (%)'] = hourly_fail.apply(lambda row: (row['sum'] / row['count'] * 100) if row['count'] > 0 else 0, axis=1)

# Create figure with secondary y-axis
fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig_dual.add_trace(
    go.Scatter(x=hourly_vol['hour_of_day'], y=hourly_vol['Volume'], name="Transaction Volume", marker=dict(color="#2ca02c"), line=dict(color="#2ca02c")),
    secondary_y=False,
)

fig_dual.add_trace(
    go.Scatter(x=hourly_fail['hour_of_day'], y=hourly_fail['Failure Rate (%)'], name="Failure Rate (%)", marker=dict(color="#d62728"), line=dict(color="#d62728", dash='dot')),
    secondary_y=True,
)

# Add figure title
fig_dual.update_layout(title_text="Hourly Transaction Volume vs. Failure Rate Correlation")

# Set x-axis title
fig_dual.update_xaxes(tick0=0, dtick=1, title_text="Hour of Day (0-23)")

# Set y-axes titles
fig_dual.update_yaxes(title_text="<b>Transaction Volume</b> (Total)", secondary_y=False, color="#2ca02c")
fig_dual.update_yaxes(title_text="<b>Failure Rate</b> (%)", secondary_y=True, color="#d62728")

if peak_vol_hour != "N/A":
    peak_vol_y = hourly_vol[hourly_vol['hour_of_day'] == peak_vol_hour]['Volume'].values[0]
    fig_dual.add_annotation(
        x=peak_vol_hour, y=peak_vol_y, yref="y", xref="x",
        text=f"<b>Peak Volume: {peak_vol_y:,}</b>", showarrow=True, 
        arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#2ca02c", 
        font=dict(color="#2ca02c", size=14), ay=-50
    )

if peak_fail_hour != "N/A":
    peak_fail_y = hourly_fail[hourly_fail['hour_of_day'] == peak_fail_hour]['Failure Rate (%)'].values[0]
    fig_dual.add_annotation(
        x=peak_fail_hour, y=peak_fail_y, yref="y2", xref="x",
        text=f"<b>Peak Failures: {peak_fail_y:.1f}%</b>", showarrow=True, 
        arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#d62728", 
        font=dict(color="#d62728", size=14), ay=50
    )

st.plotly_chart(fig_dual, width="stretch")

st.markdown("---")

# --- Correlation Insight Section ---
st.header("🔗 Correlation Insight: Volume vs. Failures")
st.info(f"**Do higher transaction volumes lead to higher failure rates?**  \n\n**No.** When comparing the hourly volume and failure rate charts above, we see that the highest traffic volume occurs at **{peak_vol_hour}:00** with a relatively stable failure rate. Conversely, the peak failure rate happens at **{peak_fail_hour}:00**, which is a distinct period of significantly lower traffic. This indicates that failures are driven by scheduled background processes or vendor downtimes, rather than being bottlenecked by an overload of user traffic.")

st.markdown("---")

# --- Insights Section ---
st.header("💡 Key Findings & Insights")

# Peak hours were calculated above in the KPI section, so we just use them here.

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    st.error(f"""
    #### 📉 Traffic vs. Failures
    
    The highest volume concentrates around **{peak_vol_hour}:00**, implying high standard load. 
    
    However, failures spike disproportionately at **{peak_fail_hour}:00**. This suggests **backend server outages** or **vendor offline events** rather than a user-driven crash.
    """)

with col_in2:
    st.success(f"""
    #### 📅 Stable Base Routine
    
    Through day-to-day and month-to-month views, the network processes a consistently resilient traffic flow. 
    
    There is **minimal volatility** between typical weekends and weekdays, establishing a highly predictable usage baseline.
    """)

with col_in3:
    st.warning(f"""
    #### 🛠️ Recommendation Pipeline
    
    Target batch reconciliations or partner updates executing near **{peak_fail_hour}:00**. 
    
    Establishing an **asynchronous request queue** during these windows could safely elevate the current **{success_rate:.1f}%** Success Rate.
    """)

st.markdown("---")
# --- ML Prediction Section ---
st.header("🔮 Predict Transaction Failure")
st.markdown("Use our trained Random Forest algorithm to test hypothetical transaction profiles.")

with st.container():
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        pred_hour = st.slider("Hour of Day", 0, 23, 14)
        pred_day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    with col_p2:
        pred_type = st.selectbox("Transaction Type", ['P2P', 'P2M', 'Recharge', 'Bill Payment'])
        pred_amt = st.number_input("Amount (INR)", min_value=1, value=500, step=100)
    with col_p3:
        pred_device = st.selectbox("Device Type", ['Android', 'iOS', 'Web'])
        pred_network = st.selectbox("Network Type", ['4G', '5G', 'WiFi', '3G'])

    if st.button("Trigger Live Prediction", type="primary"):
        try:
            import joblib
            rf_model = joblib.load('failure_model.pkl')
            model_cols = joblib.load('model_columns.pkl')
            
            # Reconstruct the one-hot array safely
            input_dict = {col: 0 for col in model_cols}
            
            # Map Continuous
            if 'hour_of_day' in input_dict: input_dict['hour_of_day'] = pred_hour
            if 'amount (INR)' in input_dict: input_dict['amount (INR)'] = pred_amt
            
            # Map Categorical
            if f'day_of_week_{pred_day}' in input_dict: input_dict[f'day_of_week_{pred_day}'] = 1
            if f'transaction type_{pred_type}' in input_dict: input_dict[f'transaction type_{pred_type}'] = 1
            if f'device_type_{pred_device}' in input_dict: input_dict[f'device_type_{pred_device}'] = 1
            if f'network_type_{pred_network}' in input_dict: input_dict[f'network_type_{pred_network}'] = 1
            
            input_df = pd.DataFrame([input_dict])
            # Extract probability for Failure (class 1)
            failure_prob = rf_model.predict_proba(input_df)[0][1] * 100
            
            st.markdown(f"### Predicted Failure Probability: **{failure_prob:.2f}%**")
            
            if failure_prob < 3.0:
                st.success("✅ **Low Risk** (< 3%): This transaction perfectly matches successful patterns.")
            elif 3.0 <= failure_prob <= 6.0:
                st.warning("⚠️ **Medium Risk** (3% - 6%): The ecosystem is showing localized stress, proceed with caution.")
            else:
                st.error("🚨 **High Risk** (> 6%): Network metrics or schedules heavily parallel known outage vectors. Recommend queuing.")
                
        except FileNotFoundError:
            st.error("Model dependency error: Base model (.pkl) files not found on disk.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey; padding: 10px;'>Project: UPI Transaction Trend Analysis | Developed by Sagar Patgar</div>", unsafe_allow_html=True)
