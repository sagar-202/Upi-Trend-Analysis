import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'data/raw/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['month'] = df['timestamp'].dt.month

daily_counts = df.groupby('date').size()
monthly_counts = df.groupby('month').size()

highest_day = daily_counts.idxmax()
highest_day_count = daily_counts.max()

highest_month = monthly_counts.idxmax()
highest_month_count = monthly_counts.max()

print(f"Highest transaction day: {highest_day} with {highest_day_count} transactions")
print(f"Highest transaction month: {highest_month} with {highest_month_count} transactions")

# Daily plot
plt.figure(figsize=(14, 6))
daily_counts.plot(kind='line', color='#1f77b4')
plt.title('Daily Transaction Volume Trend')
plt.xlabel('Date')
plt.ylabel('Total Transactions')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
daily_path = os.path.abspath('assets/images/daily_trend.png')
plt.savefig(daily_path)
plt.close()

# Monthly plot
plt.figure(figsize=(10, 6))
monthly_counts.sort_index().plot(kind='bar', color='#ff7f0e')
plt.title('Monthly Transaction Volume Trend')
plt.xlabel('Month (1-12)')
plt.ylabel('Total Transactions')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
monthly_path = os.path.abspath('assets/images/monthly_trend.png')
plt.savefig(monthly_path)
plt.close()

print(f"Daily_Path: {daily_path}")
print(f"Monthly_Path: {monthly_path}")
