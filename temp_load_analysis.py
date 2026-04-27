import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'dataset/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

# Extract date for averaging
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# 1. Hourly Transaction Volume
hourly_vol = df.groupby('hour_of_day').size()
peak_hour = hourly_vol.idxmax()
peak_vol = hourly_vol.max()
lowest_hour = hourly_vol.idxmin()
lowest_vol = hourly_vol.min()

print(f"Peak traffic hour: {peak_hour} ({peak_vol} transactions)")
print(f"Lowest traffic hour: {lowest_hour} ({lowest_vol} transactions)")

# 2. Weekend vs Weekday Volume Comparison
total_weekend_tx = len(df[df['is_weekend'] == 1])
total_weekday_tx = len(df[df['is_weekend'] == 0])

weekend_days = df[df['is_weekend'] == 1]['date'].nunique()
weekday_days = df[df['is_weekend'] == 0]['date'].nunique()

avg_weekend_daily = total_weekend_tx / weekend_days
avg_weekday_daily = total_weekday_tx / weekday_days

print(f"Total Weekend Transactions: {total_weekend_tx} (Avg: {avg_weekend_daily:.0f}/day)")
print(f"Total Weekday Transactions: {total_weekday_tx} (Avg: {avg_weekday_daily:.0f}/day)")

# 3. Correlation with Failure Rates (Checking rate at Peak Volume Hour)
df['Failure'] = (df['transaction_status'].str.upper() == 'FAILED').astype(int)
hourly_failures = df.groupby('hour_of_day')['Failure'].agg(['sum', 'count'])
hourly_failures['rate'] = (hourly_failures['sum'] / hourly_failures['count']) * 100

peak_vol_fail_rate = hourly_failures.loc[peak_hour, 'rate']
six_am_fail_rate = hourly_failures.loc[6, 'rate'] # 6 AM was the peak failure hour from before

print(f"Failure rate at peak volume hour ({peak_hour}): {peak_vol_fail_rate:.4f}%")
print(f"Failure rate at 6 AM: {six_am_fail_rate:.4f}%")

# 4. Plot Hourly Volume
plt.figure(figsize=(12, 6))
hourly_vol.plot(kind='line', marker='o', color='#2ca02c', linewidth=2)
plt.title('Hourly Transaction Volume')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Total Transactions')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
hr_vol_path = os.path.abspath('hourly_volume.png')
plt.savefig(hr_vol_path)
plt.close()

print(f"hr_vol_path: {hr_vol_path}")
