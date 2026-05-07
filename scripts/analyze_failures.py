import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'data/raw/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

df['Failure'] = (df['transaction_status'].str.upper() == 'FAILED').astype(int)

total_tx = len(df)
total_failed = df['Failure'].sum()
failure_rate = (total_failed / total_tx) * 100

print(f"Total Transactions: {total_tx}")
print(f"Total Failed Transactions: {total_failed}")
print(f"Overall Failure Rate: {failure_rate:.4f}%")

# Failure rate by hour
hour_fail = df.groupby('hour_of_day')['Failure'].agg(['sum', 'count'])
hour_fail['rate'] = (hour_fail['sum'] / hour_fail['count']) * 100
peak_fail_hour = hour_fail['rate'].idxmax()
peak_fail_hour_rate = hour_fail['rate'].max()

# Failure rate by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_fail = df.groupby('day_of_week')['Failure'].agg(['sum', 'count'])
day_fail['rate'] = (day_fail['sum'] / day_fail['count']) * 100
day_fail = day_fail.reindex(day_order)

# Failure rate by transaction type
type_fail = df.groupby('transaction type')['Failure'].agg(['sum', 'count'])
type_fail['rate'] = (type_fail['sum'] / type_fail['count']) * 100
peak_fail_type = type_fail['rate'].idxmax()
peak_fail_type_rate = type_fail['rate'].max()

print(f"Peak failure hour: {peak_fail_hour} ({peak_fail_hour_rate:.4f}%)")
print(f"Most failure-prone transaction type: {peak_fail_type} ({peak_fail_type_rate:.4f}%)")

# 1. Line chart: Failure rate by hour
plt.figure(figsize=(12, 6))
plt.plot(hour_fail.index, hour_fail['rate'], marker='o', color='#d62728', linewidth=2)
plt.title('Failure Rate by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Failure Rate (%)')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
hr_path = os.path.abspath('assets/images/failure_by_hour.png')
plt.savefig(hr_path)
plt.close()

# 2. Bar chart: Failure rate by day
plt.figure(figsize=(10, 6))
day_fail['rate'].plot(kind='bar', color='#ff7f0e')
plt.title('Failure Rate by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Failure Rate (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
day_path = os.path.abspath('assets/images/failure_by_day.png')
plt.savefig(day_path)
plt.close()

# 3. Bar chart: Failure rate by transaction type
plt.figure(figsize=(8, 6))
type_fail['rate'].sort_values(ascending=False).plot(kind='bar', color='#9467bd')
plt.title('Failure Rate by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Failure Rate (%)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
type_path = os.path.abspath('assets/images/failure_by_type.png')
plt.savefig(type_path)
plt.close()

print(f"hr_path: {hr_path}")
print(f"day_path: {day_path}")
print(f"type_path: {type_path}")
