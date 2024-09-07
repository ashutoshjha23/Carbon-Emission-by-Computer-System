import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('emissions.csv')

data.columns = [col.strip() for col in data.columns]

print("Column names:", data.columns)
print(data.head())

aggregated_data = data.groupby('run_id').agg({
    'energy_consumed': 'mean',
    'emissions': 'mean'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.bar(aggregated_data['run_id'], aggregated_data['energy_consumed'], color='b', alpha=0.6, label='Energy Consumed (kWh)')
ax1.set_xlabel('Run ID')
ax1.set_ylabel('Energy Consumed (kWh)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticklabels(aggregated_data['run_id'], rotation=45)
# Add title and legend
fig.suptitle('Average Energy Consumed by Run ID')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.show()
