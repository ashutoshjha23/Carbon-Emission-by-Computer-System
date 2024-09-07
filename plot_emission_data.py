import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
data = pd.read_csv('emissions.csv')

# Strip any extra spaces from column names
data.columns = [col.strip() for col in data.columns]

# Print column names to check
print("Column names:", data.columns)

# Print the first few rows to verify the data
print(data.head())

# Aggregate data by 'run_id'
aggregated_data = data.groupby('run_id').agg({
    'energy_consumed': 'mean',
    'emissions': 'mean'
}).reset_index()

# Plotting the bar graph for energy consumed
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar plot for energy consumed
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
