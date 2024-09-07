import torch
import torch.optim as optim
import torch.nn.functional as F
from codecarbon import EmissionsTracker
from cvt_model import load_vit_model
import pandas as pd
import time

# Load the model
model = load_vit_model(pretrained=True)
print(model)

# Initialize CodeCarbon tracker
tracker_codecarbon = EmissionsTracker()
tracker_codecarbon.start()

# Training setup
model.train()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
dummy_input = torch.randn(16, 3, 224, 224)
dummy_target = torch.randint(0, 1000, (16,))

# Track time
start_time = time.time()
max_time = 10  # maximum runtime in seconds

# Training loop
while True:
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = F.cross_entropy(output, dummy_target)
    loss.backward()
    optimizer.step()

    # Check elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_time:
        break

# Stop tracker and get data
tracker_codecarbon.stop()
emissions_data_codecarbon = tracker_codecarbon.final_emission_model

# Prepare data for CSV
data_codecarbon = {
    'Metric': ['Carbon emissions (kg CO2eq)', 'Energy consumed (kWh)'],
    'Value': [
        round(emissions_data_codecarbon['emissions'], 3),
        round(emissions_data_codecarbon['energy_consumed'], 3)
    ]
}

df_codecarbon = pd.DataFrame(data_codecarbon)

# Save to CSV
try:
    df_codecarbon.to_csv('emissions.csv', index=False)
    print("Data has been written to emissions.csv")
except Exception as e:
    print(f"An error occurred while writing to CSV: {e}")
