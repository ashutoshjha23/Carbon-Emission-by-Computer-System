import torch
import torch.optim as optim
import torch.nn.functional as F
from codecarbon import EmissionsTracker
from cvt_model import load_vit_model

model = load_vit_model(pretrained=True)
print(model)



tracker = EmissionsTracker()
tracker.start()

model.train()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
dummy_input = torch.randn(16, 3, 224, 224)
dummy_target = torch.randint(0, 1000, (16,))


for epoch in range(5):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = F.cross_entropy(output, dummy_target)
    loss.backward()
    optimizer.step()

tracker.stop()
emissions_data = tracker.final_emission_model
print(f"Carbon emissions (kg CO2eq): {emissions_data['emissions']}")
print(f"Energy consumed (kWh): {emissions_data['energy_consumed']}")

