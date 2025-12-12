import torch
from torch import nn
from model import VisionTransformer
import torch.optim as optim
from torchvision import datasets, transforms
import itertools
from util import throb

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.CelebA(root='~/training_data/', target_type="landmarks", download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = VisionTransformer()
model = model.to(device)
model = torch.compile(model)
torch.set_float32_matmul_precision('high')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

useAMP = True
scaler=torch.amp.GradScaler("cuda", enabled=useAMP)

print("started training")
# Training loop
try:
    last_loss = float("inf")
    model.train()
    for epoch in itertools.count(0):  
        running_loss = 0.0
        for batch,(inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=useAMP):
                pred = model(inputs)
                loss = criterion(pred, inputs)
                
            if loss.isnan().any():
                raise RuntimeError("numerican instability")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if batch % 30 == 0:
                loss, current = loss.item(), (batch + 1) 
                print(f"loss: {loss:>7f}  [{current:>5d}] ")
            throb()

        print(f"Epoch [{epoch+1}], Loss: {running_loss/len(train_loader)}")
        if last_loss < running_loss:
            print("early stopping triggered")
            break;
        last_loss = running_loss
except KeyboardInterrupt:
    print("exiting early")
finally:
    torch.save(model.state_dict(), "model.pth")
#    torch.save(model.state_dict(), f"model-{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth")
    print("Saved PyTorch Model State to model.pth")
