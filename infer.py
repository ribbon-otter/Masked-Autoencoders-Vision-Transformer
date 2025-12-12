import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import os.path
from model import VisionTransformer
import matplotlib.pyplot as plt
import itertools

import torch.optim as optim
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_data = datasets.CelebA(root='~/training_data/', target_type="landmarks", download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

model = VisionTransformer()
model = model.to(device)
model = torch.compile(model)
torch.set_float32_matmul_precision('high')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

with torch.no_grad():
    figure = plt.figure(figsize=(8, 8))
    rows = 4
    cols = 2
    for i, (img, labels) in enumerate(itertools.islice(test_loader, rows)):
        imgs_gpu = img.to(device)
        pred = model(imgs_gpu)
        pred = pred.cpu()
        img = img.squeeze()
        pred = pred.squeeze()

        #original imageo 
        figure.add_subplot(rows, cols, i*cols+1)
        plt.title("original")
        plt.axis("off")
        plt.imshow(img.permute(1,2,0).squeeze())
        #MEAN predicted image
        figure.add_subplot(rows, cols, i*cols+2)
        plt.title(f"predicted max:{pred.max():.2}")
        plt.axis("off")
        plt.imshow(pred.permute(1,2,0).numpy())
    plt.show()
