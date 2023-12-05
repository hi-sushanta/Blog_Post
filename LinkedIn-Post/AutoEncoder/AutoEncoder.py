import torchvision
from torchvision import transforms
from torch import optim
from torch import nn 
import torch.nn.functional as F
import torch 

# Convert To Torch Dataset.
transform = transforms.ToTensor()

# Load The Dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Encoder And Decoder Block Create.
encoder = nn.Sequential(nn.Linear(28 * 28,128),nn.ReLU(),nn.Linear(128,64))
decoder = nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128, 28 * 28))

# Define AutoEncoderClass
class AutoEncoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(AutoEncoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        x = x.view(x.size(0),-1)
        y = self.encoder(x)
        y_hat = self.decoder(y)
        return y_hat

# Initialize AutoEncoder Class and optimizer.
autenc = AutoEncoder(encoder=encoder,decoder=decoder)
autopt = optim.Adam(autenc.parameters(),lr=1e-3)

epoch = 100

# Define A Complete Training Loop To Train Your Model.
for e in range(epoch):
    total_loss = []
    for x,y in train_loader:
        autopt.zero_grad()
        y_hat = autenc(x)
        loss = F.mse_loss(y_hat,x.view(x.size(0),-1))
        total_loss.append(loss.item())
        loss.backward()
        autopt.step()
    print(f"Track Loss Every Epoch:{sum(total_loss)/len(train_loader)}")