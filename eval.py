# Imports
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import Normalize, eval_model


# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS_PATH = './checkpoints'
LINEAR_MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'linear_model.mdl')
CONV_MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'conv_model.mdl')


# Data (linear)
test_dataset = datasets.MNIST(root='./data/test', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    Normalize(squeeze_channel=True)
]))

dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=128
)

# Model
linear_checkpoint = torch.load(LINEAR_MODEL_PATH)
conv_checkpoint = torch.load(CONV_MODEL_PATH)

linear_model = linear_checkpoint['model'].to(DEVICE)
conv_model = conv_checkpoint['model'].to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)

# Eval 
print('Eval linear')
eval_model(
    dataloader=dataloader,
    model=linear_model,
    criterion=criterion 
)

# Data (conv)
test_dataset = datasets.MNIST(root='./data/test', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    Normalize(squeeze_channel=False)
]))

dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=128
)

print('Eval conv')
eval_model(
    dataloader=dataloader,
    model=conv_model,
    criterion=criterion
)
