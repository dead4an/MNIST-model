# Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import LinearModel, ConvModel
from utils import Normalize, train_epoch, save_checkpoint


# Constants
torch.manual_seed(42)
torch.cuda.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 28
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_EPOCHS = 15
BATCH_SIZE = 1024
INPUT_DIM = NUM_PIXELS
HIDDEN_DIM = 784
OUTPUT_DIM = 10
LR = 4e-3
CHECKPOINTS_PATH = './checkpoints'

# Entry point
if __name__ == '__main__':
    # Data (linear)
    train_dataset = datasets.MNIST(root='./data/train', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        Normalize(squeeze_channel=True)
    ]))

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE
    )

    # Linear model
    linear_model = LinearModel(
        input_dim=NUM_PIXELS,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    ).to(DEVICE)


    linear_optimizer = optim.Adam(params=linear_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Linear model training cycle
    print('Train linear')
    for epoch in range(NUM_EPOCHS):
        train_epoch(
            dataloader=dataloader,
            model=linear_model,
            optimizer=linear_optimizer,
            criterion=criterion,
            epoch=epoch
        )
    
    # Save model
    save_checkpoint(
        model=linear_model,
        optimizer=linear_optimizer,
        epoch=NUM_EPOCHS - 1,
        path=os.path.join(CHECKPOINTS_PATH, 'linear_model.mdl')
    )

    # Data (convolutional)
    train_dataset = datasets.MNIST(root='./data/train', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        Normalize(squeeze_channel=False)
    ]))

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE
    )

    # Conv model
    conv_model = ConvModel(
        out_channels=16,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    ).to(DEVICE)

    conv_optimizer = optim.Adam(params=conv_model.parameters(), lr=LR)

    # Convolutional model training cycle
    print('Train convolutional')
    for epoch in range(NUM_EPOCHS):
        train_epoch(
            dataloader=dataloader,
            model=conv_model,
            optimizer=conv_optimizer,
            criterion=criterion,
            epoch=epoch
        )

    # Save model
    save_checkpoint(
        model=conv_model,
        optimizer=conv_optimizer,
        epoch=NUM_EPOCHS - 1,
        path=os.path.join(CHECKPOINTS_PATH, 'conv_model.mdl')
    )
