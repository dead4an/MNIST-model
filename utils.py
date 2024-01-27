# Imports
import torch
from torch.utils.data import DataLoader
from time import time


# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
class Normalize():
    def __init__(self, squeeze_channel: bool=True) -> None:
        self.squeeze_channel = squeeze_channel

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image /= 255
        if not self.squeeze_channel:
            return image
        
        return image.squeeze(0)


# Functions
def train_epoch(dataloader: DataLoader, model: callable, optimizer: any, 
                criterion: callable, epoch: int) -> None:
    total_loss = 0
    start_time = time()

    for features, targets in dataloader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss
    
    # Print epoch info
    print(f'Epoch: {epoch} | Loss: {(total_loss / len(dataloader)):.6f}', 
          f'| Epoch Time: {(time() - start_time):.4f}',
          f'| Batch Time: {((time() - start_time) / len(dataloader)):.4f}')

def save_checkpoint(model: callable, optimizer: callable, epoch: int, 
                    path: str) -> None:
    torch.save({
        'model': model,
        'optim': optimizer,
        'epoch': epoch
    }, path)
    
