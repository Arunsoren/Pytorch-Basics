import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import troch.optim as optim
from model import UNET
from utils import (
    load_checkpoints,
    save_checkpoints,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyper-parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32 
NUM_EPOCHS  = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH  = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL  = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) #progress bar

    for batch_idx, (data, targets) in enumerates(loop):
        data = data.to(device= DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(                  # For Training   
        [
            A.Resize(height= IMAGE_HEIGHT, width= IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean= [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,          # Divide by 255
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(                   # For Validation
        [
            A.Resize(height= IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ]
    )
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)     #out channels=3 (for multiclass for examples=3)
    loss_fn = nn.BCEWithLogitsLoss()                           #cross entropy loss (for Multiclass)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoints(checkpoint)
        #check accuracy
        check_accuracy(val_loader, model, device=DEVICE)        

        #print some examples in a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()



























