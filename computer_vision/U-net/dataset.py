'''
--->train_images
--->train_masks (black and white images)

---> val_images
---> val_masks (black and white images)
'''

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))       # same name but_mask.gif image names
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)                        # mask pad for gray scale :"L"
        #0.0, 255.0(white)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transformations(image=image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask






































