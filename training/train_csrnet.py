import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models.csrnet import CSRNet


class CrowdDataset(Dataset):

    def __init__(self, img_dirs, density_dirs):

        self.img_paths = []
        self.den_paths = []

        print("Preparing dataset paths...")

        for img_dir, den_dir in zip(img_dirs, density_dirs):

            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            den_files = sorted([f for f in os.listdir(den_dir) if f.endswith(".npy")])

            for img_name, den_name in zip(img_files, den_files):

                self.img_paths.append(os.path.join(img_dir, img_name))
                self.den_paths.append(os.path.join(den_dir, den_name))

        print("Total images:", len(self.img_paths))


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):

        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        density = np.load(self.den_paths[idx])

        h, w, _ = img.shape

        crop_size = 256

        # Ensure fixed size patches
        if h < crop_size or w < crop_size:

            img = cv2.resize(img, (crop_size, crop_size))
            density = cv2.resize(density, (crop_size, crop_size))

        else:

            x = np.random.randint(0, w - crop_size)
            y = np.random.randint(0, h - crop_size)

            img = img[y:y+crop_size, x:x+crop_size]
            density = density[y:y+crop_size, x:x+crop_size]

        # Random horizontal flip augmentation
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            density = np.fliplr(density).copy()

        img = img / 255.0

        img = torch.tensor(img).permute(2,0,1).float()
        density = torch.tensor(density).unsqueeze(0).float()

        return img, density


dataset = CrowdDataset(
    [
        "dataset/ShanghaiTech/part_A/train_data/images",
        "dataset/ShanghaiTech/part_B/train_data/images"
    ],
    [
        "dataset/ShanghaiTech/density_maps/part_A",
        "dataset/ShanghaiTech/density_maps/part_B"
    ]
)


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CSRNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

criterion = nn.MSELoss()


for epoch in range(50):

    total_loss = 0

    for img, density in loader:

        img = img.to(device)
        density = density.to(device)

        pred_density = model(img)

        pred_density = torch.nn.functional.interpolate(
            pred_density,
            size=density.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(pred_density, density)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss)


torch.save(model.state_dict(), "checkpoints/csrnet.pth")

print("Training complete")