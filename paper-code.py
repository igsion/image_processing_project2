import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from Losses import DiceBCELoss
import matplotlib.pyplot as plt
import os
import time

class HAD_Net(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(HAD_Net, self).__init__()
        self.enc1 = self._conv_block(in_channels, k)
        self.enc2 = self._conv_block(k, 2 * k)
        self.enc3 = self._conv_block(2 * k, 4 * k)
        self.bottleneck = self._conv_block(4 * k, 8 * k, dropout=True)
        self.up1 = nn.ConvTranspose2d(8 * k, 4 * k, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(4 * k, 2 * k, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(2 * k, k, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(8 * k, 4 * k, dropout=True)
        self.dec2 = self._conv_block(4 * k, 2 * k, dropout=True)
        self.dec3 = self._conv_block(2 * k, k, dropout=True)
        self.out = nn.Conv2d(k, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch, dropout=False):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            layers.append(nn.Dropout(p=0.5))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.enc3(F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.bottleneck(F.max_pool2d(x3, kernel_size=2, stride=2))
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x5 = self.dec1(x)
        x = self.up2(x5)
        x = torch.cat([x, x2], dim=1)
        x6 = self.dec2(x)
        x = self.up3(x6)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)
        return self.out(x)


class DriveSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.tif')
        ])
        self.mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith('.gif')
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)
        mask = (mask > 0).float()

        return image, mask


def compute_accuracy(predictions, labels, threshold=0.5):
    preds = torch.sigmoid(predictions)
    preds = (preds > threshold).float()

    correct = (preds == labels).float()
    acc = correct.sum() / correct.numel()
    return acc.item()


def compute_iou(predictions, labels, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(predictions)
    preds = (preds > threshold).float()

    intersection = (preds * labels).sum(dim=(1, 2, 3))
    union = (preds + labels - preds * labels).sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# k = 64, 300 Epochs -> 62%
# k = 128, 200 Epochs ->


learning_rate = 0.001
batch_size = 2
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

training_dataset = DriveSegmentationDataset(
    image_dir="./data/DRIVE/training/images",
    mask_dir="./data/DRIVE/training/1st_manual",
    transform=transform
)

validation_dataset = DriveSegmentationDataset(
    image_dir="./data/DRIVE/test/images",
    mask_dir="./data/DRIVE/test/mask",
    transform=transform
)

train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

model = HAD_Net(in_channels=3, out_channels=1, k=128)
model.load_state_dict(torch.load("model/model_diceBCE.pth"))
print('Model Loaded')
model.to(device)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('Starting')

start_time = time.time()

total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # unique_vals = np.unique(np.array(labels[0].cpu()))
        # print(unique_vals)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = compute_accuracy(outputs, labels)
            iou = compute_iou(outputs, labels)

        if (i + 1) % 2 == 0:
            print(
                f'epoch {epoch + 1} / {epochs}, steps {i + 1} / {total_steps}, loss = {loss.item():.4f}, accuracy = {acc:.4f}, IoU = {iou:.4f}')

print("Training Time:", time.time() - start_time)

torch.save(model.state_dict(), "model/model_diceBCE.pth")


# def show_tensor_image(img_tensor, title=None):
#     # If image is on GPU, move to CPU
#     img = img_tensor.cpu().clone()
#
#     # If batched, pick the first
#     if img.dim() == 4:
#         img = img[0]
#
#     # Undo normalization if needed (assumes [-1,1] or [0,1] range)
#     # img = img * 0.5 + 0.5
#
#     img = img.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
#     plt.imshow(img)
#     if title:
#         plt.title(title)
#     plt.axis('off')
#     plt.show()


print()
print('Validation Set:')
# Test
total_iou = 0
n_batches = 0
with torch.no_grad():
    for images, labels in validation_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        iou = compute_iou(outputs, labels)

        # show_tensor_image(images[0])
        # show_tensor_image(labels[0])
        # show_tensor_image(outputs[0])

        total_iou += iou
        n_batches += 1

    mean_iou = total_iou / n_batches
    print(f'Mean IoU on Validation Set: {mean_iou:.4f}')