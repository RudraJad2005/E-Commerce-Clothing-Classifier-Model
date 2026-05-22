import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt

# Load datasets
from torchvision import datasets
import torchvision.transforms as transforms

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())



class ECommerce(nn.Module):
    def __init__(self, num_classes=10, num_d_c=5):
        super().__init__()

        self.relu = nn.ReLU()

        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(64*7*7, num_classes)

    def forward(self, x):
        x = self.image_layer(x)
        x = self.classifier(x)

        return x



train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])
dataset_train = datasets.FashionMNIST(
    root = './data',
    train=True,
    transform=test_transform
)

dataset_test = datasets.FashionMNIST(
    root='./data',
    train=False,
    transform=test_transform
)

dataloader_train = DataLoader(
    train_data,
    shuffle=True,
    batch_size=16,
)

dataloader_test = DataLoader(
    dataset_test,
    shuffle=False,
    batch_size=16
)

model = ECommerce(num_classes=10)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1
model.train()

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in dataloader_train:

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    average_loss = running_loss / len(dataloader_train)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")


    model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for images,labels in dataloader_test:
        outputs = model(images)
        _, predicted_classes = torch.max(outputs, 1)
        predictions.extend(predicted_classes.tolist())
        true_labels.extend(labels.tolist())

pred_tensor = torch.tensor(predictions)
labels_tensor = torch.tensor(true_labels)

acc_calc = Accuracy(task="multiclass", num_classes=10)
prec_calc = Precision(task="multiclass", num_classes=10, average=None)
rec_calc = Recall(task="multiclass", num_classes=10, average=None)

accuracy = acc_calc(pred_tensor, labels_tensor).item()
precision = prec_calc(pred_tensor, labels_tensor).tolist()
recall = rec_calc(pred_tensor, labels_tensor).tolist()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x = np.arange(len(class_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, precision, width, label='Precision', color='skyblue')
rects2 = ax.bar(x + width/2, recall, width, label='Recall', color='salmon')

ax.set_ylabel('Score (0.0 to 1.0)')
ax.set_title('Model Performance: Precision vs. Recall per Class')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right") 
ax.legend()

plt.tight_layout()
plt.show()

