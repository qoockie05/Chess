import torch
import torchvision.datasets
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

n_epochs = 20
batch_size_train = 32 #ile probek naraz
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
DIRECTORY=r"C:\Users\Ania\PycharmProjects\Moje_prywatne\Apka_do_triggerowania_obrazkow\pythonProject1\pionki\pioneks_ag"
DIRECTORY_TEST=r"C:\Users\Ania\PycharmProjects\Moje_prywatne\Apka_do_triggerowania_obrazkow\pythonProject1\pionki\test"
IMAGE_SIZE=64

train_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    #transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1,contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
test_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

train_dataset=datasets.ImageFolder(DIRECTORY,transform=train_transform)
test_dataset=datasets.ImageFolder(DIRECTORY_TEST,transform=test_transform)
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


print(f'Klasy {train_dataset.classes}')
print(f'Train {len(train_dataset)}')
print("TRAIN:", train_dataset.class_to_idx)
print("TEST :", test_dataset.class_to_idx)

class Net(nn.Module):
    def __init__(self, num_folders=12):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 30 * 30, num_folders)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x
network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataset) for i in range(n_epochs + 1)]
print(test_counter)
os.makedirs("results", exist_ok=True)
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_dataset)))
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')
def evaluate():
    network.eval()
    test_loss = 0
    correct = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy().flatten())

    test_loss /= len(test_dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset),
        100. * correct / len(test_dataset)))
    return all_targets,all_preds
for epoch in range(1, n_epochs + 1):
    train(epoch)
    all_targets, all_preds = evaluate()

cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=train_dataset.classes
    )

plt.figure(figsize=(10, 10))
disp.plot(cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()