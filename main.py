import torch
from torchvision import transforms,datasets
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

n_epochs = 15
batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
DIRECTORY=r"C:\Users\Ania\PycharmProjects\Moje_prywatne\Apka_do_triggerowania_obrazkow\pythonProject1\pionki\dataset"
DIRECTORY_TEST=r"C:\Users\Ania\PycharmProjects\Moje_prywatne\Apka_do_triggerowania_obrazkow\pythonProject1\pionki\test"
IMAGE_SIZE=64

train_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.1,contrast=0.1),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
test_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


train_dataset=datasets.ImageFolder(DIRECTORY,transform=train_transform)
test_dataset=datasets.ImageFolder(DIRECTORY_TEST,transform=test_transform)
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


print(f'Klasy {train_dataset.classes}')
print(f'Train {len(train_dataset)}')
print("TRAIN:", train_dataset.class_to_idx)
print("TEST :", test_dataset.class_to_idx)
print("Liczba klas w train:", len(train_dataset.classes))
print("Liczba klas w test:", len(test_dataset.classes))
if train_dataset.classes != test_dataset.classes:
    print("UWAGA: Różne klasy w zbiorach!")
class Net(nn.Module):
    def __init__(self, num_folders=14):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5) #3 kanaly wejsciowe, 8 filtrow, 5 rozmiar jedenego filtra
        self.drop = nn.Dropout2d() #nie uczy sie na pamiec
        self.pool = nn.MaxPool2d(2, 2) #wybieranie najwazniejszego kwadratu; ssplaszczanie obrazu;
        self.fc1 = nn.Linear(8 * 30 * 30, num_folders) #polaczona warstwa

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = torch.flatten(x, 1) #splaszzczenie tensora do wektora
        x = self.fc1(x) #warstwa liniowa
        x = F.log_softmax(x, dim=1)
        return x

network = Net().to(device)
optimizer = optim.AdamW(network.parameters(), lr=0.001, weight_decay=1e-4)
#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

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
        output = network(data).to(device)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_dataset)))
            torch.save(network.state_dict(), 'model/model.pth')
            torch.save(optimizer.state_dict(), 'model/optimizer.pth')
def evaluate(loader, name="Eval", epoch = 0):
    network.eval()
    total_loss = 0
    correct = 0

    all_targets = []
    all_preds = []

    counter = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = network(data)

            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

            if epoch == 20:
                idx = (pred != target).nonzero(as_tuple=True)[0].tolist()

                for id in idx:
                    save_image(data[id], f'results/{counter}_rozpoznano: {pred[id].item()}_prawidłowy: {target[id].item()}.png')
                    counter += 1

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    total_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    print(f'\n{name}: Avg. loss: {total_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n')

    return total_loss, accuracy, all_targets, all_preds

train_accuracies = []
test_accuracies = []
train_epoch_losses = []
test_epoch_losses = []

for epoch in range(1, n_epochs + 1):
    train(epoch)

    train_loss, train_acc, _, _ = evaluate(train_loader, "Train set")
    test_loss, test_acc, all_targets, all_preds = evaluate(test_loader, "Test set", epoch)

    train_epoch_losses.append(train_loss)
    test_epoch_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)


epochs = range(1, n_epochs + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracies, marker='o', label='Train accuracy')
plt.plot(epochs, test_accuracies, marker='s', label='Test accuracy')
plt.xlabel('Epoka')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy na train i test - dane rzeczywiste')
plt.xticks(list(epochs))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_epoch_losses, marker='o', label='Train loss')
plt.plot(epochs, test_epoch_losses, marker='s', label='Test loss')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.title('Loss na train i test')
plt.xticks(list(epochs))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()



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

