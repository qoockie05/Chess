import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

IMAGE_SIZE = 64
MODEL_PATH = "lastmodel_21.06.2026/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLASS_TO_SYMBOL = {
    "czarny_goniec": "bG",
    "czarny_pusto": ".",
    "czarny_krol": "bK",
    "czarny_skoczek": "bS",
    "czarny_pionek": "bP",
    "czarny_krolowa": "bQ",
    "czarny_wieza": "bW",
    "bialy_goniec": "wG",
    "bialy_pusto": ".",
    "bialy_krol": "wK",
    "bialy_skoczek": "wS",
    "bialy_pionek": "wP",
    "bialy_krolowa": "wQ",
    "bialy_wieza": "wW",
}
SYMBOL_TO_CLASS = {v: k for k, v in CLASS_TO_SYMBOL.items()}

import json

with open("modeltest/class_to_idx.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

class Net(nn.Module):
    def __init__(self, num_folders=14):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5) # 3 kanaly wejsciowe, 8 filtrow, 5 rozmiar jedenego filtra
        self.drop = nn.Dropout2d() # nie uczy sie na pamiec
        self.pool = nn.MaxPool2d(2, 2) # wybieranie najwazniejszego kwadratu; ssplaszczanie obrazu;
        self.fc1 = nn.Linear(8 * 30 * 30, num_folders) # polaczona warstwa

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = torch.flatten(x, 1) # splaszzczenie tensora do wektora
        x = self.fc1(x) # warstwa liniowa
        x = F.log_softmax(x, dim=1)
        return x

model = Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict_square(square_bgr):
    square_rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(square_rgb)
    x = test_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.exp(out)
        confidence, pred_idx = probs.max(dim=1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()

    class_name = idx_to_class[pred_idx]
    symbol = CLASS_TO_SYMBOL[class_name]

    if confidence < 0.2:
        return class_name, ".", confidence

    return class_name, symbol, confidence

