import os
import cv2
from _datetime import datetime
from modelPredict import SYMBOL_TO_CLASS

DATASET_DIR = r"C:\Users\Ania\Desktop\nowy_dataset"


def save_square_to_dataset(board_img, r, c, key):
    class_name = SYMBOL_TO_CLASS.get(key)
    if class_name is None:
        print(f"Nieznany symbol: {key}")
        return

    h, w = board_img.shape[:2]
    cell_h = h // 8
    cell_w = w // 8

    pad_x = int(cell_w)
    pad_y = int(cell_h)

    y1 = r * cell_h
    y2 = (r + 1) * cell_h if r < 7 else h
    x1 = c * cell_w
    x2 = (c + 1) * cell_w if c < 7 else w

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)

    crop = board_img[y1p:y2p, x1p:x2p]

    folder_path = os.path.join(DATASET_DIR, class_name)
    os.makedirs(folder_path, exist_ok=True)

    filename = f"captured_{class_name}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
    try:
        cv2.imwrite(os.path.join(f'{folder_path}', filename), crop)
        print(f'zapisano w {folder_path}')

    except Exception as e:
        print(f"Nie udalo sie zapisac pliku {filename}")