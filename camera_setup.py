import shutil

import numpy as np
import cv2
import os
OUTPUT_DIR = "squares2"
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  #top-left
    rect[2] = pts[np.argmax(s)]  #bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  #top-right
    rect[3] = pts[np.argmax(diff)]  #bottom-left

    return rect


def draw_points_preview(img, points):
    preview = img.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(preview, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),

        )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(preview, points[i], points[i + 1], (0, 255, 0), 2)

    if len(points) == 4:
        cv2.line(preview, points[3], points[0], (0, 255, 0), 2)

    cv2.putText(preview, "Kliknij 4 rogi planszy", (20, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 1)
    cv2.putText(preview, "ENTER = zatwierdz, R = reset, C/ESC = anuluj", (20, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 1)

    return preview





def clear_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def four_point_transform_to_square(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    side = int(max(width_a, width_b, height_a, height_b))
    side = max(side, 400)

    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (side, side))
    return warped


def save_64_squares(board_img):
    clear_output_dir()
    squares_tab = []
    h, w = board_img.shape[:2]
    cell_h = h // 8
    cell_w = w // 8

    pad_x = int(cell_w * 0.05)
    pad_y = int(cell_h * 0.05)

    grid_preview = board_img.copy()
    for row in range(8):
        for col in range(8):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h if row < 7 else h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w if col < 7 else w

            rank = 8 - row
            file = chr(ord('a') + col)

            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w, x2 + pad_x)
            y2p = min(h, y2 + pad_y)

            square = board_img[y1p:y2p, x1p:x2p]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"square_{row + 1}{chr(ord('a') + col)}.jpg"), square)
            squares_tab.append(square)

            cv2.rectangle(grid_preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                grid_preview,
                f"{rank}{file}",
                (x1 + 5, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )
    return grid_preview, squares_tab