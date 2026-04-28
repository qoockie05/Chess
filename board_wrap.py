import cv2
import numpy as np
import os
import shutil

OUTPUT_DIR = "squares2"
WINDOW_CAMERA = "Camera"
WINDOW_CALIB = "Calibration"
WINDOW_WARPED = "Warped board"
WINDOW_GRID = "64 squares preview"

clicked_points = []
frozen_frame = None
display_frame = None

calibration_points = None
capture_counter = 0


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

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
            2
        )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(preview, points[i], points[i + 1], (0, 255, 0), 2)

    if len(points) == 4:
        cv2.line(preview, points[3], points[0], (0, 255, 0), 2)

    cv2.putText(preview, "Kliknij 4 rogi planszy", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(preview, "ENTER = zatwierdz, R = reset, C/ESC = anuluj", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return preview


def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_frame, frozen_frame

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        display_frame = draw_points_preview(frozen_frame, clicked_points)
        cv2.imshow(WINDOW_CALIB, display_frame)


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

    h, w = board_img.shape[:2]
    cell_h = h // 8
    cell_w = w // 8

    grid_preview = board_img.copy()

    for row in range(8):
        for col in range(8):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h if row < 7 else h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w if col < 7 else w

            square = board_img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"square_{row}_{col}.jpg"), square)

            cv2.rectangle(grid_preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                grid_preview,
                f"{row},{col}",
                (x1 + 5, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

    return grid_preview


def run_calibration(frame):
    global clicked_points, frozen_frame, display_frame, calibration_points

    clicked_points = []
    frozen_frame = frame.copy()
    display_frame = draw_points_preview(frozen_frame, clicked_points)

    cv2.namedWindow(WINDOW_CALIB)
    cv2.setMouseCallback(WINDOW_CALIB, mouse_callback)

    while True:
        cv2.imshow(WINDOW_CALIB, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("c"):
            cv2.destroyWindow(WINDOW_CALIB)
            return False

        elif key == ord("r"):
            clicked_points = []
            display_frame = draw_points_preview(frozen_frame, clicked_points)

        elif key == 13:
            if len(clicked_points) != 4:
                print("Musisz zaznaczyc dokladnie 4 punkty.")
                continue

            calibration_points = order_points(clicked_points)
            cv2.destroyWindow(WINDOW_CALIB)
            print("Kalibracja zapisana.")
            return True


def process_capture(frame):
    global calibration_points, capture_counter

    if calibration_points is None:
        print("Najpierw wykonaj kalibracje klawiszem U.")
        return

    warped = four_point_transform_to_square(frame, calibration_points)
    grid_preview = save_64_squares(warped)

    capture_counter += 1
    cv2.imwrite(f"captured_frame_{capture_counter}.jpg", frame)
    cv2.imwrite(f"warped_board_{capture_counter}.jpg", warped)
    cv2.imwrite(f"board_grid_preview_{capture_counter}.jpg", grid_preview)

    cv2.imshow(WINDOW_WARPED, warped)
    cv2.imshow(WINDOW_GRID, grid_preview)

    print(f"Zapisano cropy do folderu {OUTPUT_DIR}.")


def draw_camera_overlay(frame):
    overlay = frame.copy()

    status = "Kalibracja: OK" if calibration_points is not None else "Kalibracja: BRAK"

    cv2.putText(
        overlay,
        "SPACJA = zrob zdjecie i zapisz cropy",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    cv2.putText(
        overlay,
        "U = nowa kalibracja | ESC = wyjscie",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    cv2.putText(
        overlay,
        status,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if calibration_points is not None else (0, 0, 255),
        2
    )

    if calibration_points is not None:
        pts = calibration_points.astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    return overlay


def main():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Nie udalo sie otworzyc kamerki.")
        return

    cv2.namedWindow(WINDOW_CAMERA)

    first_frame_done = False

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Nie udalo sie pobrac klatki.")
            break

        if not first_frame_done:
            print("Wcisnij U, aby wykonac pierwsza kalibracje.")
            first_frame_done = True

        overlay = draw_camera_overlay(frame)
        cv2.imshow(WINDOW_CAMERA, overlay)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord("u"):
            run_calibration(frame)

        elif key == 32:
            process_capture(frame)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()