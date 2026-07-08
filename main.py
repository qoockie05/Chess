from board_wrap import process_capture
from camera_setup import *
from board_state import process_capture
from windows import *


clicked_points = []
frozen_frame = None
display_frame = None

calibration_points = None
capture_counter = 0

def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_frame, frozen_frame

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        display_frame = draw_points_preview(frozen_frame, clicked_points)
        cv2.imshow("Calibration", display_frame)

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


def draw_camera_overlay(frame):
    overlay = frame.copy()

    status = "Kalibracja: OK" if calibration_points is not None else "Kalibracja: BRAK"

    cv2.putText(
        overlay,
        "SPACJA = zrob zdjecie i zapisz cropy",
        (20, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    cv2.putText(
        overlay,
        "U = nowa kalibracja | ESC = wyjscie",
        (20, 60),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    cv2.putText(
        overlay,
        status,
        (20, 90),
        cv2.FONT_HERSHEY_COMPLEX,
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
                cv2.FONT_HERSHEY_COMPLEX,
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

        elif key == (ord("u") or ord("U")):
            run_calibration(frame)

        elif key == 32:
            process_capture(frame, calibration_points)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()