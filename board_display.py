import cv2
from modelPredict import predict_square


def write_squares(start_board, board_img):
    annotated = board_img.copy()
    h, w = annotated.shape[:2]
    cell_h = h // 8
    cell_w = w // 8

    for row in range(8):
        for col in range(8):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h if row < 7 else h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w if col < 7 else w

            expected = start_board[row][col]

            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            
            if expected != ".":
                cv2.putText(
                    annotated,
                    expected,
                    (x1 + 10, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA
                )
    return annotated, start_board


def predict_move(r1, c1, r2, c2, board):
        annotated = board.copy()
        h, w = annotated.shape[:2]
        cell_h = h // 8
        cell_w = w // 8

        for row in range(8):
            for col in range(8):
                if row == r1 and col == c1:
                    y1 = row * cell_h
                    y2 = (row + 1) * cell_h if row < 7 else h
                    x1 = col * cell_w
                    x2 = (col + 1) * cell_w if col < 7 else w
                    color = (131, 107, 240)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                if row == r2 and col == c2:
                    y1 = row * cell_h
                    y2 = (row + 1) * cell_h if row < 7 else h
                    x1 = col * cell_w
                    x2 = (col + 1) * cell_w if col < 7 else w
                    color = (131, 107, 240)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
        start_x = c1 * cell_w + cell_w // 2
        start_y = r1 * cell_h + cell_h // 2
        end_x = c2 * cell_w + cell_w // 2
        end_y = r2 * cell_h + cell_h // 2

        cv2.arrowedLine(
            annotated,
            (start_x, start_y),
            (end_x, end_y),
            (23, 186, 146),
            thickness=3,
            tipLength=0.2
        )
        return annotated


def next_board(board_img, margin=0.001):
    annotated = board_img.copy()
    h, w = annotated.shape[:2]
    cell_h = h // 8
    cell_w = w // 8
    predicted_board = []

    pad_x = int(cell_w * margin)
    pad_y = int(cell_h * margin)

    for row in range(8):
        row_preds = []
        for col in range(8):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h if row < 7 else h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w if col < 7 else w

            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w, x2 + pad_x)
            y2p = min(h, y2 + pad_y)

            square = board_img[y1p:y2p, x1p:x2p]
            class_name, symbol, confidence = predict_square(square)
            row_preds.append(symbol)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if symbol != ".":
                cv2.putText(annotated, symbol, (x1 + 10, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"{confidence:.2f}", (x1 + 7, y1 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255), 1, cv2.LINE_AA)

        predicted_board.append(row_preds)

    return annotated, predicted_board


def predict_square_from_board(board_img, r, c, margin=0.001):
    h, w = board_img.shape[:2]
    cell_h = h // 8
    cell_w = w // 8

    pad_x = int(cell_w * margin)
    pad_y = int(cell_h * margin)

    y1 = r * cell_h
    y2 = (r + 1) * cell_h if r < 7 else h
    x1 = c * cell_w
    x2 = (c + 1) * cell_w if c < 7 else w

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)

    square = board_img[y1p:y2p, x1p:x2p]
    return predict_square(square)