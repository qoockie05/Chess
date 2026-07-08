import os
import cv2
from board_logic import (
    possible_pieces, detect_castling, kings_alive, pieces_count, ask_promotion
)
from board_display import write_squares, predict_move, next_board
from dataset_save import save_square_to_dataset
from minimax_engine import get_best_move
from windows import *
boardd = [
    ["bW", "bS", "bG", "bQ", "bK", "bG", "bS", "bW"],
    ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
    ["wW", "wS", "wG", "wQ", "wK", "wG", "wS", "wW"]
]
current_turn = 'w'
bot_color = 'b'
depth = 5
prev_board = None
predicted_move = None


def collect_misclassified_squares(warped, logical_board, predicted_board, moved_squares):
    for (r, c) in moved_squares:
        expected = logical_board[r][c]
        predicted = predicted_board[r][c]
        if expected != "." and expected != predicted:
            save_square_to_dataset(warped, r, c, expected)


def find_best_move_from_predictions(prev_board, predicted_board):
    candidates = []

    for r1 in range(8):
        for c1 in range(8):
            old_piece = prev_board[r1][c1]

            if old_piece == ".":
                continue

            predicted_from = predicted_board[r1][c1]


            if predicted_from == old_piece:
                continue

            for r2 in range(8):
                for c2 in range(8):
                    if r1 == r2 and c1 == c2:
                        continue

                    target_before = prev_board[r2][c2]
                    predicted_to = predicted_board[r2][c2]

                    capture = prev_board[r2][c2] != "." and prev_board[r2][c2][0] != old_piece[0]

                    candidates_for_piece = possible_pieces(prev_board, r1, c1, r2, c2, old_piece, capture)
                    if old_piece[1] not in candidates_for_piece:
                        continue

                    score = 0

                    if predicted_to == old_piece:
                        score += 3
                    elif predicted_to != "." and predicted_to[0] == old_piece[0]:
                        score += 2

                    if predicted_from == ".":
                        score += 3
                    elif predicted_from != "." and predicted_from[0] != old_piece[0]:
                        score += 1


                    if capture:
                        score += 1

                    if score > 0:
                        candidates.append((score, r1, c1, r2, c2, old_piece, predicted_to, capture))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0]


def apply_bot_move(best, prev_board, predicted_board, warped):
    r1, c1, r2, c2 = best
    predicted_move = predict_move(r1, c1, r2, c2, warped)

    moving_piece = prev_board[r1][c1]
    captured_piece = prev_board[r2][c2]
    if predicted_board[r2][c2] != moving_piece:
        save_square_to_dataset(warped, r1, c1, moving_piece)

    prev_board[r2][c2] = moving_piece
    prev_board[r1][c1] = '.'

    from minimax_engine import is_in_check
    if is_in_check(prev_board, 'w'):
        print("SZACH – białe są w szachu!")

    from_ = f"{chr(ord('a') + c1)}{8 - r1}"
    to_ = f"{chr(ord('a') + c2)}{8 - r2}"
    print(f"Bot (czarne) gra: {from_} -> {to_}")

    wK, bK = kings_alive(prev_board)
    if not wK:
        print("KONIEC GRY – wygrały czarne!")
        cv2.imshow("Best move by AI", predicted_move)
        return prev_board, predicted_move, True

    return prev_board, predicted_move, False


def apply_move_and_collect(prev_board, warped, predicted_board):

    removed = []
    appeared = []

    for r in range(8):
        for c in range(8):
            logical = prev_board[r][c]
            predicted = predicted_board[r][c]

            if logical != "." and predicted == ".":
                removed.append((r, c, logical))
            elif logical == "." and predicted != ".":
                appeared.append((r, c, predicted))
            elif (
                    logical != "."
                    and predicted != "."
                    and logical != predicted
                    and logical[0] != predicted[0]
            ):
                appeared.append((r, c, predicted))
    print(f"removed={removed}")
    print(f"appeared={appeared}")

    annotated_img, _ = write_squares(prev_board, warped)

    castling = detect_castling(prev_board, removed, appeared)
    if castling is not None:
        kr1, kc1, kr2, kc2, rr1, rc1, rr2, rc2, color = castling
        piece_k = f"{color}K"
        piece_r = f"{color}W"
        print(f"Roszada: {color}, król {chr(ord('a')+kc1)}{8-kr1} -> {chr(ord('a')+kc2)}{8-kr2}")
        updated_board = [row[:] for row in prev_board]
        updated_board[kr1][kc1] = "."
        updated_board[rr1][rc1] = "."
        updated_board[kr2][kc2] = piece_k
        updated_board[rr2][rc2] = piece_r
        if predicted_board[kr2][kc2] != piece_k:
            save_square_to_dataset(warped, kr2, kc2, piece_k)
        if predicted_board[rr2][rc2] != piece_r:
            save_square_to_dataset(warped, rr2, rc2, piece_r)
        annotated_img, _ = write_squares(updated_board, warped)
        return updated_board, annotated_img, None

    if len(removed) != 1 or len(appeared) < 1:
        return prev_board, annotated_img, "Czarny pionek nie został zmieniony lub został źle zmieniony albo biały się jeszcze nie ruszył."

    r1, c1, piece = removed[0]

    for r2, c2, _ in appeared:
        capture = prev_board[r2][c2] != "." and prev_board[r2][c2][0] != piece[0]
        candidates = possible_pieces(prev_board, r1, c1, r2, c2, piece, capture)
        if piece[1] in candidates:
            print(f"Ruch: {piece} {chr(ord('a')+c1)}{8-r1} -> {chr(ord('a')+c2)}{8-r2}, capture={capture}")
            updated_board = [row[:] for row in prev_board]
            updated_board[r1][c1] = "."
            updated_board[r2][c2] = piece  # nadpisuje zbitą figurę
            collect_misclassified_squares(warped, updated_board, predicted_board, [(r1, c1), (r2, c2)])

            annotated_img, _ = write_squares(updated_board, warped)
            if abs(pieces_count(prev_board) - pieces_count(updated_board)) > 1:
                return prev_board, annotated_img, "BŁĄD_RUCHU"
            for c in range(8):
                if updated_board[0][c] == 'wP':
                    updated_board[0][c] = ask_promotion('w')
                    break
            return updated_board, annotated_img, None

    col_from = chr(ord('a') + c1)
    err = f"Niedozwolony ruch: {piece} z {col_from}{8-r1}. Cofnij i zagraj ponownie."
    return prev_board, annotated_img, err


def process_capture(frame, calibration_points):
    global prev_board, predicted_move, current_turn, bot_color
    os.makedirs("board", exist_ok=True)

    from camera_setup import four_point_transform_to_square, save_64_squares

    if calibration_points is None:
        print("Najpierw wykonaj kalibracje klawiszem U.")
        return

    warped = four_point_transform_to_square(frame, calibration_points)
    grid_preview, squares_tab = save_64_squares(warped)
    model_img, predicted_board = next_board(warped, margin=0.001)



    if prev_board is None:
        prev_board = [row[:] for row in boardd]
        logical_img, _ = write_squares(prev_board, warped)
        predicted_move = warped.copy()
        print("Plansza zainicjalizowana. Tura: białe – wykonaj ruch i zrób zdjęcie.")

    else:

        updated_board, logical_img, error_msg = apply_move_and_collect(prev_board, warped, predicted_board)

        if error_msg == "BRAK_RUCHU" or error_msg == "BŁĄD_RUCHU":
            print(f"Nie wykryto poprawnego ruchu ({error_msg}). Spróbuj ponownie.")
            logical_img, _ = write_squares(prev_board, warped)
            predicted_move = warped.copy()

        elif error_msg:
            print("Błąd ruchu:", error_msg)
            logical_img, _ = write_squares(prev_board, warped)
            predicted_move = warped.copy()

        else:
            prev_board = updated_board
            from minimax_engine import is_in_check
            if is_in_check(prev_board, bot_color):
                print("SZACH, czarne są w szachu!")
            logical_img, _ = write_squares(prev_board, warped)


            wK, bK = kings_alive(prev_board)
            if not bK:
                print("KONIEC GRY – wygrały białe!")
                cv2.imshow(WINDOW_WARPED, warped)
                return
            if not wK:
                print("KONIEC GRY – wygrały czarne!")
                cv2.imshow(WINDOW_WARPED, warped)
                return


            print("Tura czarnych – wyliczam ruch bota...")
            #white_score = sum(PIECE_VALUES.get(prev_board[r][c][1], 0) for r in range(8) for c in range(8) if
            #                   prev_board[r][c] != '.' and prev_board[r][c][0] == 'w')
            # black_score = sum(PIECE_VALUES.get(prev_board[r][c][1], 0) for r in range(8) for c in range(8) if
            #                   prev_board[r][c] != '.' and prev_board[r][c][0] == 'b')
            # print(f"Białe: {white_score}, Czarne: {black_score}, Różnica: {white_score - black_score}")
            best = get_best_move(prev_board, color=bot_color, depth=depth)
            if best is None:
                current_depth = depth
                while not best and current_depth <= 1:
                    current_depth -= 1
                    best = get_best_move(prev_board, color=bot_color, depth=current_depth)
            if best:
                prev_board, predicted_move, game_over = apply_bot_move(
                    best, prev_board, predicted_board, warped
                )
                if game_over:
                    return
            else:
                predicted_move = warped.copy()
                if is_in_check(boardd, bot_color):
                    print(f"MAT dla {bot_color}")
                else:
                    print(f"PAT - remis")
                    return None
                print("Bot nie ma dostępnych ruchów.")

        print("Tura białych – wykonaj ruch i zrób zdjęcie.")


    cv2.imwrite("board/captured_frame.jpg", frame)
    cv2.imwrite("board/warped_board.jpg", warped)
    cv2.imwrite("board/board_grid_preview.jpg", grid_preview)
    cv2.imwrite("board/board_prediction_logical.jpg", logical_img)
    cv2.imwrite("board/board_prediction_model.jpg", model_img)
    cv2.imwrite("board/best_move.jpg", predicted_move)

    cv2.imshow("Best move by AI", predicted_move)
    cv2.imshow(WINDOW_GRID, grid_preview)
    cv2.imshow("Logical board", logical_img)
    cv2.imshow("Predictions", model_img)