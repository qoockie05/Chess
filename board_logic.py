def get_changed_squares(logical_board, predicted_board):
    changed = []
    for r in range(8):
        for c in range(8):
            if logical_board[r][c] != predicted_board[r][c]:
                changed.append((r, c, logical_board[r][c], predicted_board[r][c]))

    return changed


def find_differences(prev_board, new_board):
    removed = []
    added = []
    for r in range(8):
        for c in range(8):
            if prev_board[r][c] != new_board[r][c]:
                if prev_board[r][c] != "." and new_board[r][c] == ".":
                    removed.append((r, c, prev_board[r][c]))
                elif prev_board[r][c] == "." and new_board[r][c] != ".":
                    added.append((r, c, new_board[r][c]))
                else:
                    removed.append((r, c, prev_board[r][c]))
                    added.append((r, c, new_board[r][c]))

    return removed, added


def possible_pieces(board, r1, c1, r2, c2, piece, capture=False):
    candidates = []

    target = board[r2][c2]
    if target != "." and target[0] == piece[0]:
        return candidates
    if king_turn(r1, c1, r2, c2):
        candidates.append("K")
    if queen_turn(r1, c1, r2, c2) and is_path_clear(board, r1, c1, r2, c2):
        candidates.append("Q")
    if rook_turn(r1, c1, r2, c2) and is_path_clear(board, r1, c1, r2, c2):
        candidates.append("W")
    if bishop_turn(r1, c1, r2, c2) and is_path_clear(board, r1, c1, r2, c2):
        candidates.append("G")
    if knight_turn(r1, c1, r2, c2):
        candidates.append("S")
    if pawn_turn(r1, c1, r2, c2, piece, capture):
        if abs(r2-r1)==2:
            check_r = (r1+r2)//2
            if board[check_r][c1] == ".":
                candidates.append("P")
        else:
            candidates.append("P")
    return candidates


def king_turn(r1, c1, r2, c2):
    resultR = abs(r1 - r2)
    resultC = abs(c1 - c2)
    if resultR <= 1 and resultC <= 1 and not (resultR == 0 and resultC == 0):
        return True
    return False


def queen_turn(r1, c1, r2, c2):
    resultR = abs(r1 - r2)
    resultC = abs(c1 - c2)
    if resultR == resultC and resultR != 0:
        return True
    if (r1 == r2 or c1 == c2) and not (r1 == r2 and c1 == c2):
        return True
    return False


def knight_turn(r1, c1, r2, c2):
    resultR = abs(r1 - r2)
    resultC = abs(c1 - c2)
    if (resultR == 2 and resultC == 1) or (resultR == 1 and resultC == 2):
        return True
    return False


def bishop_turn(r1, c1, r2, c2):
    resultR = abs(r1 - r2)
    resultC = abs(c1 - c2)
    if resultR == resultC and resultC != 0:
        return True
    return False


def pawn_turn(r1, c1, r2, c2, piece, capture=False):
    color = piece[0]
    resultR = r2 - r1
    resultC = c2 - c1
    if color == 'w':
        if capture:
            if resultR == -1 and abs(resultC) == 1:
                return True
            return False
        if resultC != 0:
            return False
        if resultR == -1:
            return True
        if r1 == 6 and resultR == -2:
            return True
    if color == 'b':
        if capture:
            if resultR == 1 and abs(resultC) == 1:
                return True
            return False
        if resultC != 0:
            return False
        if resultR == 1:
            return True
        if r1 == 1 and resultR == 2:
            return True
    return False


def rook_turn(r1, c1, r2, c2):
    return (r1 == r2 or c1 == c2) and (r1 != r2 or c1 != c2)


def is_path_clear(board, r1, c1, r2, c2):
    resultR = r2 - r1
    resultC = c2 - c1

    counter_r =  0 if resultR == 0 else (1 if resultR > 0 else -1)
    counter_c =  0 if resultC == 0 else (1 if resultC > 0 else -1)

    current_r = r1 + counter_r
    current_c = c1 + counter_c
    while (current_r, current_c) != (r2, c2):
        if board[current_r][current_c] != ".":
            return False
        current_r += counter_r
        current_c += counter_c
    return True


def detect_castling(prev_board, removed, appeared):
    appeared_cols_by_row = {}
    for r, c, p in appeared:
        appeared_cols_by_row.setdefault(r, set()).add(c)

    removed_set = {(r, c) for r, c, _ in removed}

    for color, king_row in [("w", 7), ("b", 0)]:
        piece_k = f"{color}K"
        piece_r = f"{color}W"

        if prev_board[king_row][4] != piece_k:
            continue


        if (king_row, 4) not in removed_set:
            continue

        cols = appeared_cols_by_row.get(king_row, set())

        if (
            6 in cols
            and prev_board[king_row][7] == piece_r
            and prev_board[king_row][5] == "."
            and prev_board[king_row][6] == "."
        ):
            return (king_row, 4, king_row, 6, king_row, 7, king_row, 5, color)

        if (
            2 in cols
            and prev_board[king_row][0] == piece_r
            and prev_board[king_row][1] == "."
            and prev_board[king_row][2] == "."
            and prev_board[king_row][3] == "."
        ):
            return (king_row, 4, king_row, 2, king_row, 0, king_row, 3, color)

    return None


def is_captured(logical_board, predicted_board):
    count_logical = sum(1 for row in logical_board for piece in row if piece != ".")
    count_predicted = sum(1 for row in predicted_board for piece in row if piece != ".")
    if count_logical != count_predicted:
        return True
    return False


def pieces_count(board):
    return sum(1 for r in range(8) for c in range(8) if board[r][c] != '.')


def kings_alive(board):
    wK = any(board[r][c] == 'wK' for r in range(8) for c in range(8))
    bK = any(board[r][c] == 'bK' for r in range(8) for c in range(8))
    return wK, bK


def ask_promotion(color):
    options = {'Q': 'Hetman', 'W': 'Wieża', 'G': 'Goniec', 'S': 'Skoczek'}
    print("Pionek dotarł do końca. Wybierz figurę:")
    for key, name in options.items():
        print(f"  {key} = {name}")
    while True:
        choice = input("Wpisz literę (Q/W/G/S): ").strip().upper()
        if choice in options:
            return f"{color}{choice}"
        print("Nieprawidłowy wybór, spróbuj ponownie.")


def count_pieces_by_color(board):
    white = 0
    black = 0

    for row in board:
        for field in row:
            if field.startswith("w"):
                white += 1
            elif field.startswith("b"):
                black += 1

    return white, black


def counter_clear_spaces(board):
    counter=0
    for row in range(8):
        for col in range(8):
            if board[row][col] != ".":
                counter+=1
    return counter