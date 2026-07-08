import random

PIECE_VALUES = {
    "P": 1,
    "S": 3,
    "G": 3,
    "W": 5,
    "Q": 9,
    "K": 9999
}
last_bot_move = None
def evaluate_board(board):
    score = 0
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == '.':
                continue
            value = PIECE_VALUES.get(piece[1], 0)

            if piece[0] == 'w':
                score += value
            else:
                score -= value
    return score
def is_in_check(board, color):
    king_pos = None
    for r in range(8):
        for c in range(8):
            if board[r][c] == f"{color}K":
                king_pos = (r, c)
                break
    if king_pos is None:
        return True
    opponent = 'b' if color == 'w' else 'w'
    opponent_moves = generate_moves(board, opponent)
    return any(r2 == king_pos[0] and c2 == king_pos[1]
               for _, _, r2, c2 in opponent_moves)

def generate_moves(board, color):
    moves = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == '.' or piece[0] != color:
                continue
            piece_type = piece[1]
            if piece_type == 'P':
                moves += get_pawn_moves(board, row, col, piece)
            elif piece_type == 'G':
                moves += get_sliding_moves(board, row, col, piece, [(1,1),(1,-1),(-1,1),(-1,-1)])
            elif piece_type == 'W':
                moves += get_sliding_moves(board, row, col, piece, [(1,0),(-1,0),(0,-1),(0,1)])
            elif piece_type == 'Q':
                moves += get_sliding_moves(board, row, col, piece, [(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,-1),(0,1)])
            elif piece_type == 'S':
                moves += get_knight_moves(board, row, col, piece)
            elif piece_type == 'K':
                moves += get_king_moves(board, row, col, piece)
    return moves
def generate_legal_moves(board, color):
    moves = generate_moves(board, color)
    return [m for m in moves if not is_in_check(apply_move(board, m), color)]

def get_pawn_moves(board, r, c, piece):
    moves = []
    color = piece[0]
    direction = -1 if color == 'w' else 1
    start_position = 6 if color == 'w' else 1

    ruchDoPrzodu = r + direction
    if 0 <= ruchDoPrzodu < 8 and board[ruchDoPrzodu][c] == '.':
        moves.append((r, c, ruchDoPrzodu, c))
        if r == start_position and board[r + (2 * direction)][c] == '.':
            moves.append((r, c, r + (2 * direction), c))
    for dirCol in [1, -1]:
        ruchDoPrzodu = r + direction
        ruchwBok = c + dirCol
        if 0 <= ruchDoPrzodu < 8 and 0 <= ruchwBok < 8:
            target = board[ruchDoPrzodu][ruchwBok]
            if target != '.' and target[0] != color:
                moves.append((r, c, ruchDoPrzodu, ruchwBok))
    return moves

def get_knight_moves(board, r, c, piece):
    moves = []
    color = piece[0]
    for dirRow, dirCol in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        ruchDoPrzodu, ruchwBok = r + dirRow, c + dirCol
        if 0 <= ruchDoPrzodu < 8 and 0 <= ruchwBok < 8:
            target = board[ruchDoPrzodu][ruchwBok]
            if target == '.' or target[0] != color:
                moves.append((r, c, ruchDoPrzodu, ruchwBok))
    return moves

def get_sliding_moves(board, r, c, piece, directions):
    moves = []
    color = piece[0]
    for dirRow, dirCol in directions:
        ruchDoPrzodu, ruchwBok = r + dirRow, c + dirCol
        while 0 <= ruchDoPrzodu < 8 and 0 <= ruchwBok < 8:
            target = board[ruchDoPrzodu][ruchwBok]
            if target == '.':
                moves.append((r, c, ruchDoPrzodu, ruchwBok))
            elif target[0] != color:
                moves.append((r, c, ruchDoPrzodu, ruchwBok))
                break
            else:
                break
            ruchDoPrzodu += dirRow
            ruchwBok += dirCol
    return moves

def get_king_moves(board, r, c, piece):
    moves = []
    color = piece[0]
    for dirRow in [-1, 0, 1]:
        for dirCol in [-1, 0, 1]:
            if dirRow == 0 and dirCol == 0:
                continue
            ruchDoPrzodu = r + dirRow
            ruchwBok = c + dirCol
            if 0 <= ruchDoPrzodu < 8 and 0 <= ruchwBok < 8:
                target = board[ruchDoPrzodu][ruchwBok]
                if target == '.' or target[0] != color:
                    moves.append((r, c, ruchDoPrzodu, ruchwBok))
    return moves

def apply_move(board, move):
    r1, c1, r2, c2 = move
    new_board = [row[:] for row in board]
    piece = new_board[r1][c1]
    new_board[r2][c2] = new_board[r1][c1]
    new_board[r1][c1] = '.'
    #promocja pionka
    if piece == 'wP' and r2 == 0:
        new_board[r2][c2] = 'wQ'
    elif piece == 'bP' and r2 == 7:
        new_board[r2][c2] = 'bQ'
    return new_board

def minimax(board, depth, alpha, beta, maximizing_white, bot_color):
    if depth == 0:
        return evaluate_board(board), None

    color = 'w' if maximizing_white else 'b'
    moves = generate_legal_moves(board, color)

    if not moves:
        if is_in_check(board, color):
            return (float('-inf'), None) if maximizing_white else (float('inf'), None) #mat
        else:
            return 0, None #pat

    best_move = None

    if maximizing_white:
        max_score = float('-inf')
        for move in moves:
            new_board = apply_move(board, move)
            score, _ = minimax(new_board, depth - 1, alpha, beta, False, bot_color)
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return max_score, best_move
    else:
        min_score = float('inf')
        for move in moves:
            new_board = apply_move(board, move)
            score, _ = minimax(new_board, depth - 1, alpha, beta, True, bot_color)
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score, best_move



def get_best_move(board, color, depth=3):
    global last_bot_move
    maximizing = (color == 'w')
    score, move = minimax(board, depth, float('-inf'), float('inf'), maximizing, color)

    #unikam cofania poprzedniego ruchu
    if move and last_bot_move:
        r1, c1, r2, c2 = move
        pr1, pc1, pr2, pc2 = last_bot_move
        if r1 == pr2 and c1 == pc2 and r2 == pr1 and c2 == pc1:
            moves = generate_legal_moves(board, color)
            moves = [m for m in moves if m != move]
            if moves:
                if color == 'w':
                    move = max(moves, key=lambda m: minimax(
                        apply_move(board, m), depth - 1,
                        float('-inf'), float('inf'),
                        not maximizing, color)[0])
                else:
                    move = min(moves, key=lambda m: minimax(
                    apply_move(board, m), depth-1,
                    float('-inf'), float('inf'),
                    not maximizing, color)[0])

    last_bot_move = move

    if move:
        r1, c1, r2, c2 = move
        from_ = f"{chr(ord('a') + c1)}{8 - r1}"
        to_ = f"{chr(ord('a') + c2)}{8 - r2}"
        print(f"Najlepszy ruch dla {color}: z {from_} do {to_}, wynik: {score}")


    return move

