import cv2
import numpy as np

img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
print(lines)

for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
def split_lines(lines):
    vertical = []
    horizontal = []

    if lines is None:
        return vertical, horizontal

    tolerance = np.deg2rad(10)

    for line in lines:
        rho, theta = line[0]
        theta = theta % np.pi

        if min(abs(theta), abs(theta - np.pi)) < tolerance:
            vertical.append((rho, theta))
        elif abs(theta - np.pi / 2) < tolerance:
            horizontal.append((rho, theta))

    return vertical, horizontal
def merge_similar_lines(lines, max_r_difference=20):
    if len(lines) == 0:
        return []

    lines = sorted(lines, key=lambda line: line[0])

    merged_lines = []
    current_r, current_theta = lines[0]

    for i in range(1, len(lines)):
        r, theta = lines[i]

        if abs(r - current_r) < max_r_difference:
            current_r = (current_r + r) / 2
            current_theta = (current_theta + theta) / 2
        else:
            merged_lines.append((current_r, current_theta))
            current_r, current_theta = r, theta

    merged_lines.append((current_r, current_theta))

    return merged_lines
def intersection(line1, line2):
    r1, theta1 = line1
    r2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([r1, r2])

    try:
        x, y = np.linalg.solve(A, b)
        return int(round(x)), int(round(y))
    except np.linalg.LinAlgError:
        return None
def get_intersections(vertical_lines, horizontal_lines):
    points = []

    for vertical_line in vertical_lines:
        for horizontal_line in horizontal_lines:
            point = intersection(vertical_line, horizontal_line)

            if point is not None:
                points.append(point)

    return points
def warp_square(img, p1, p2, p3, p4, size=64):
    src = np.array([p1, p2, p4, p3], dtype=np.float32)
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped
def sort_points_into_grid(points, rows=9, cols=9, y_tolerance=20):
    points = sorted(points, key=lambda p: p[1])

    rows_list = []
    current_row = []

    for p in points:
        if not current_row:
            current_row.append(p)
            continue

        if abs(p[1] - current_row[-1][1]) < y_tolerance:
            current_row.append(p)
        else:
            rows_list.append(sorted(current_row, key=lambda p: p[0]))
            current_row = [p]

    if current_row:
        rows_list.append(sorted(current_row, key=lambda p: p[0]))

    rows_list = rows_list[:rows]

    for i in range(len(rows_list)):
        rows_list[i] = rows_list[i][:cols]

    return rows_list
cv2.imwrite('linesDetected1.jpg', img)
import cv2
import numpy as np

# 1. Wczytaj obraz
img = cv2.imread('image4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Krawędzie
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 3. Hough
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# 4. Twoje funkcje
max_rho_difference=20
vertical, horizontal = split_lines(lines)
vertical = merge_similar_lines(vertical,max_rho_difference)
vertical = [
    (rho, theta) for rho, theta in vertical
    if min(abs(theta), abs(theta - np.pi)) < np.deg2rad(10)
]
horizontal = merge_similar_lines(horizontal, max_rho_difference)
points = get_intersections(vertical, horizontal)
grid = sort_points_into_grid(points, rows=9, cols=9, y_tolerance=20)
import os

os.makedirs("squares", exist_ok=True)

for i in range(8):
    for j in range(8):
        top_left = grid[i][j]
        top_right = grid[i][j + 1]
        bottom_left = grid[i + 1][j]
        bottom_right = grid[i + 1][j + 1]

        square = warp_square(img, top_left, top_right, bottom_left, bottom_right, size=64)

        if square is not None and square.size > 0:
            cv2.imwrite(f"squares/square_{i}_{j}.jpg", square)
        else:
            print(f"Problem z polem {i},{j}")
for row in grid:
    print(row)

# 5. Debug: narysuj linie
debug = img.copy()
#6. dodanie\
print("VERTICAL LINES:")
for idx, (rho, theta) in enumerate(vertical):
    print(idx, "rho=", rho, "theta=", theta)
debug_lines = img.copy()

for idx, (rho, theta) in enumerate(vertical):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    px = int(x0)
    py = int(y0)
    cv2.putText(debug_lines, f"V{idx}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite("debug_vertical_numbered.jpg", debug_lines)
for rho, theta in vertical:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

for rho, theta in horizontal:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

for x, y in points:
    cv2.circle(debug, (x, y), 4, (0, 0, 255), -1)

cv2.imwrite('debug_result.jpg', debug)

print('vertical:', len(vertical))
print('horizontal:', len(horizontal))
print('points:', len(points))
import os

os.makedirs("squares", exist_ok=True)

h, w = img.shape[:2]

for i in range(8):
    for j in range(8):
        x1, y1 = grid[i][j]
        x2, y2 = grid[i][j + 1]
        x3, y3 = grid[i + 1][j]
        x4, y4 = grid[i + 1][j + 1]

        left = min(x1, x2, x3, x4)
        right = max(x1, x2, x3, x4)
        top = min(y1, y2, y3, y4)
        bottom = max(y1, y2, y3, y4)

        left = max(0, left)
        right = min(w, right)
        top = max(0, top)
        bottom = min(h, bottom)

        if left < right and top < bottom:
            crop = img[top:bottom, left:right]

            if crop.size > 0:
                cv2.imwrite(f"squares/square_{i}_{j}.jpg", crop)
            else:
                print(f"Pusty crop dla pola {i},{j}")
        else:
            print(f"Niepoprawne współrzędne dla pola {i},{j}:",
                  left, right, top, bottom)
for i in range(8):
    print(
        f"pole {i},2:",
        "TL=", grid[i][2],
        "TR=", grid[i][3],
        "BL=", grid[i+1][2],
        "BR=", grid[i+1][3]
    )
    debug_points = img.copy()

    for i, row in enumerate(grid):
        for j, (x, y) in enumerate(row):
            cv2.circle(debug_points, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(debug_points, f"{i},{j}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imwrite("debug_grid_indices.jpg", debug_points)