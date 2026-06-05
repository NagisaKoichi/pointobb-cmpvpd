import argparse
import glob
import os
from typing import List, Optional, Tuple

import numpy as np


def parse_line(line: str) -> Optional[Tuple[List[float], str, List[str]]]:
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    coords = [float(v) for v in parts[:8]]
    label = parts[8]
    tail = parts[9:]
    return coords, label, tail


def parse_score(tail: List[str]) -> Optional[float]:
    if len(tail) != 1:
        return None
    try:
        return float(tail[0])
    except ValueError:
        return None

def sort_rectangle_points(points):
    """
    将构成矩形的4个点排序为 0->1->2->3 的顺序（逆时针遍历）。
    
    输入:  numpy数组, shape (4, 2)
    输出:  同 shape 的 numpy数组, 点按逆时针顺序排列
    
    原理:
    1. 计算四个点的质心（对矩形来说就是几何中心）
    2. 计算每个点相对于质心的极角 (atan2)
    3. 按极角从小到大排序 → 自然得到逆时针遍历顺序
    """
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    
    # 计算每个点相对于质心的极角
    angles = np.arctan2(points[:, 1] - centroid[1], 
                        points[:, 0] - centroid[0])
    
    # 按角度排序（逆时针）
    order = np.argsort(angles)
    
    return points[order]

def poly_to_obb(poly: List[float]) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Convert 8-point polygon to (cx, cy, w, h, theta) OBB format.
    Args:
        poly (List[float]): List of 8 floats representing the polygon vertices in order:
              [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        Tuple: A tuple (cx, cy, w, h, theta) representing the oriented bounding box. 
        always with h >= w and theta aligned with the longer edge.
    """
    # the polygon points may not follow the expected order, correct it by checking the diagonal length
    pts = np.array(poly, dtype=np.float32).reshape(4, 2)
    pts = sort_rectangle_points(pts)
    edge1 = np.linalg.norm(pts[1] - pts[0])
    edge2 = np.linalg.norm(pts[2] - pts[1])
    edge3 = np.linalg.norm(pts[3] - pts[2])
    diag = np.linalg.norm(pts[2] - pts[0])
    # if diag < edge1 or diag < edge2:
    #     raise ValueError("The provided polygon points do not form a valid rectangle")
        
    if edge1 < 1e-6 or edge2 < 1e-6:
        return None
    h = max(edge1, edge2)
    w = min(edge1, edge2)
    if edge1 >= edge2:
        theta = np.arctan2(float(pts[1, 1] - pts[0, 1]), float(pts[1, 0] - pts[0, 0]))
    else:
        # theta = np.arctan2(float(pts[3, 1] - pts[0, 1]), float(pts[3, 0] - pts[0, 0]))
        theta = np.arctan2(float(pts[2, 1] - pts[1, 1]), float(pts[2, 0] - pts[1, 0]))
    # theta = theta % (np.pi)
    # if theta <= 0:
    #     theta += np.pi
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    
    print("p2b ", theta / (np.pi) * 180)
    theta_deg = theta / (np.pi) * 180

    return cx, cy, w, h, theta, theta_deg


def obb_to_poly(cx: float, cy: float, w: float, h: float, theta: float) -> List[float]:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = w / 2.0
    dy = h / 2.0
    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy),
    ]
    poly = []
    for x, y in corners:
        # rx = x * cos_t - y * sin_t + cx
        # ry = x * sin_t + y * cos_t + cy
        rx = x * sin_t + y * cos_t + cx
        ry = -x * cos_t + y * sin_t + cy
        poly.extend([rx, ry])
    return poly


def circular_mean(angles: List[float], weights: List[float], period: float) -> float:
    if len(angles) != len(weights):
        raise ValueError("Angles and weights length mismatch")
    scale = 2.0 * np.pi / period
    sin_sum = 0.0
    cos_sum = 0.0
    for a, w in zip(angles, weights):
        sin_sum += np.sin(a * scale) * w
        cos_sum += np.cos(a * scale) * w
    mean = np.arctan2(sin_sum, cos_sum) / scale
    mean = mean % period
    if mean <= 0:
        mean += period
    return float(mean)

def ang_mean(ang1: float, ang2: float, w1: float, w2: float) -> float:
    if abs(ang1 - ang2) > np.pi / 2:
        if ang1 > ang2:
            ang2 += np.pi
        else:
            ang1 += np.pi
    # w1, w2 = 3, 2
    mean = (ang1 * w1 + ang2 * w2) / (w1 + w2)
    mean = mean % np.pi
    return mean


def fuse_pair(poly1: List[float], poly2: List[float], w1: float, w2: float) -> List[float]:
    """
    Use the theta of poly1 and the center/size of a weighted average of poly1 and poly2.
    theta: the theta-ray always align with the longer edge.
    """
    
    obb1 = poly_to_obb(poly1)

    obb2 = poly_to_obb(poly2)
    if obb1 is None or obb2 is None:
        return poly1
    cx1, cy1, bw1, bh1, th1, thd1 = obb1
    cx2, cy2, bw2, bh2, th2, thd2 = obb2
    
    # print(th1 / (np.pi) * 180)

    
    # convert to height > width format to avoid theta flipping
    # if bw1 > bh1:
    #     raise ValueError("Unexpected box with width > height in poly1")
    #     bw1, bh1 = bh1, bw1
    #     th1 = (th1 + np.pi / 2) % (np.pi)
    # if bw2 > bh2:
    #     raise ValueError("Unexpected box with width > height in poly2")
    #     bw2, bh2 = bh2, bw2
    #     th2 = (th2 + np.pi / 2) % (np.pi)
    
    total = w1 + w2
    cx = (cx1 * w1 + cx2 * w2) / total
    cy = (cy1 * w1 + cy2 * w2) / total
    bw = (bw1 * w1 + bw2 * w2) / total
    bh = (bh1 * w1 + bh2 * w2) / total
    # theta = circular_mean([th1, th2], [w1*2, w2], period=np.pi / 2)
    
    w1 *= 2
    theta = ang_mean(th1, th2, w1, w2)
    # theta = th1
    # print(theta / (np.pi) * 180)
    
    # if theta >= np.pi / 2:
    #     bw, bh = bh, bw
    # bw = bw * 0.7
    # bh = bh * 0.7
        
    return obb_to_poly(cx, cy, bw, bh, theta)


def format_line(poly: List[float], label: str, tail: List[str], fmt: str) -> str:
    coords_str = " ".join(fmt.format(v) for v in poly)
    if tail:
        return f"{coords_str} {label} {' '.join(tail)}\n"
    return f"{coords_str} {label}\n"


def fuse_files(
    file1: str,
    file2: str,
    output: str,
    w1: float,
    w2: float,
    score_mode: str,
    fmt: str,
    strict: bool,
) -> None:
    """
    files format: DOTA-style txt with lines like:
    x1 y1 x2 y2 x3 y3 x4 y4 label 0
    """
    
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # print(f"Fusing:\n  {file1}\n  {file2}\n-> {output}")
    # print(f"  lines: {len(lines1)} vs {len(lines2)}  weights: {w1}:{w2}  score_mode: {score_mode}")

    if strict and len(lines1) != len(lines2):
        raise ValueError(f"Line count mismatch: {file1} vs {file2}")

    fused_lines = []
    for i, line1 in enumerate(lines1):
        if i >= len(lines2):
            if strict:
                raise ValueError(f"Missing line {i} in {file2}")
            break
        line2 = lines2[i]
        item1 = parse_line(line1)
        item2 = parse_line(line2)
        if item1 is None or item2 is None:
            msg = f"Skipping malformed line {i} in {file1} or {file2}"
            if strict:
                raise ValueError(msg)
            print("  WARNING:", msg)
            continue
        poly1, label1, tail1 = item1
        print("pre ", poly_to_obb(poly1)[5])
        poly2, label2, tail2 = item2
        if label1 != label2 and strict:
            raise ValueError(f"Label mismatch at line {i}: {label1} vs {label2}")
        label = label1

        score1 = parse_score(tail1)
        score2 = parse_score(tail2)
        tail = tail1
        if score1 is not None and score2 is not None:
            if score_mode == "avg":
                tail = [str((score1 + score2) / 2.0)]
            elif score_mode == "max":
                tail = [str(max(score1, score2))]
            elif score_mode == "min":
                tail = [str(min(score1, score2))]
            elif score_mode == "second":
                tail = [str(score2)]
            else:
                tail = [str(int(score1))]

        # tail = ["0" for v in tail]
        
        fused_poly = fuse_pair(poly1, poly2, w1, w2)
        fused_lines.append(format_line(fused_poly, label, tail, fmt))

    with open(output, "w") as f:
        f.writelines(fused_lines)
    # print(f"  wrote {len(fused_lines)} fused lines -> {output}\n")
    
def fues_files_per_class(file1: str, file2: str, output: str, w1: float, w2: float, score_mode: str, fmt: str, strict: bool) -> None:
    return

def resolve_inputs(input1: str, input2: str) -> List[Tuple[str, str, str]]:
    if os.path.isdir(input1) and os.path.isdir(input2):
        files1 = sorted(glob.glob(os.path.join(input1, "*.txt")))
        pairs = []
        for f1 in files1:
            name = os.path.basename(f1)
            f2 = os.path.join(input2, name)
            if os.path.isfile(f2):
                pairs.append((f1, f2, name))
        return pairs
    if os.path.isfile(input1) and os.path.isfile(input2):
        name = os.path.basename(input1)
        return [(input1, input2, name)]
    raise ValueError("input1/input2 must both be files or both be directories")


def should_treat_output_as_dir(output: str, pairs: List[Tuple[str, str, str]], input1: str, input2: str) -> bool:
    if os.path.isdir(output):
        return True
    if os.path.exists(output):
        return os.path.isdir(output)
    if os.path.isdir(input1) or os.path.isdir(input2):
        return True
    if len(pairs) > 1:
        return True
    return os.path.splitext(output)[1] == ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse two aligned DOTA-style OBB txt sets")
    parser.add_argument("--input1", required=True, help="First txt file or directory")
    parser.add_argument("--input2", required=True, help="Second txt file or directory")
    parser.add_argument("--output", required=True, help="Output txt file or directory")
    parser.add_argument("--w1", type=float, default=1.0, help="Weight for input1")
    parser.add_argument("--w2", type=float, default=1.0, help="Weight for input2")
    parser.add_argument(
        "--score-mode",
        choices=["first", "second", "avg", "min", "max"],
        default="first",
        help="How to merge numeric tail scores (deprecated, now overridden to always zero)",
    )
    parser.add_argument("--fuse-method", type=str, default="per_class", help="Method to fuse boxes)")
    parser.add_argument("--float-format", default="{:.1f}")
    parser.add_argument("--strict", action="store_true", help="Fail on mismatch")
    args = parser.parse_args()

    pairs = resolve_inputs(args.input1, args.input2)
    if not pairs:
        raise ValueError("No matching files found")

    output_is_dir = should_treat_output_as_dir(args.output, pairs, args.input1, args.input2)
    if output_is_dir:
        os.makedirs(args.output, exist_ok=True)

    for f1, f2, name in pairs:
        out_path = args.output
        if output_is_dir:
            out_path = os.path.join(args.output, name)
        fuse_files(
            f1,
            f2,
            out_path,
            args.w1,
            args.w2,
            args.score_mode,
            args.float_format,
            args.strict,
        )


if __name__ == "__main__":
    # 621.8 807.6 681.0 867.0 666.7 881.3 607.5 821.8
    # points = np.array([[-621.8, 807.6], 
    #                    [-681.0, 867.0], 
    #                    [-666.7, 881.3], 
    #                    [-607.5, 821.8]])
    # points = sort_rectangle_points(points)
    # obb = poly_to_obb(points.flatten().tolist())
    # print("Original points:")
    # print(points)
    # print("Converted OBB (cx, cy, w, h, theta):")
    # print(obb)
    # recon_points = obb_to_poly(*obb)
    # print("Reconstructed points from OBB:")
    # print(np.array(recon_points).reshape(4, 2))
    
    # fused = fuse_pair(points.flatten().tolist(), points.flatten().tolist(), 1.0, 0.001)
    # print("Fused points (should be close to original):")
    # print(np.array(fused).reshape(4, 2))
    
    # print(xxx)
    main()
