import argparse
import json
import os
from collections import defaultdict

try:
	from mmrotate.datasets.dota import DOTADataset

	DOTA_CLASSES = DOTADataset.CLASSES
except Exception:
	DOTA_CLASSES = (
		"plane",
		"baseball-diamond",
		"bridge",
		"ground-track-field",
		"small-vehicle",
		"large-vehicle",
		"ship",
		"tennis-court",
		"basketball-court",
		"storage-tank",
		"soccer-ball-field",
		"roundabout",
		"harbor",
		"swimming-pool",
		"helicopter",
		"container-crane",
		"airport",
		"helipad",
	)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Merge two pseudo-label sets by selecting the box with higher IoU to GT for each GT instance."
	)
	parser.add_argument("--gt-dir", required=True, help="Directory containing DOTA GT txt files.")
	parser.add_argument("--pseudo-dir-a", required=True, help="First pseudo label directory.")
	parser.add_argument("--pseudo-dir-b", required=True, help="Second pseudo label directory.")
	parser.add_argument(
		"--output-dir",
		required=True,
		help="Output directory. Merged labels will be written to <output-dir>/pseudo_labels.",
	)
	parser.add_argument(
		"--fallback-to-gt",
		action="store_true",
		default=True,
		help="Fallback to GT box when neither pseudo set provides a match for a GT instance.",
	)
	parser.add_argument(
		"--no-fallback-to-gt",
		dest="fallback_to_gt",
		action="store_false",
		help="Do not write a GT fallback box when both pseudo sets miss a GT instance.",
	)
	parser.add_argument(
		"--stats-name",
		default="merge_stats.json",
		help="Name of the JSON file to store statistics in the output directory.",
	)
	return parser.parse_args()


def _signed_area(points):
	area = 0.0
	for index in range(len(points)):
		x1, y1 = points[index]
		x2, y2 = points[(index + 1) % len(points)]
		area += x1 * y2 - x2 * y1
	return area * 0.5


def _normalize_polygon(points):
	if len(points) < 3:
		return []
	normalized = [(float(x), float(y)) for x, y in points]
	if _signed_area(normalized) < 0:
		normalized.reverse()
	return normalized


def _polygon_area(points):
	if len(points) < 3:
		return 0.0
	return abs(_signed_area(points))


def _inside(point, edge_start, edge_end):
	return (
		(edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
		- (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
	) >= 0.0


def _intersection(point_a, point_b, edge_start, edge_end):
	x1, y1 = point_a
	x2, y2 = point_b
	x3, y3 = edge_start
	x4, y4 = edge_end

	denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
	if abs(denominator) < 1e-12:
		return point_b

	determinant_ab = x1 * y2 - y1 * x2
	determinant_cd = x3 * y4 - y3 * x4
	x = (determinant_ab * (x3 - x4) - (x1 - x2) * determinant_cd) / denominator
	y = (determinant_ab * (y3 - y4) - (y1 - y2) * determinant_cd) / denominator
	return (x, y)


def _polygon_clip(subject_polygon, clip_polygon):
	output_list = list(subject_polygon)
	if len(output_list) < 3:
		return []

	clip_polygon = _normalize_polygon(clip_polygon)
	if len(clip_polygon) < 3:
		return []

	for clip_index in range(len(clip_polygon)):
		input_list = output_list
		output_list = []
		if not input_list:
			break

		edge_start = clip_polygon[clip_index]
		edge_end = clip_polygon[(clip_index + 1) % len(clip_polygon)]
		start_point = input_list[-1]

		for end_point in input_list:
			end_inside = _inside(end_point, edge_start, edge_end)
			start_inside = _inside(start_point, edge_start, edge_end)

			if end_inside:
				if not start_inside:
					output_list.append(_intersection(start_point, end_point, edge_start, edge_end))
				output_list.append(end_point)
			elif start_inside:
				output_list.append(_intersection(start_point, end_point, edge_start, edge_end))

			start_point = end_point

	return output_list


def polygon_iou(poly_a, poly_b):
	try:
		points_a = list(zip(poly_a[0::2], poly_a[1::2]))
		points_b = list(zip(poly_b[0::2], poly_b[1::2]))
	except Exception:
		return 0.0

	polygon_a = _normalize_polygon(points_a)
	polygon_b = _normalize_polygon(points_b)
	if len(polygon_a) < 3 or len(polygon_b) < 3:
		return 0.0

	inter_polygon = _polygon_clip(polygon_a, polygon_b)
	inter_area = _polygon_area(inter_polygon)
	area_a = _polygon_area(polygon_a)
	area_b = _polygon_area(polygon_b)
	union = area_a + area_b - inter_area
	if union <= 0:
		return 0.0
	return float(inter_area / union)


def format_pseudo_line(poly, cls_name, score):
	coords = " ".join(f"{value:.1f}" for value in poly)
	return f"{coords} {cls_name} {score:.6f}"


def read_gt_file(path):
	records = []
	if not os.path.isfile(path):
		return records

	with open(path, "r", encoding="utf-8") as handle:
		for raw_line in handle:
			parts = raw_line.strip().split()
			if len(parts) < 10:
				continue
			try:
				poly = [float(value) for value in parts[:8]]
			except ValueError:
				continue
			cls_name = parts[8]
			difficult = parts[9]
			records.append(
				{
					"poly": poly,
					"cls": cls_name,
					"difficult": difficult,
					"line": raw_line.rstrip("\n"),
				}
			)
	return records


def read_pseudo_file(path):
	records = []
	if not os.path.isfile(path):
		return records

	with open(path, "r", encoding="utf-8") as handle:
		for raw_line in handle:
			parts = raw_line.strip().split()
			if len(parts) < 10:
				continue
			try:
				poly = [float(value) for value in parts[:8]]
				score = float(parts[9])
			except ValueError:
				continue
			cls_name = parts[8]
			records.append(
				{
					"poly": poly,
					"cls": cls_name,
					"score": score,
					"line": raw_line.rstrip("\n"),
				}
			)
	return records


def greedy_match(gt_records, pseudo_records):
	if not gt_records or not pseudo_records:
		return [None] * len(gt_records)

	pairs = []
	for gt_index, gt_record in enumerate(gt_records):
		for pseudo_index, pseudo_record in enumerate(pseudo_records):
			iou = polygon_iou(gt_record["poly"], pseudo_record["poly"])
			if iou > 0:
				pairs.append((iou, pseudo_record.get("score", 0.0), gt_index, pseudo_index))

	pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
	matched_gt = set()
	matched_pseudo = set()
	matches = [None] * len(gt_records)

	for iou, _, gt_index, pseudo_index in pairs:
		if gt_index in matched_gt or pseudo_index in matched_pseudo:
			continue
		matched_gt.add(gt_index)
		matched_pseudo.add(pseudo_index)
		matches[gt_index] = {
			"pseudo_index": pseudo_index,
			"iou": iou,
		}

	return matches


def list_gt_files(gt_dir):
	return sorted(
		[os.path.join(gt_dir, file_name) for file_name in os.listdir(gt_dir) if file_name.endswith(".txt")]
	)


def ensure_dir(path):
	os.makedirs(path, exist_ok=True)


def main():
	args = parse_args()

	output_pseudo_dir = os.path.join(args.output_dir, "pseudo_labels")
	ensure_dir(output_pseudo_dir)

	gt_files = list_gt_files(args.gt_dir)
	cls_names = list(DOTA_CLASSES)
	class_stats = {
		cls_name: {
			"gt_count": 0,
			"a_matched": 0,
			"b_matched": 0,
			"a_selected": 0,
			"b_selected": 0,
			"fallback_to_gt": 0,
			"a_iou_sum": 0.0,
			"b_iou_sum": 0.0,
			"selected_iou_sum": 0.0,
			"both_matched": 0,
		}
		for cls_name in cls_names
	}

	totals = defaultdict(float)

	for gt_path in gt_files:
		stem = os.path.splitext(os.path.basename(gt_path))[0]
		pseudo_a_path = os.path.join(args.pseudo_dir_a, stem + ".txt")
		pseudo_b_path = os.path.join(args.pseudo_dir_b, stem + ".txt")
		output_path = os.path.join(output_pseudo_dir, stem + ".txt")

		gt_records = read_gt_file(gt_path)
		pseudo_a_records = read_pseudo_file(pseudo_a_path)
		pseudo_b_records = read_pseudo_file(pseudo_b_path)

		gt_by_class = defaultdict(list)
		a_by_class = defaultdict(list)
		b_by_class = defaultdict(list)

		for record in gt_records:
			if record["cls"] in class_stats:
				gt_by_class[record["cls"]].append(record)

		for record in pseudo_a_records:
			if record["cls"] in class_stats:
				a_by_class[record["cls"]].append(record)

		for record in pseudo_b_records:
			if record["cls"] in class_stats:
				b_by_class[record["cls"]].append(record)

		merged_lines = []

		for cls_name in cls_names:
			gt_cls = gt_by_class[cls_name]
			if not gt_cls:
				continue

			a_matches = greedy_match(gt_cls, a_by_class[cls_name])
			b_matches = greedy_match(gt_cls, b_by_class[cls_name])

			for gt_index, gt_record in enumerate(gt_cls):
				class_stats[cls_name]["gt_count"] += 1

				a_match = a_matches[gt_index]
				b_match = b_matches[gt_index]

				a_iou = a_match["iou"] if a_match is not None else 0.0
				b_iou = b_match["iou"] if b_match is not None else 0.0

				if a_match is not None:
					class_stats[cls_name]["a_matched"] += 1
					class_stats[cls_name]["a_iou_sum"] += a_iou

				if b_match is not None:
					class_stats[cls_name]["b_matched"] += 1
					class_stats[cls_name]["b_iou_sum"] += b_iou

				if a_match is not None and b_match is not None:
					class_stats[cls_name]["both_matched"] += 1

				selected_record = None
				selected_iou = 0.0
				selected_source = None

				if a_match is not None and (b_match is None or a_iou > b_iou):
					selected_record = a_by_class[cls_name][a_match["pseudo_index"]]
					selected_iou = a_iou
					selected_source = "a"
				elif b_match is not None and (a_match is None or b_iou > a_iou):
					selected_record = b_by_class[cls_name][b_match["pseudo_index"]]
					selected_iou = b_iou
					selected_source = "b"
				elif a_match is not None and b_match is not None:
					a_score = a_by_class[cls_name][a_match["pseudo_index"]]["score"]
					b_score = b_by_class[cls_name][b_match["pseudo_index"]]["score"]
					if a_score >= b_score:
						selected_record = a_by_class[cls_name][a_match["pseudo_index"]]
						selected_iou = a_iou
						selected_source = "a"
					else:
						selected_record = b_by_class[cls_name][b_match["pseudo_index"]]
						selected_iou = b_iou
						selected_source = "b"

				if selected_source == "a":
					class_stats[cls_name]["a_selected"] += 1
				elif selected_source == "b":
					class_stats[cls_name]["b_selected"] += 1
				elif args.fallback_to_gt:
					class_stats[cls_name]["fallback_to_gt"] += 1
					selected_iou = 1.0

				class_stats[cls_name]["selected_iou_sum"] += selected_iou
				totals["gt_count"] += 1
				totals["a_matched"] += 1 if a_match is not None else 0
				totals["b_matched"] += 1 if b_match is not None else 0
				totals["a_selected"] += 1 if selected_source == "a" else 0
				totals["b_selected"] += 1 if selected_source == "b" else 0
				totals["fallback_to_gt"] += 1 if (selected_source is None and args.fallback_to_gt) else 0
				totals["a_iou_sum"] += a_iou
				totals["b_iou_sum"] += b_iou
				totals["selected_iou_sum"] += selected_iou

				if selected_record is not None:
					merged_lines.append(selected_record["line"])
				elif args.fallback_to_gt:
					merged_lines.append(format_pseudo_line(gt_record["poly"], gt_record["cls"], 1.0))

		with open(output_path, "w", encoding="utf-8") as handle:
			handle.write("\n".join(merged_lines))
			if merged_lines:
				handle.write("\n")

	summary = {
		"gt_dir": args.gt_dir,
		"pseudo_dir_a": args.pseudo_dir_a,
		"pseudo_dir_b": args.pseudo_dir_b,
		"output_dir": output_pseudo_dir,
		"fallback_to_gt": args.fallback_to_gt,
		"overall": {
			"gt_count": int(totals["gt_count"]),
			"a_matched_count": int(totals.get("a_matched", 0)),
			"b_matched_count": int(totals.get("b_matched", 0)),
			"a_selected_count": int(totals.get("a_selected", 0)),
			"b_selected_count": int(totals.get("b_selected", 0)),
			"fallback_count": int(totals.get("fallback_to_gt", 0)),
			"a_mean_iou_all_gt": float(totals["a_iou_sum"] / totals["gt_count"]) if totals["gt_count"] else 0.0,
			"b_mean_iou_all_gt": float(totals["b_iou_sum"] / totals["gt_count"]) if totals["gt_count"] else 0.0,
			"a_mean_iou_matched": float(totals["a_iou_sum"] / totals.get("a_matched", 1)) if totals.get("a_matched", 0) else 0.0,
			"b_mean_iou_matched": float(totals["b_iou_sum"] / totals.get("b_matched", 1)) if totals.get("b_matched", 0) else 0.0,
			"selected_mean_iou_all_gt": float(totals["selected_iou_sum"] / totals["gt_count"]) if totals["gt_count"] else 0.0,
			"selected_mean_iou_selected": float(totals["selected_iou_sum"] / max(1.0, (totals.get("a_selected", 0) + totals.get("b_selected", 0)))) if totals.get("a_selected", 0) + totals.get("b_selected", 0) else 0.0,
			"a_selected_rate": float(totals["a_selected"] / totals["gt_count"]) if totals["gt_count"] else 0.0,
			"b_selected_rate": float(totals["b_selected"] / totals["gt_count"]) if totals["gt_count"] else 0.0,
			"fallback_rate": float(totals["fallback_to_gt"] / totals["gt_count"]) if totals["gt_count"] else 0.0,
		},
		"per_class": {},
	}

	for cls_name, stat in class_stats.items():
		gt_count = stat["gt_count"]
		if gt_count == 0:
			continue
		a_matched = stat["a_matched"]
		b_matched = stat["b_matched"]
		a_selected = stat["a_selected"]
		b_selected = stat["b_selected"]
		both_matched = stat["both_matched"]
		fallback_count = stat["fallback_to_gt"]

		summary["per_class"][cls_name] = {
			"gt_count": gt_count,
			"a_matched_count": int(a_matched),
			"b_matched_count": int(b_matched),
			"a_selected_count": int(a_selected),
			"b_selected_count": int(b_selected),
			"both_matched_count": int(both_matched),
			"fallback_count": int(fallback_count),
			"a_matched_rate": a_matched / gt_count,
			"b_matched_rate": b_matched / gt_count,
			"a_mean_iou_all_gt": stat["a_iou_sum"] / gt_count,
			"b_mean_iou_all_gt": stat["b_iou_sum"] / gt_count,
			"a_mean_iou_matched": stat["a_iou_sum"] / a_matched if a_matched else 0.0,
			"b_mean_iou_matched": stat["b_iou_sum"] / b_matched if b_matched else 0.0,
			"selected_mean_iou_all_gt": stat["selected_iou_sum"] / gt_count,
			"selected_mean_iou_selected": stat["selected_iou_sum"] / max(1.0, (a_selected + b_selected)) if (a_selected + b_selected) else 0.0,
			"a_selected_rate": a_selected / gt_count,
			"b_selected_rate": b_selected / gt_count,
			"fallback_rate": fallback_count / gt_count,
			"both_matched_rate": both_matched / gt_count,
		}

	stats_path = os.path.join(args.output_dir, args.stats_name)
	with open(stats_path, "w", encoding="utf-8") as handle:
		json.dump(summary, handle, indent=2, ensure_ascii=False)

	print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
	print(f"Merged pseudo labels written to: {output_pseudo_dir}")
	print(f"Statistics written to: {stats_path}")


if __name__ == "__main__":
	main()
