import json
import math
import os
import statistics
from pathlib import Path

DEFAULT_FILES = [
    Path(r"C:\Users\Cassmall\Downloads\TimeTraversal.json"),
    Path(r"C:\Users\Cassmall\Downloads\TimeIteration.json"),
]


def load_points(path: Path):
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return [{"timestamp": row[0], "iteration": row[1], "duration": row[2]} for row in rows]


def linear_regression(points):
    xs = [point["iteration"] for point in points]
    ys = [point["duration"] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def pearson_correlation(points):
    xs = [point["iteration"] for point in points]
    ys = [point["duration"] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return numerator / (denominator_x * denominator_y)


def summarize_segments(points, segment_size=50):
    max_iteration = max(point["iteration"] for point in points)
    start = 1
    while start <= max_iteration:
        end = start + segment_size - 1
        segment = [point["duration"] for point in points if start <= point["iteration"] <= end]
        if segment:
            yield start, end, len(segment), statistics.mean(segment), min(segment), max(segment)
        start += segment_size


def top_slowest(points, limit=10):
    return sorted(points, key=lambda point: point["duration"], reverse=True)[:limit]


def top_jumps(points, limit=10):
    jumps = []
    ordered = sorted(points, key=lambda point: point["iteration"])
    for previous, current in zip(ordered, ordered[1:]):
        jumps.append({
            "from": previous["iteration"],
            "to": current["iteration"],
            "previous_duration": previous["duration"],
            "current_duration": current["duration"],
            "delta": current["duration"] - previous["duration"],
        })
    return sorted(jumps, key=lambda jump: jump["delta"], reverse=True)[:limit]


def build_report(files=DEFAULT_FILES):
    lines = []
    loaded = {path.name: load_points(path) for path in files}

    lines.append("TIME ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    for filename, points in loaded.items():
        durations = [point["duration"] for point in points]
        slope, _intercept = linear_regression(points)
        correlation = pearson_correlation(points)
        first20 = statistics.mean(durations[:20])
        last20 = statistics.mean(durations[-20:])

        lines.append(filename)
        lines.append("-" * 80)
        lines.append(f"Records: {len(points)}")
        lines.append(f"Min duration: {min(durations):.3f} sec")
        lines.append(f"Max duration: {max(durations):.3f} sec")
        lines.append(f"Average duration: {statistics.mean(durations):.3f} sec")
        lines.append(f"Median duration: {statistics.median(durations):.3f} sec")
        lines.append(f"First 20 average: {first20:.3f} sec")
        lines.append(f"Last 20 average: {last20:.3f} sec")
        lines.append(f"Growth first20 -> last20: {last20 / first20:.2f}x")
        lines.append(f"Linear trend: +{slope:.5f} sec per iteration")
        lines.append(f"Estimated trend growth per 100 iterations: +{slope * 100:.3f} sec")
        lines.append(f"Iteration/duration correlation: {correlation:.3f}")
        lines.append("")
        lines.append("Segments:")
        for start, end, count, avg, min_value, max_value in summarize_segments(points):
            lines.append(f"  {start:03d}-{end:03d}: n={count:3d}, avg={avg:6.3f}, min={min_value:6.3f}, max={max_value:6.3f}")
        lines.append("")
        lines.append("Top 10 slowest iterations:")
        for point in top_slowest(points):
            lines.append(f"  iteration={point['iteration']:3d}, duration={point['duration']:6.3f} sec")
        lines.append("")
        lines.append("Top 10 positive jumps between adjacent iterations:")
        for jump in top_jumps(points):
            lines.append(
                f"  {jump['from']:3d}->{jump['to']:3d}: "
                f"{jump['previous_duration']:6.3f} -> {jump['current_duration']:6.3f} "
                f"delta=+{jump['delta']:6.3f} sec"
            )
        lines.append("")

    traversal = loaded.get("TimeTraversal.json")
    iteration = loaded.get("TimeIteration.json")
    if traversal and iteration:
        traversal_by_iter = {point["iteration"]: point["duration"] for point in traversal}
        iteration_by_iter = {point["iteration"]: point["duration"] for point in iteration}
        common_iterations = sorted(set(traversal_by_iter) & set(iteration_by_iter))
        overheads = [iteration_by_iter[i] - traversal_by_iter[i] for i in common_iterations]
        lines.append("COMPARISON")
        lines.append("-" * 80)
        lines.append(f"Common iterations: {len(common_iterations)}")
        lines.append(f"Average TimeIteration - TimeTraversal: {statistics.mean(overheads):.6f} sec")
        lines.append(f"Median TimeIteration - TimeTraversal: {statistics.median(overheads):.6f} sec")
        lines.append(f"Min overhead: {min(overheads):.6f} sec")
        lines.append(f"Max overhead: {max(overheads):.6f} sec")
        lines.append("")
        lines.append("Conclusion:")
        lines.append("  TimeIteration almost equals TimeTraversal. The slowdown is inside traversal/data collection.")
        lines.append("  The most likely reason is that traversals become larger/longer as the policy changes")
        lines.append("  or that buffers/state/memory pressure grow during training.")
        lines.append("")
        lines.append("What to verify next:")
        lines.append("  1. Log max depth and number of recursive cfr_traverse_multi calls per iteration.")
        lines.append("  2. Compare those counters with Time/Traversal.")
        lines.append("  3. Watch Memory/Strategy, Memory/PG, Memory/AdvantageTotal and process RSS.")
        lines.append("  4. If nodes/depth grow, the slowdown is expected behavior from longer game trees.")
        lines.append("  5. If nodes/depth do not grow but RSS grows, look for memory pressure/GC/paging.")

    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    report_path = Path("TimeAnalysisReport.txt")
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {report_path.resolve()}")
