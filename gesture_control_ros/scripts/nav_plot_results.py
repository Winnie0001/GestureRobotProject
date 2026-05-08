#!/usr/bin/env python3

import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot navigation evaluation results')
    parser.add_argument('--input', type=str, default='nav_evaluation_results.json')
    parser.add_argument('--output', type=str, default='nav_evaluation_plots.png')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    runs = data.get('runs', [])
    if not runs:
        raise SystemExit('No runs found in input JSON')

    outcomes = [r.get('outcome') or 'UNKNOWN' for r in runs]
    times = [r.get('time_to_done_s') for r in runs]
    goal_ids = [r.get('goal_id', '') for r in runs]

    counts = Counter(outcomes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = list(counts.keys())
    sizes = [counts[k] for k in labels]
    axes[0].pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90)
    axes[0].set_title('Navigation Goal Outcomes')

    xs = list(range(len(times)))
    safe_times = [t if t is not None else 0.0 for t in times]
    axes[1].bar(xs, safe_times, color='#378ADD')
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(goal_ids, rotation=20, ha='right')
    axes[1].set_ylabel('Time to done (s)')
    axes[1].set_title('Time to Goal (per run)')

    fig.suptitle('Gesture-Controlled Navigation Evaluation', fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()
