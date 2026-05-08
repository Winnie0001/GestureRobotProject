#!/usr/bin/env python3

import argparse
import json
import math
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_events(path: str):
    with open(path, 'r') as f:
        payload = json.load(f)
    return payload


def _extract_series(events, topic: str):
    ts = []
    xs = []
    for e in events:
        if e.get('topic') != topic:
            continue
        ts.append(float(e.get('t_s', 0.0)))
        xs.append(e.get('data', {}))
    return np.array(ts, dtype=float), xs


def _cmd_vel_series(events):
    t, data = _extract_series(events, '/cmd_vel')
    if len(t) == 0:
        return t, np.array([]), np.array([])
    lin_x = np.array([float(d.get('linear_x', 0.0)) for d in data], dtype=float)
    ang_z = np.array([float(d.get('angular_z', 0.0)) for d in data], dtype=float)
    return t, lin_x, ang_z


def _estimate_jerk(t: np.ndarray, v: np.ndarray) -> float:
    if len(t) < 3:
        return 0.0
    dt = np.diff(t)
    if np.any(dt <= 0):
        return 0.0
    a = np.diff(v) / dt
    dt2 = np.diff(t[1:])
    if np.any(dt2 <= 0) or len(a) < 2:
        return 0.0
    j = np.diff(a) / dt2
    return float(np.mean(np.abs(j))) if len(j) else 0.0


def _stop_reaction_time(events, vel_eps: float = 0.02):
    t_cmd, cmd_data = _extract_series(events, '/gesture/command')
    t_vel, lin_x, ang_z = _cmd_vel_series(events)
    if len(t_cmd) == 0 or len(t_vel) == 0:
        return None

    stop_times = [t_cmd[i] for i, d in enumerate(cmd_data) if d.get('command') == 'STOP']
    if not stop_times:
        return None

    speeds = np.maximum(np.abs(lin_x), np.abs(ang_z))

    rts = []
    for ts in stop_times:
        idx = np.searchsorted(t_vel, ts, side='left')
        if idx >= len(t_vel):
            continue
        j = idx
        while j < len(t_vel) and speeds[j] > vel_eps:
            j += 1
        if j < len(t_vel):
            rts.append(float(t_vel[j] - ts))

    if not rts:
        return None
    return float(np.mean(rts))


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze gesture_session_*.json and output metrics/plots')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--metrics-out', type=str, default='session_metrics.json')
    parser.add_argument('--plot-out', type=str, default='session_plots.png')
    args = parser.parse_args()

    payload = _load_events(args.input)
    events = payload.get('events', [])

    duration_s = float(payload.get('duration_s', 0.0))

    t_g, g_data = _extract_series(events, '/gesture/detected')
    gestures = [d.get('gesture') for d in g_data if d.get('gesture')]

    t_c, c_data = _extract_series(events, '/gesture/command')
    commands = [d.get('command') for d in c_data if d.get('command')]

    t_v, lin_x, ang_z = _cmd_vel_series(events)

    cmd_rate_hz = (len(commands) / duration_s) if duration_s > 0 else 0.0

    jerk_lin = _estimate_jerk(t_v, lin_x) if len(t_v) else 0.0
    jerk_ang = _estimate_jerk(t_v, ang_z) if len(t_v) else 0.0

    stop_rt = _stop_reaction_time(events)

    gesture_counts = Counter(gestures)
    command_counts = Counter(commands)

    metrics = {
        'input': args.input,
        'duration_s': round(duration_s, 3),
        'gesture_events': len(gestures),
        'command_events': len(commands),
        'cmd_rate_hz': round(float(cmd_rate_hz), 3),
        'jerk_mean_abs_linear_x': round(float(jerk_lin), 6),
        'jerk_mean_abs_angular_z': round(float(jerk_ang), 6),
        'stop_reaction_time_s': None if stop_rt is None else round(float(stop_rt), 3),
        'gesture_counts': dict(gesture_counts),
        'command_counts': dict(command_counts),
    }

    with open(args.metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # cmd_vel
    if len(t_v):
        axes[0, 0].plot(t_v, lin_x, label='linear.x')
        axes[0, 0].plot(t_v, ang_z, label='angular.z')
        axes[0, 0].set_title('cmd_vel over time')
        axes[0, 0].set_xlabel('t (s)')
        axes[0, 0].legend()
    else:
        axes[0, 0].set_title('cmd_vel over time (no data)')

    # command histogram
    if command_counts:
        labels = list(command_counts.keys())
        vals = [command_counts[k] for k in labels]
        axes[0, 1].bar(labels, vals, color='#1D9E75')
        axes[0, 1].set_title('Command counts')
        axes[0, 1].tick_params(axis='x', rotation=20)
    else:
        axes[0, 1].set_title('Command counts (no data)')

    # gesture histogram
    if gesture_counts:
        labels = list(gesture_counts.keys())
        vals = [gesture_counts[k] for k in labels]
        axes[1, 0].bar(labels, vals, color='#378ADD')
        axes[1, 0].set_title('Gesture counts')
        axes[1, 0].tick_params(axis='x', rotation=20)
    else:
        axes[1, 0].set_title('Gesture counts (no data)')

    # summary text
    axes[1, 1].axis('off')
    lines = [
        f"Duration: {duration_s:.1f}s",
        f"Cmd rate: {cmd_rate_hz:.2f} Hz",
        f"Jerk lin.x: {jerk_lin:.4f}",
        f"Jerk ang.z: {jerk_ang:.4f}",
        f"STOP reaction: {('n/a' if stop_rt is None else f'{stop_rt:.2f}s')}",
    ]
    axes[1, 1].text(0.02, 0.98, '\n'.join(lines), va='top', fontsize=11)

    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()
