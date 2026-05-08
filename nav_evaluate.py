#!/usr/bin/env python3

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import rospy
from std_msgs.msg import String


@dataclass
class GoalRun:
    goal_id: str
    sent_time_s: float
    active_time_s: Optional[float] = None
    done_time_s: Optional[float] = None
    outcome: Optional[str] = None

    @property
    def time_to_active_s(self) -> Optional[float]:
        if self.active_time_s is None:
            return None
        return self.active_time_s - self.sent_time_s

    @property
    def time_to_done_s(self) -> Optional[float]:
        if self.done_time_s is None:
            return None
        return self.done_time_s - self.sent_time_s


class NavEvaluator:
    def __init__(self):
        self.runs: Dict[str, GoalRun] = {}
        self.last_sent_goal: Optional[str] = None

        self.output_path = rospy.get_param('~output', 'nav_evaluation_results.json')
        self.session_start = time.time()

        rospy.Subscriber('/gesture/nav_status', String, self._status_cb)
        rospy.on_shutdown(self._on_shutdown)

    def _status_cb(self, msg: String) -> None:
        now = time.time()
        data = (msg.data or '').strip()
        if not data:
            return

        # Expected formats:
        # SENT:WP1
        # ACTIVE:WP1
        # FEEDBACK:WP1
        # DONE:WP1:SUCCEEDED
        # CANCELED
        parts = data.split(':')
        tag = parts[0]

        if tag == 'SENT' and len(parts) >= 2:
            goal_id = parts[1]
            self.runs[goal_id] = GoalRun(goal_id=goal_id, sent_time_s=now)
            self.last_sent_goal = goal_id
            rospy.loginfo(f"[NavEval] SENT {goal_id}")
            return

        if tag == 'ACTIVE' and len(parts) >= 2:
            goal_id = parts[1]
            run = self.runs.get(goal_id)
            if run and run.active_time_s is None:
                run.active_time_s = now
                rospy.loginfo(f"[NavEval] ACTIVE {goal_id} (t={run.time_to_active_s:.2f}s)")
            return

        if tag == 'DONE' and len(parts) >= 3:
            goal_id = parts[1]
            outcome = parts[2]
            run = self.runs.get(goal_id)
            if run and run.done_time_s is None:
                run.done_time_s = now
                run.outcome = outcome
                t = run.time_to_done_s
                rospy.loginfo(f"[NavEval] DONE {goal_id} -> {outcome} (t={t:.2f}s)")
            return

        if tag == 'CANCELED':
            # Mark last sent goal as canceled if not already done
            goal_id = self.last_sent_goal
            if goal_id:
                run = self.runs.get(goal_id)
                if run and run.done_time_s is None:
                    run.done_time_s = now
                    run.outcome = 'CANCELED'
                    t = run.time_to_done_s
                    rospy.loginfo(f"[NavEval] CANCELED {goal_id} (t={t:.2f}s)")
            return

    def _summarize(self) -> dict:
        total = len(self.runs)
        succeeded = sum(1 for r in self.runs.values() if r.outcome == 'SUCCEEDED')
        aborted = sum(1 for r in self.runs.values() if r.outcome == 'ABORTED')
        canceled = sum(1 for r in self.runs.values() if r.outcome == 'CANCELED')

        durations = [r.time_to_done_s for r in self.runs.values() if r.time_to_done_s is not None]
        avg_t = sum(durations) / len(durations) if durations else 0.0

        return {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'session_duration_s': round(time.time() - self.session_start, 3),
            'total_goals': total,
            'succeeded': succeeded,
            'aborted': aborted,
            'canceled': canceled,
            'success_rate': round((succeeded / total) if total else 0.0, 4),
            'avg_time_to_done_s': round(avg_t, 3),
            'runs': [asdict(r) | {
                'time_to_active_s': r.time_to_active_s,
                'time_to_done_s': r.time_to_done_s,
            } for r in self.runs.values()],
        }

    def _on_shutdown(self):
        try:
            summary = self._summarize()
            with open(self.output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            rospy.loginfo(f"[NavEval] Saved -> {self.output_path}")
        except Exception as exc:
            rospy.logwarn(f"[NavEval] Failed to save results: {exc}")


def main():
    rospy.init_node('nav_evaluator', anonymous=True)
    rospy.loginfo('[NavEval] Listening to /gesture/nav_status (Ctrl+C to stop)')
    NavEvaluator()
    rospy.spin()


if __name__ == '__main__':
    main()
