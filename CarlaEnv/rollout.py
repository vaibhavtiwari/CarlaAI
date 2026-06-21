from dataclasses import dataclass, field


def _safe_div(numerator, denominator):
    if abs(denominator) < 1e-8:
        return 0.0
    return numerator / denominator


@dataclass
class EpisodeTrace:
    scenario_name: str
    steps: list = field(default_factory=list)
    terminal_reason: str = "Running..."

    def reset(self):
        self.steps.clear()
        self.terminal_reason = "Running..."

    def record_step(self, step_record):
        self.steps.append(step_record)

    def finalize(self, terminal_reason=None):
        if terminal_reason:
            self.terminal_reason = terminal_reason

    def summary(self):
        if not self.steps:
            return {
                "scenario": self.scenario_name,
                "num_steps": 0,
                "total_reward": 0.0,
                "distance_traveled": 0.0,
                "average_speed_kmh": 0.0,
                "center_lane_deviation": 0.0,
                "average_center_lane_deviation": 0.0,
                "distance_over_deviation": 0.0,
                "collisions": 0,
                "lane_invasions": 0,
                "terminal_reason": self.terminal_reason,
            }

        last = self.steps[-1]
        num_steps = len(self.steps)
        total_reward = sum(step["reward"] for step in self.steps)
        distance_traveled = last["distance_traveled"]
        center_lane_deviation = last["center_lane_deviation"]
        average_speed_kmh = _safe_div(sum(step["speed_kmh"] for step in self.steps), num_steps)
        average_center_lane_deviation = _safe_div(center_lane_deviation, num_steps)
        summary = {
            "scenario": self.scenario_name,
            "num_steps": num_steps,
            "total_reward": total_reward,
            "distance_traveled": distance_traveled,
            "average_speed_kmh": average_speed_kmh,
            "center_lane_deviation": center_lane_deviation,
            "average_center_lane_deviation": average_center_lane_deviation,
            "distance_over_deviation": _safe_div(distance_traveled, center_lane_deviation),
            "collisions": sum(1 for step in self.steps if step["collision"]),
            "lane_invasions": sum(1 for step in self.steps if step["lane_invasion"]),
            "terminal_reason": self.terminal_reason,
        }
        summary.update(last.get("task_metrics", {}))
        return summary
