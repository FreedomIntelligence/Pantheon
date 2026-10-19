"""A deliberately small local, deterministic CPU simulation reference.

Not a distributed runtime or trained persona policy.
"""
from dataclasses import dataclass
from random import Random


@dataclass
class Agent:
    agent_id: int
    preference: float
    state: int = 0


@dataclass(frozen=True)
class SimulationResult:
    agents: int
    ticks: int
    decisions: int
    positive_actions: int


class Simulation:
    def __init__(self, population: int, seed: int = 42):
        if population < 0:
            raise ValueError("population must be non-negative")
        rng = Random(seed)
        self.agents = [Agent(i, rng.random()) for i in range(population)]
        self.tick_count = 0
        self.decisions = 0
        self.positive_actions = 0

    def step(self):
        """Toy preference-conditioned decision (not a learned model)."""
        for agent in self.agents:
            action = int(agent.preference > 0.5)
            agent.state += action
            self.positive_actions += action
            self.decisions += 1
        self.tick_count += 1

    def run(self, ticks: int) -> SimulationResult:
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        for _ in range(ticks):
            self.step()
        return SimulationResult(
            agents=len(self.agents),
            ticks=self.tick_count,
            decisions=self.decisions,
            positive_actions=self.positive_actions,
        )
