from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable

from architect.state_schema import TripState


class AbstractPlanner(ABC):
    @abstractmethod
    def get_initial_state(self) -> TripState:
        raise NotImplementedError

    @abstractmethod
    def get_goal_test(self) -> Callable[[TripState], bool]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self) -> Callable[[TripState], Iterable[TripState]]:
        raise NotImplementedError

