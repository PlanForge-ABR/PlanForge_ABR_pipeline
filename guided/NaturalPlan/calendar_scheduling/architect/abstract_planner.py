from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

from architect.state_schema import CalendarAction, CalendarState


class AbstractPlanner(ABC):
    @abstractmethod
    def get_initial_state(self) -> CalendarState:
        raise NotImplementedError

    @abstractmethod
    def get_goal_test(self, state: CalendarState) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_successors(
        self, state: CalendarState
    ) -> Iterable[Tuple[CalendarAction, CalendarState]]:
        raise NotImplementedError

