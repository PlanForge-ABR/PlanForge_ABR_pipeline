"""Planner interface required by the builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .state_schema import MeetingState


class AbstractPlanner(ABC):
    @abstractmethod
    def get_initial_state(self) -> MeetingState:
        raise NotImplementedError

    @abstractmethod
    def get_goal_test(self):
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, state: MeetingState) -> Iterable[MeetingState]:
        raise NotImplementedError

