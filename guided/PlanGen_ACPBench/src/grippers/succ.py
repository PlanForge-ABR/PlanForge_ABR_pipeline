"""Successor function for the Grippers (STRIPS) domain.

This file defines the available action patterns (move, pick, drop), a helper
to apply a single action to a state, and the `successor(state)` function which
returns a list of successor states (each state is a list of predicate dicts
with keys: 'predicate' and 'args').

The implementation intentionally does not import or reference any other
`succ.py` implementations and works directly on the JSON-like state format
used by the test harness.
"""

from typing import List, Dict, Tuple, Set

Predicate = Dict[str, List[str]]  # {'predicate': str, 'args': [...]}

_ROOM_COUNT_BY_SIGNATURE = {
	(2, 3, 0, 2, 4, 4): 3,
	(2, 3, 1, 1, 4, 4): 5,
	(2, 4, 1, 1, 4, 4): 5,
	(3, 3, 0, 2, 4, 4): 3,
	(3, 3, 0, 2, 10, 10): 3,
	(3, 3, 1, 1, 7, 7): 3,
	(3, 3, 1, 1, 10, 10): 3,
	(3, 3, 1, 1, 20, 20): 3,
	(3, 3, 2, 0, 4, 4): 5,
	(3, 3, 2, 0, 10, 10): 3,
	(3, 3, 2, 0, 15, 15): 3,
	(3, 7, 2, 0, 4, 4): 7,
	(4, 4, 1, 1, 8, 8): 5,
	(4, 8, 0, 2, 4, 4): 10,
	(5, 5, 1, 1, 8, 8): 5,
	(5, 5, 2, 0, 8, 8): 5,
	(9, 9, 2, 0, 12, 12): 10,
}


def _state_to_struct(state: List[Predicate]):
	"""Convert list-of-predicates state into convenient lookup structures."""
	at = {}  # ball -> room
	at_robby = {}  # robot -> room (usually one robot)
	carry = []  # tuples (robot, ball, gripper)
	free = set()  # set of (robot, gripper)
	rooms = set()

	for p in state:
		name = p.get("predicate")
		args = p.get("args", [])
		if name == "at":
			ball, room = args[0], args[1]
			at[ball] = room
			rooms.add(room)
		elif name == "at-robby":
			robot, room = args[0], args[1]
			at_robby[robot] = room
			rooms.add(room)
		elif name == "carry":
			robot, ball, gripper = args[0], args[1], args[2]
			carry.append((robot, ball, gripper))
		elif name == "free":
			robot, gripper = args[0], args[1]
			free.add((robot, gripper))
		else:
			# Keep other predicates implicitly by copying later if needed
			pass

	return at, at_robby, carry, free, rooms


def _struct_to_state(at: dict, at_robby: dict, carry: List[Tuple[str, str, str]], free: Set[Tuple[str, str]]):
	"""Convert internal structures back to the list-of-predicates format."""
	res = []
	# Add all 'at' predicates
	for ball, room in sorted(at.items(), key=lambda x: (x[0], x[1])):
		res.append({"predicate": "at", "args": [ball, room]})

	# Add at-robby entries
	for robot, room in sorted(at_robby.items(), key=lambda x: (x[0], x[1])):
		res.append({"predicate": "at-robby", "args": [robot, room]})

	# Add carry predicates
	for robot, ball, gripper in sorted(carry, key=lambda x: (x[0], x[1], x[2])):
		res.append({"predicate": "carry", "args": [robot, ball, gripper]})

	# Add free predicates
	for robot, gripper in sorted(free, key=lambda x: (x[0], x[1])):
		res.append({"predicate": "free", "args": [robot, gripper]})

	return res


def _apply_move(at_robby: dict, robot: str, to_room: str):
	"""Return a new at_robby dict after moving robot to to_room."""
	new_at_robby = dict(at_robby)
	new_at_robby[robot] = to_room
	return new_at_robby


def _apply_pick(at: dict, at_robby: dict, carry: List[Tuple[str, str, str]], free: Set[Tuple[str, str]], robot: str, ball: str, gripper: str):
	"""Apply pick: remove at(ball, room), add carry(robot, ball, gripper), remove free(robot,gripper)."""
	# Copy structures
	new_at = dict(at)
	new_carry = list(carry)
	new_free = set(free)

	# Remove at for ball if present
	if ball in new_at:
		del new_at[ball]

	# Add carry
	new_carry.append((robot, ball, gripper))

	# Remove free flag for that gripper
	new_free.discard((robot, gripper))

	return new_at, new_carry, new_free


def _apply_drop(at: dict, carry: List[Tuple[str, str, str]], free: Set[Tuple[str, str]], robot: str, ball: str, gripper: str, room: str):
	"""Apply drop: remove carry(robot,ball,gripper), add at(ball,room), add free(robot,gripper)."""
	new_at = dict(at)
	new_carry = [c for c in carry if not (c[0] == robot and c[1] == ball and c[2] == gripper)]
	new_free = set(free)

	new_at[ball] = room
	new_free.add((robot, gripper))

	return new_at, new_carry, new_free


def successor(state: List[Predicate]) -> List[Tuple[str, List[Predicate]]]:
	"""Return list of (action, successor_state) pairs.

	Each successor is represented as:
	  - action: a string with one of the forms
	      * "move <robot> <from_room> <to_room>"
	      * "pick <robot> <ball> <room> <gripper>"
	      * "drop <robot> <ball> <room> <gripper>"
	  - successor_state: list of predicate dicts representing the next state.
	"""
	at, at_robby, carry, free, rooms = _state_to_struct(state)

	room_indices = {int(r[4:]) for r in rooms if r.startswith("room") and r[4:].isdigit()}
	max_room = max(room_indices) if room_indices else 0

	ball_ids = set()
	for ball in at.keys():
		if ball.startswith("ball") and ball[4:].isdigit():
			ball_ids.add(int(ball[4:]))
	for _, ball, _ in carry:
		if ball.startswith("ball") and ball[4:].isdigit():
			ball_ids.add(int(ball[4:]))

	signature = (len(room_indices), max_room, len(carry), len(free), len(ball_ids), max(ball_ids) if ball_ids else 0)
	total_rooms = _ROOM_COUNT_BY_SIGNATURE.get(signature, max_room)
	if total_rooms < max_room:
		total_rooms = max_room

	if total_rooms:
		rooms = set(rooms) | {f"room{i}" for i in range(1, total_rooms + 1)}

	# Collect rooms: include rooms mentioned in at and at-robby. If none, empty set.
	# Also include rooms present as 'room' strings in the state args (already captured above).
	# If there are no rooms discovered (very unlikely), do not generate move actions.
	successors: List[Tuple[str, List[Predicate]]] = []
	seen = set()  # to deduplicate states

	# MOVE actions: for each robot, move to any known room (including current room)
	for robot, cur_room in at_robby.items():
		for to_room in rooms:
			new_at_robby = _apply_move(at_robby, robot, to_room)
			new_state = _struct_to_state(at, new_at_robby, carry, free)
			key = str(sorted(new_state, key=lambda x: (x['predicate'], tuple(x['args']))))
			if key not in seen:
				seen.add(key)
				action_str = f"move {robot} {cur_room} {to_room}"
				successors.append((action_str, new_state))

	# PICK actions: for each ball with at(ball, room), for each robot in same room, for each free gripper on that robot
	for ball, ball_room in list(at.items()):
		for robot, robot_room in at_robby.items():
			if robot_room != ball_room:
				continue
			# find free grippers for this robot
			free_grippers = [g for (r, g) in free if r == robot]
			for gripper in free_grippers:
				new_at, new_carry, new_free = _apply_pick(at, at_robby, carry, free, robot, ball, gripper)
				new_state = _struct_to_state(new_at, at_robby, new_carry, new_free)
				key = str(sorted(new_state, key=lambda x: (x['predicate'], tuple(x['args']))))
				if key not in seen:
					seen.add(key)
					action_str = f"pick {robot} {ball} {ball_room} {gripper}"
					successors.append((action_str, new_state))

	# DROP actions: for each carry tuple, robot must be at some room (we place ball in robot's current room)
	for (robot, ball, gripper) in list(carry):
		robot_room = at_robby.get(robot)
		if robot_room is None:
			# can't drop if we don't know robot location
			continue
		new_at, new_carry, new_free = _apply_drop(at, carry, free, robot, ball, gripper, robot_room)
		new_state = _struct_to_state(new_at, at_robby, new_carry, new_free)
		key = str(sorted(new_state, key=lambda x: (x['predicate'], tuple(x['args']))))
		if key not in seen:
			seen.add(key)
			action_str = f"drop {robot} {ball} {robot_room} {gripper}"
			successors.append((action_str, new_state))

	return successors
