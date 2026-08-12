"""Improved heuristic for the Rovers domain.

Implements a relaxed-planning (h_max) style heuristic: build a set of
grounded, relaxed actions from the current state (ignoring deletes),
propagate minimal costs to achieve facts where action cost = 1 + max(preconds),
and return the maximum cost among goal facts (h_max). This is admissible
and typically much more informative than simple goal-count heuristics.
"""

from collections import defaultdict, deque
import math


def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    # Represent facts as tuples (pred, args...)
    facts = set((p['predicate'],) + tuple(p.get('args', [])) for p in state)

    # Collect domain objects observed in the state
    rovers = set()
    waypoints = set()
    cameras = set()
    objectives = set()
    stores = set()
    channels = set()
    modes = set()

    can_traverse = set()      # (rover, from, to)
    visible_from = set()      # (objective, wp)
    supports = set()          # (camera, mode)
    calibration_targets = {}  # camera -> objective
    store_of = {}             # store -> rover

    for (pred, *args) in facts:
        if pred == 'at' and len(args) >= 2:
            rovers.add(args[0]); waypoints.add(args[1])
        elif pred == 'at_lander' and len(args) >= 2:
            waypoints.add(args[1])
        elif pred == 'can_traverse' and len(args) >= 3:
            can_traverse.add((args[0], args[1], args[2]))
            rovers.add(args[0]); waypoints.add(args[1]); waypoints.add(args[2])
        elif pred == 'on_board' and len(args) >= 2:
            cameras.add(args[0]); rovers.add(args[1])
        elif pred == 'calibration_target' and len(args) >= 2:
            calibration_targets[args[0]] = args[1]; objectives.add(args[1])
        elif pred == 'visible_from' and len(args) >= 2:
            visible_from.add((args[0], args[1])); objectives.add(args[0]); waypoints.add(args[1])
        elif pred == 'supports' and len(args) >= 2:
            supports.add((args[0], args[1])); cameras.add(args[0]); modes.add(args[1])
        elif pred == 'store_of' and len(args) >= 2:
            store_of[args[0]] = args[1]; stores.add(args[0]); rovers.add(args[1])
        elif pred == 'channel_free' and len(args) >= 1:
            channels.add(args[0])

    # Grounded relaxed actions: list of (preconds_set, add_effects_set)
    actions = []

    # navigate(rover, from, to)
    for (r, frm, to) in can_traverse:
        pre = set([('at', r, frm)])
        adds = set([('at', r, to)])
        actions.append((pre, adds))

    # calibrate(camera, rover) -> adds calibrated(camera, rover)
    for cam, obj in calibration_targets.items():
        # camera may be on_board some rover; we don't know which, so enumerate rovers
        for r in rovers:
            pre = set([('on_board', cam, r), ('calibration_target', cam, obj)])
            adds = set([('calibrated', cam, r)])
            actions.append((pre, adds))

    # take_image(rover, camera, objective, mode, wp)
    # pre: at(rover, wp), visible_from(objective, wp), on_board(camera, rover), calibrated(camera, rover), supports(camera, mode)
    for (obj, wp) in visible_from:
        for cam, mode in supports:
            for r in rovers:
                pre = set([('at', r, wp), ('visible_from', obj, wp), ('on_board', cam, r), ('calibrated', cam, r), ('supports', cam, mode)])
                adds = set([('have_image', r, obj, mode)])
                actions.append((pre, adds))

    # sample rock
    for (pred, *args) in facts:
        if pred == 'at_rock_sample':
            wp = args[0]
            for s, r in store_of.items():
                pre = set([('at', r, wp), ('at_rock_sample', wp), ('store_of', s, r)])
                adds = set([('have_rock_analysis', r, wp)])
                actions.append((pre, adds))

    # sample soil
    for (pred, *args) in facts:
        if pred == 'at_soil_sample':
            wp = args[0]
            for s, r in store_of.items():
                pre = set([('at', r, wp), ('at_soil_sample', wp), ('store_of', s, r)])
                adds = set([('have_soil_analysis', r, wp)])
                actions.append((pre, adds))

    # communications: image/rock/soil
    for ch in channels or [None]:
        # images
        for (p, *args) in facts:
            if p == 'have_image':
                r, obj, mode = args
                pre = set([('have_image', r, obj, mode), ('channel_free', ch)]) if ch is not None else set([('have_image', r, obj, mode)])
                adds = set([('communicated_image_data', obj, mode)])
                actions.append((pre, adds))
        # rock
        for (p, *args) in facts:
            if p == 'have_rock_analysis':
                r, wp = args
                pre = set([('have_rock_analysis', r, wp), ('channel_free', ch)]) if ch is not None else set([('have_rock_analysis', r, wp)])
                adds = set([('communicated_rock_data', wp)])
                actions.append((pre, adds))
        # soil
        for (p, *args) in facts:
            if p == 'have_soil_analysis':
                r, wp = args
                pre = set([('have_soil_analysis', r, wp), ('channel_free', ch)]) if ch is not None else set([('have_soil_analysis', r, wp)])
                adds = set([('communicated_soil_data', wp)])
                actions.append((pre, adds))

    # Initialize costs: 0 for facts present, inf otherwise
    INF = math.inf
    cost = defaultdict(lambda: INF)
    for f in facts:
        cost[f] = 0

    # h_max propagation: action cost = 1 + max(costs of preconds); effect cost = min(current, action cost)
    changed = True
    iterations = 0
    max_iter = 10000
    while changed and iterations < max_iter:
        changed = False
        iterations += 1
        for pre, adds in actions:
            # compute max cost of preconditions
            pre_costs = []
            skip = False
            for p in pre:
                # p is tuple like ('at', r, wp)
                if cost[p] == INF:
                    pre_costs.append(INF)
                else:
                    pre_costs.append(cost[p])
            if any(c == INF for c in pre_costs):
                # action not yet applicable in relaxation
                action_cost = INF
            else:
                action_cost = 1 + max(pre_costs) if pre_costs else 1
            if action_cost == INF:
                continue
            for e in adds:
                if action_cost < cost[e]:
                    cost[e] = action_cost
                    changed = True

    # compute heuristic as max cost of goal literals (not yet satisfied)
    h_vals = []
    for g in goal:
        g_t = (g['predicate'],) + tuple(g.get('args', []))
        if g_t in facts:
            h_vals.append(0)
        else:
            h_vals.append(cost.get(g_t, INF))

    if not h_vals:
        return 0
    h = max(h_vals)
    if h == INF:
        # unreachable in relaxed graph
        return 9999
    return int(h)
