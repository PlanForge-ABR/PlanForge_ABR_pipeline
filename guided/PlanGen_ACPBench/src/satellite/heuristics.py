from collections import defaultdict


def heuristic(state, goal):
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    """
    Heuristic for the Satellite domain.

    We approximate the effort to achieve each goal fact by looking at:
    - Which instruments support which modes (`supports`),
    - Which satellite they are on (`on_board`),
    - Whether they are calibrated (`calibrated`) and powered (`power_on`),
    - Whether the satellite has available power (`power_avail`),
    - Current pointing direction of satellites (`pointing`).

    For `have_image(obj, mode)` we estimate a cost based on missing
    preconditions for some suitable instrument/satellite pair.
    For `pointing(sat, target)` we charge 1 if the satellite is not currently
    pointing at that target.
    All other unsatisfied goals contribute cost 1.
    """
    state_set = {(p.get("predicate"), tuple(p.get("args", []))) for p in state}

    # Extract domain structure
    supports_mode = defaultdict(set)  # mode -> {instrument}
    instrument_sat = {}  # instrument -> satellite
    calibrated = set()
    power_on = set()
    power_avail = set()
    pointing = {}  # satellite -> target

    for p in state:
        pred = p.get("predicate")
        args = p.get("args", [])

        if pred == "supports" and len(args) >= 2:
            inst, mode = args[0], args[1]
            supports_mode[mode].add(inst)
        elif pred == "on_board" and len(args) >= 2:
            inst, sat = args[0], args[1]
            instrument_sat[inst] = sat
        elif pred == "calibrated" and len(args) >= 1:
            calibrated.add(args[0])
        elif pred == "power_on" and len(args) >= 1:
            power_on.add(args[0])
        elif pred == "power_avail" and len(args) >= 1:
            power_avail.add(args[0])
        elif pred == "pointing" and len(args) >= 2:
            sat, target = args[0], args[1]
            pointing[sat] = target

    total = 0

    for g in goal:
        pred = g.get("predicate")
        args = g.get("args", [])
        key = (pred, tuple(args))

        if key in state_set:
            continue

        if pred == "have_image" and len(args) >= 2:
            obj, mode = args[0], args[1]
            candidates = supports_mode.get(mode, set())

            best = None
            for inst in candidates:
                sat = instrument_sat.get(inst)
                if sat is None:
                    continue

                cost = 0
                if inst not in calibrated:
                    cost += 1
                if inst not in power_on:
                    cost += 1
                if sat not in power_avail:
                    cost += 1
                if pointing.get(sat) != obj:
                    cost += 1

                if best is None or cost < best:
                    best = cost

            total += best if best is not None else 4

        elif pred == "pointing" and len(args) >= 2:
            sat, target = args[0], args[1]
            # If satellite exists but points elsewhere, assume 1 step to repoint.
            if pointing.get(sat) is None:
                total += 1
            elif pointing.get(sat) != target:
                total += 1
            else:
                # Already covered by state_set check, but keep for clarity
                total += 0

        else:
            total += 1

    return int(total)

