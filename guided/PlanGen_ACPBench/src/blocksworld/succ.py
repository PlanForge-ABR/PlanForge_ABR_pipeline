from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple, Optional


Fact = Tuple[str, Tuple[str, ...]]


def _normalize_facts(items: Iterable[Dict[str, Any]]) -> Set[Fact]:
    out: Set[Fact] = set()
    for f in items:
        pred = f.get("predicate")
        args = f.get("args", [])
        if not isinstance(args, (list, tuple)):
            args = [args]
        out.add((str(pred), tuple(str(a) for a in args)))
    return out


def _denormalize_facts(facts: Set[Fact]) -> List[Dict[str, Any]]:
    # Return a deterministically ordered list for stable comparisons
    ordered = sorted(facts, key=lambda x: (x[0],) + x[1])
    out: List[Dict[str, Any]] = []
    for pred, args in ordered:
        out.append({"predicate": pred, "args": list(args)})
    return out


def _add(facts: Set[Fact], pred: str, *args: str) -> None:
    facts.add((pred, tuple(args)))


def _rm(facts: Set[Fact], pred: str, *args: str) -> None:
    facts.discard((pred, tuple(args)))


def _is_clear(facts: Set[Fact], block: str) -> bool:
    # explicit clear fact or no block on top of it
    if ("clear", (block,)) in facts:
        return True
    for p, a in facts:
        if p == "on" and len(a) == 2 and a[1] == block:
            return False
    return True


def _holding_block(facts: Set[Fact]) -> Optional[str]:
    for p, a in facts:
        if p == "holding" and len(a) == 1:
            return a[0]
    return None


def _all_blocks(facts: Set[Fact]) -> Set[str]:
    blocks: Set[str] = set()
    for p, a in facts:
        if p == "on" and len(a) == 2:
            blocks.add(a[0])
            blocks.add(a[1])
        elif p in {"on-table", "ontable", "clear", "holding"} and len(a) >= 1:
            blocks.add(a[0])
    return blocks


def _copy(facts: Set[Fact]) -> Set[Fact]:
    return set(facts)


def _choose_ontable_name(facts: Set[Fact]) -> str:
    # Prefer the form present in the input facts
    for p, _ in facts:
        if p == "ontable":
            return "ontable"
        if p == "on-table":
            return "on-table"
    return "ontable"


def _on_table(facts: Set[Fact], x: str) -> bool:
    return ("ontable", (x,)) in facts or ("on-table", (x,)) in facts


def _apply_pickup(facts: Set[Fact], x: str) -> Optional[Set[Fact]]:
    # Preconditions: clear(x), on-table(x), handempty
    if not _is_clear(facts, x):
        return None
    if not _on_table(facts, x):
        return None
    if ("handempty", tuple()) not in facts:
        return None
    nf = _copy(facts)
    # remove whichever ontable variant exists
    _rm(nf, "ontable", x)
    _rm(nf, "on-table", x)
    _rm(nf, "clear", x)  # many formulations remove clear(x) when holding
    _rm(nf, "handempty")
    _add(nf, "holding", x)
    return nf


def _apply_unstack(facts: Set[Fact], x: str, y: str) -> Optional[Set[Fact]]:
    # Preconditions: clear(x), on(x,y), handempty
    if not _is_clear(facts, x):
        return None
    if ("on", (x, y)) not in facts:
        return None
    if ("handempty", tuple()) not in facts:
        return None
    nf = _copy(facts)
    _rm(nf, "on", x, y)
    _rm(nf, "clear", x)
    _rm(nf, "handempty")
    _add(nf, "clear", y)
    _add(nf, "holding", x)
    return nf


def _apply_putdown(facts: Set[Fact], x: str) -> Optional[Set[Fact]]:
    # Preconditions: holding(x)
    if ("holding", (x,)) not in facts:
        return None
    nf = _copy(facts)
    _rm(nf, "holding", x)
    ontable_name = _choose_ontable_name(facts)
    _add(nf, ontable_name, x)
    _add(nf, "clear", x)
    _add(nf, "handempty")
    return nf


def _apply_stack(facts: Set[Fact], x: str, y: str) -> Optional[Set[Fact]]:
    # Preconditions: holding(x) and clear(y) and x != y
    if x == y:
        return None
    if ("holding", (x,)) not in facts:
        return None
    if not _is_clear(facts, y):
        return None
    nf = _copy(facts)
    _rm(nf, "holding", x)
    _rm(nf, "clear", y)
    _add(nf, "on", x, y)
    _add(nf, "clear", x)
    _add(nf, "handempty")
    return nf


def successor(state_obj: Any) -> List[List[Dict[str, Any]]]:
    """Generate all valid successor states for BlocksWorld.

    Input state format:
      { "state": [ {"predicate": str, "args": [str, ...]}, ... ] }

    Predicates supported: on(x,y), on-table(x), clear(x), holding(x), handempty().
    """
    if isinstance(state_obj, dict):
        facts = _normalize_facts(state_obj.get("state", []))
    else:
        facts = _normalize_facts(state_obj or [])

    # We keep (action, successor_facts) pairs so we can return
    # both the action description and the resulting state.
    succs: List[Tuple[str, Set[Fact]]] = []
    seen: Set[Tuple[Fact, ...]] = set()

    held = _holding_block(facts)
    blocks = _all_blocks(facts)

    if held is None and ("handempty", tuple()) in facts:
        # pick-up for blocks on table
        for x in blocks:
            nf = _apply_pickup(facts, x)
            if nf is not None:
                key = tuple(sorted(nf))
                if key not in seen:
                    seen.add(key)
                    succs.append((f"pick-up {x}", nf))

        # unstack for blocks on others
        for p, a in list(facts):
            if p == "on" and len(a) == 2:
                x, y = a
                nf = _apply_unstack(facts, x, y)
                if nf is not None:
                    key = tuple(sorted(nf))
                    if key not in seen:
                        seen.add(key)
                        succs.append((f"unstack {x} {y}", nf))
    else:
        # If holding something, we can putdown or stack
        if held is not None:
            # putdown
            nf = _apply_putdown(facts, held)
            if nf is not None:
                key = tuple(sorted(nf))
                if key not in seen:
                    seen.add(key)
                    succs.append((f"put-down {held}", nf))

            # stack on any clear block y != held
            for y in blocks:
                if y == held:
                    continue
                nf = _apply_stack(facts, held, y)
                if nf is not None:
                    key = tuple(sorted(nf))
                    if key not in seen:
                        seen.add(key)
                        succs.append((f"stack {held} {y}", nf))

    # Convert to output format: list of (action, state_list)
    return [(action, _denormalize_facts(s)) for action, s in succs]
