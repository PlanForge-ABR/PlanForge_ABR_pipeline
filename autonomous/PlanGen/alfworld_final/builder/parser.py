"""Natural-language parser for ACPBench ALFWorld instances."""

import re
from typing import Dict, Iterable, List, Tuple

from architect.spec import AWGoals, AWState, DirectGoal, ValidationGoal


OBJ_RE = r"[a-z]+[a-z0-9]*\d+"
LOC_RE = r"location\d+"


def parse_instance(context: str, inputs: str) -> Tuple[AWState, AWGoals]:
    text = _norm(context)
    object_types, receptacle_types = _parse_declared_types(text)
    receptacle_locations = _parse_receptacle_locations(text)
    object_locations = _parse_object_locations(text)
    object_receptacle = _parse_object_receptacles(text)
    agent_location = _parse_agent_location(text)
    open_receptacles, closed_receptacles = _parse_open_closed(text, set(receptacle_locations))
    toggled_objects = _parse_toggled(text)
    validated = _parse_validated(text)
    holding = _parse_holding(text)

    for receptacle in receptacle_locations:
        receptacle_types.setdefault(_name_to_type(receptacle), []).append(receptacle)
    for obj in set(object_locations) | set(object_receptacle):
        if obj not in receptacle_locations:
            object_types.setdefault(_name_to_type(obj), []).append(obj)

    locations = set(re.findall(LOC_RE, text))
    state = AWState(
        agent_location=agent_location,
        locations=locations,
        objects_by_type=_dedupe_lists(object_types),
        receptacles_by_type=_dedupe_lists(receptacle_types),
        object_locations=object_locations,
        receptacle_locations=receptacle_locations,
        object_receptacle=object_receptacle,
        open_receptacles=open_receptacles,
        closed_receptacles=closed_receptacles,
        toggled_objects=toggled_objects,
        validated=validated,
        holding=holding,
    )
    return state, parse_goals(inputs)


def parse_goals(inputs: str) -> AWGoals:
    goal_text = _norm(inputs).split("holds:", 1)[-1].strip().rstrip("?")
    goals = AWGoals()
    validation_spans = []

    for match in re.finditer(
        r"It has been validated that (?P<count>an object|two objects) of type (?P<otype>[a-z0-9]+type)"
        r"(?: is (?P<prop>clean|hot|cool|examined) (?:and is in a receptacle of type (?P<rtype1>[a-z0-9]+type)|under an object of type (?P<tool>[a-z0-9]+type))|"
        r" are in a receptacle of type (?P<rtype2>[a-z0-9]+type)|"
        r" is in a receptacle of type (?P<rtype3>[a-z0-9]+type))",
        goal_text,
    ):
        count = 2 if match.group("count") == "two objects" else 1
        goals.validations.append(
            ValidationGoal(
                object_type=match.group("otype"),
                receptacle_type=match.group("rtype1") or match.group("rtype2") or match.group("rtype3"),
                property_name=match.group("prop"),
                count=count,
                tool_type=match.group("tool"),
            )
        )
        validation_spans.append(match.span())

    consumed_parts = []
    cursor = 0
    for start, end in validation_spans:
        consumed_parts.append(goal_text[cursor:start])
        cursor = end
    consumed_parts.append(goal_text[cursor:])
    consumed = " ".join(consumed_parts)
    for fact in _split_facts(consumed):
        fact = fact.strip(" .?")
        if not fact:
            continue
        locs = re.findall(LOC_RE, fact)
        if re.search(r"agent agent1 is at location", fact) and locs:
            goals.direct.append(DirectGoal("agent_at", "agent1", locs[-1]))
            continue
        m = re.match(rf"({OBJ_RE}) is (?:in|on) ({OBJ_RE})$", fact)
        if m:
            goals.direct.append(DirectGoal("object_in_receptacle", m.group(1), m.group(2)))
            continue
        m = re.match(rf"({OBJ_RE}) is open$", fact)
        if m:
            goals.direct.append(DirectGoal("open", m.group(1)))
            continue
        m = re.match(rf"({OBJ_RE}|{LOC_RE}) is toggled$", fact)
        if m:
            goals.direct.append(DirectGoal("toggled", m.group(1)))

    return goals


def _parse_declared_types(text: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    obj_types: Dict[str, List[str]] = {}
    rec_types: Dict[str, List[str]] = {}
    m = re.search(r"There are .*? object types: (.*?), \d+ receptacle types: (.*?), and \d+ locations", text)
    if not m:
        return obj_types, rec_types
    for count, plural in re.findall(r"(\d+) ([a-z]+[a-z0-9]*)", m.group(1)):
        obj_types[_plural_to_type(plural)] = []
    for count, plural in re.findall(r"(\d+) ([a-z]+[a-z0-9]*)", m.group(2)):
        rec_types[_plural_to_type(plural)] = []
    return obj_types, rec_types


def _parse_receptacle_locations(text: str) -> Dict[str, str]:
    section = _between(text, "The receptacles are at locations as follows.", "Receptacles that are neither")
    result: Dict[str, str] = {}
    for sentence in section.split("."):
        m = re.search(rf"(.+?) (?:is|are) at ({LOC_RE})", sentence)
        if not m:
            continue
        for name in _names(m.group(1)):
            result[name] = m.group(2)
    return result


def _parse_object_locations(text: str) -> Dict[str, str]:
    section = _between(text, "Currently, the objects are at locations as follows.", "The objects are in/on receptacle")
    result: Dict[str, str] = {}
    for sentence in section.split("."):
        m = re.search(rf"(.+?) (?:is|are) at ({LOC_RE})", sentence)
        if not m:
            continue
        for name in _names(m.group(1)):
            if name != "agent1":
                result[name] = m.group(2)
    return result


def _parse_object_receptacles(text: str) -> Dict[str, str]:
    section = _between(text, "The objects are in/on receptacle as follows.", "agent1's hands")
    if not section:
        section = _between(text, "The objects are in/on receptacle as follows.", "")
    result: Dict[str, str] = {}
    for sentence in section.split("."):
        m = re.search(rf"(.+?) (?:is|are) (?:in|on) ({OBJ_RE})", sentence)
        if not m:
            continue
        rec = m.group(2)
        for name in _names(m.group(1)):
            result.setdefault(name, rec)
    return result


def _parse_agent_location(text: str) -> str:
    m = re.search(rf"agent agent1 is at location ({LOC_RE})", text)
    return m.group(1) if m else ""


def _parse_open_closed(text: str, receptacles: Iterable[str]):
    open_set = set()
    closed_set = set()
    for sentence in text.split("."):
        if " are closed" in sentence or " is closed" in sentence:
            for name in _names(sentence):
                if name in receptacles:
                    closed_set.add(name)
        if " are open" in sentence or " is open" in sentence:
            for name in _names(sentence):
                if name in receptacles:
                    open_set.add(name)
    return open_set, closed_set


def _parse_toggled(text: str):
    toggled = set()
    for sentence in text.split("."):
        if " is toggled" in sentence or " toggled" in sentence:
            toggled.update(_names(sentence))
    return toggled


def _parse_validated(text: str):
    if "Nothing has been validated" in text:
        return set()
    return {m.group(0) for m in re.finditer(r"validated that .*?(?=\.|$)", text)}


def _parse_holding(text: str):
    if "hands are empty" in text:
        return None
    m = re.search(r"agent1 is holding ([a-z0-9]+)", text)
    return m.group(1) if m else None


def _split_facts(text: str):
    return re.split(r"\s+and\s+|,\s*", text)


def _names(text: str):
    return re.findall(OBJ_RE, text)


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    tail = text.split(start, 1)[1]
    if end and end in tail:
        return tail.split(end, 1)[0]
    return tail


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u2019", "'")).strip()


def _plural_to_type(plural: str) -> str:
    if plural.endswith("ies"):
        singular = plural[:-3] + "y"
    elif plural.endswith("s") and not plural.endswith("ss"):
        singular = plural[:-1]
    else:
        singular = plural
    return f"{singular}type"


def _name_to_type(name: str) -> str:
    return re.sub(r"\d+$", "", name) + "type"


def _dedupe_lists(data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {k: sorted(set(v), key=_natural_key) for k, v in data.items()}


def _natural_key(name: str):
    m = re.match(r"([a-z]+)(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)
