import ast
import json
import os
import re
import logging
from typing import Any

def setup_logging():
    """Setup logging configuration to save logs to 'logs/' folder with timestamp."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

DOMAIN_DESCRIPTIONS = {
        "blocksworld": """This is a blocksworld domain where blocks can be placed on top of each other or on the table. 
        There is one robotic arm that can manipulate blocks. The arm can pick up a block (if it's clear and 
        the arm is empty), put down a block (if the arm is holding it), stack a block on another block 
        (if the target block is clear and the arm is holding a block), and unstack a block from another 
        block (if the block is clear and the arm is empty). Key constraints: only one block can be on top 
        of another, the arm can hold at most one block, and only clear blocks (with nothing on top) can be moved.
        The 'clear' predicate indicates a block has nothing on top of it, 'on' indicates one block is directly 
        on another, 'ontable' means a block is directly on the table, 'holding' means the arm is holding a block, 
        and 'handempty' means the arm is not holding anything.""",
        
        "ferry": """This is a ferry transportation domain where a ferry moves cars between different locations. 
        The ferry can sail between locations, and cars can board (embark) and disembark from the ferry. 
        Key constraints: the ferry can only be at one location at a time, cars can only board the ferry 
        when both the car and ferry are at the same location, and cars can only disembark when the ferry 
        is at their destination. The 'at' predicate indicates a car's location, 'at-ferry' shows the ferry's 
        current location, 'on' means a car is on the ferry, 'empty-ferry' indicates the ferry has no cars, 
        and 'not-eq' establishes that locations are distinct from each other.""",
        
        "logistics": """This is a logistics domain involving packages, trucks, airplanes, and locations. 
        Packages need to be transported between cities using trucks (for local transport within a city) 
        and airplanes (for transport between cities). Key constraints: trucks can only operate within 
        their home city, airplanes can fly between airports, packages must be loaded onto vehicles 
        before transport, and vehicles must be at the same location as packages to load them. 
        The 'at' predicate shows locations of objects, 'in' indicates a package is in a vehicle, 
        'in-city' establishes which city a location belongs to.""",
        
        "grippers": """This is a grippers domain where robots with grippers move balls between rooms. 
        Each robot has two grippers and can carry at most two balls simultaneously. Robots can move 
        between rooms, pick up balls (if they have a free gripper and are in the same room), and 
        drop balls in rooms. Key constraints: each gripper can hold at most one ball, robots must 
        be in the same room as a ball to pick it up, and balls can only be in one location at a time. 
        The 'at' predicate indicates locations, 'carry' shows which gripper is holding which ball, 
        and 'free' indicates an available gripper.""",
        
        "rovers": """This is a planetary rover domain where rovers navigate terrain to collect samples and data. 
        Rovers can move between waypoints, take images with cameras, collect soil/rock samples, and transmit 
        data to landers. Key constraints: rovers have limited battery and storage, some waypoints may be 
        unreachable due to terrain, and certain instruments are required for specific objectives. 
        The 'at' predicate shows rover locations, 'have_soil_analysis' indicates collected samples, 
        'communicated_soil_data' shows transmitted information.""",
        
        "visitall": """This is a visit-all domain where an agent must visit every location exactly once. 
        The agent can move between connected locations but cannot revisit locations they've already been to. 
        Key constraints: each location can only be visited once, the agent can only move between directly 
        connected locations, and the goal is typically to visit all locations. The 'at' predicate shows 
        the agent's current location, 'visited' indicates which locations have been visited, and 'connected' 
        defines valid movement paths between locations.""",
        
        "grid": """This is a grid navigation domain where an agent moves on a rectangular grid to reach 
        target positions or collect objects. The agent can move up, down, left, or right to adjacent 
        grid cells. Key constraints: the agent cannot move outside the grid boundaries, some cells may 
        be blocked or contain obstacles, and movement is typically to adjacent cells only. The 'at' 
        predicate indicates the agent's position, 'adjacent' defines valid moves between grid cells.""",
        
        "floortile": """This is a floor tiling domain where robots paint tiles on a floor in specific colors 
        and patterns. Robots can move between adjacent tiles and paint tiles they are standing on. 
        Key constraints: robots can only paint the tile they are currently on, some tiles may already 
        be painted and cannot be changed, and robots must coordinate to avoid conflicts. The 'robot-at' 
        predicate shows robot locations, 'painted' indicates which tiles have been painted with which colors, 
        and 'adjacent' defines movement possibilities between tiles.""",

        "alfworld": """This is an embodied household-task domain adapted from ALFWorld/ALFRED, 
        where an agent interacts with everyday objects inside indoor environments such as kitchens, 
        bedrooms, and living rooms. The agent can navigate rooms, open and close containers (like drawers, 
        cabinets, fridges), pick up and put down objects, toggle appliances on or off, and place items in 
        target receptacles. Key constraints: the agent must be in the same room and within reach of an object 
        to manipulate it; only open containers can be accessed; and objects have unique types and allowed 
        receptacles. The 'at' predicate represents an agent’s or object’s location, 'in' indicates containment, 
        'open' shows whether a container is open, and 'holding' indicates the agent is carrying an object.""",

        "depot": """This is a supply-depot management domain where pallets, crates, and hoists interact 
        within warehouses. Crates are stored on pallets, hoists can lift and move crates, trucks arrive to 
        load crates, and pallets can stack crates in restricted configurations. Key constraints: a hoist can 
        carry at most one crate, crates can only be moved when fully supported, and trucks must be at a docking 
        bay to load or unload goods. The 'on' predicate indicates crate stacking, 'at' shows the locations of 
        trucks or hoists, 'in' indicates crates loaded onto trucks, and 'lifting' shows when a hoist is holding 
        a crate.""",

        "goldminer": """This is a mining-and-resource-collection domain where a miner navigates tunnels to 
        extract gold chunks and deliver them to a safe location. The agent can move between adjacent tunnel 
        cells, dig to uncover hidden gold, pick up gold pieces, and drop them at designated collection sites. 
        Key constraints: digging may be required before gold becomes accessible, the miner can carry only a 
        limited number of gold pieces at once, and some cells may be blocked. The 'at' predicate shows the 
        miner’s location, 'gold-at' indicates the location of gold pieces, and 'carrying' specifies which gold 
        items the miner currently holds.""",

        "satellite": """This is a satellite-imaging domain where satellites capture images of celestial 
        targets using specialized instruments. Satellites can turn to face different directions, calibrate 
        their instruments using calibration targets, and take images once calibration is valid. Key constraints: 
        instruments must be calibrated immediately before use, satellites can only point in one direction at a 
        time, and some satellites have restricted slewing capabilities. The 'pointing' predicate indicates the 
        satellite's current orientation, 'calibrated' shows whether an instrument is currently calibrated, and 
        'have-image' indicates that an image of a target has been successfully taken.""",

        "swap": """This is a swapping domain where agents exchange the positions of objects across a set of 
        locations. Objects occupy unique slots, and the agent can swap the contents of two slots if both are 
        accessible. Key constraints: each location can hold exactly one object; swaps are atomic (a pairwise 
        exchange); and some locations may not be directly swappable unless intermediate steps are taken. 
        The 'at' predicate shows which object is in which location, and 'swappable' indicates whether two 
        positions can be legally swapped in a single action.""",

        "frogs_jumping": """This is a puzzle domain where frogs must swap positions on a line of lilypads. 
        There are left-facing frogs and right-facing frogs. Frogs can slide to an adjacent empty lilypad 
        or jump over an adjacent frog to an empty lilypad. Key constraints: frogs can only move in their 
        facing direction, and jumps require a specific configuration of frogs and empty spaces. The 'at' 
        predicate shows a frog's location, 'empty' indicates an unoccupied lilypad, and 'next' defines 
        the linear order of lilypads.""",

        "hanoi": """This is the classic Towers of Hanoi domain where disks of different sizes are stacked 
        on pegs. Disks can be moved one at a time from the top of one stack to another. Key constraints: 
        a larger disk cannot be placed on top of a smaller disk, and only the top disk of a stack can be 
        moved. The 'on' predicate shows which disk/peg is on top of another, 'smaller' defines the size 
        relationship allowing placement, and 'clear' indicates a disk has nothing on top of it."""
    }

def convert_json_str_to_json(json_str: str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def parse_json(json_str: str) -> Any:
    """Parse ```json ``` using regex"""
    pattern = r"```json(.*?)```"
    match = re.search(pattern, json_str, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            return None
    return None

def parse_code(code_str: str) -> str:
    """Parse ```python ``` using regex"""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_str.strip()

def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments without touching quoted strings."""
    result = []
    in_string = False
    string_char = ''
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            result.append(ch)
            if ch == string_char and (i == 0 or text[i - 1] != '\\'):
                in_string = False
            i += 1
            continue
        if ch in {'"', "'"}:
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == '/':
                i += 2
                while i < len(text) and text[i] not in {'\n', '\r'}:
                    i += 1
                continue
            if nxt == '*':
                i += 2
                while i + 1 < len(text) and not (text[i] == '*' and text[i + 1] == '/'):
                    i += 1
                i += 2
                continue
        result.append(ch)
        i += 1
    return ''.join(result)


def robust_json_parse(obj):
    """Recursively parse JSON-like strings, forgiving comments and Python literals."""
    while isinstance(obj, str):
        text = obj.strip()
        if not text:
            break
        if text.startswith("```"):
            # Remove markdown code blocks
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            obj = json.loads(text)
            continue
        except json.JSONDecodeError:
            cleaned = _strip_json_comments(text)
            if cleaned != text:
                obj = cleaned
                continue
            try:
                obj = ast.literal_eval(text)
            except Exception:
                break
        except Exception:
            break
    return obj
