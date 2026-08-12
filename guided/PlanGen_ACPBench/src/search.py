"""
Search for a goal state in a planning domain using generated successor and is_goal functions.
using BFS for simplicity. Can be extended to other search strategies.

usage:
python src/search.py --domain blocksworld  --timeout 20
"""

import argparse
import importlib.util
import os
import time
import json
import logging
from src.utils import robust_json_parse, setup_logging

class Search:
    def __init__(self, domain, initial_state_json_str, goal_state_json_str, timeout=600):
        self.domain = domain
        # Ensure initial_state and goal_state are lists of dicts, not JSON strings or dicts
        self.initial_state = robust_json_parse(initial_state_json_str)
        self.goal_state = robust_json_parse(goal_state_json_str)
        self.timeout = timeout
        self.succ_func = self._import_function(os.path.join('src', domain, 'succ.py'), 'successor')
        self.is_goal_func = self._import_function(os.path.join('src', domain, 'is_goal.py'), 'is_goal')
        self.heuristic_func = self._import_heuristic(os.path.join('src', domain, 'heuristics.py'))

    def _import_function(self, file_path, func_name, allow_missing=False):
        if not os.path.exists(file_path):
            if allow_missing:
                return None
            raise FileNotFoundError(f"Expected file {file_path} for domain '{self.domain}'")
        spec = importlib.util.spec_from_file_location(func_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    def _import_heuristic(self, file_path):
        default = lambda *_: 0
        module = None
        spec = None
        if os.path.exists(file_path):
            spec = importlib.util.spec_from_file_location('heuristics', file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            return default

        candidate_names = [
            'heuristic',
            'compute_heuristic',
            f"{self.domain}_heuristic",
            f"{self.domain}_cost",
            'heuristic_cost',
        ]

        # Domain-specific fallbacks
        domain_specific = {
            'blocksworld': ['blocksworld9_cost'],
        }
        candidate_names = domain_specific.get(self.domain, []) + candidate_names

        for name in candidate_names:
            if module and hasattr(module, name):
                func = getattr(module, name)
                if callable(func):
                    return func
        return default

    def _state_as_list(self, state_obj):
        parsed = robust_json_parse(state_obj)
        if isinstance(parsed, dict) and 'state' in parsed:
            return parsed['state']
        return parsed

    def _state_as_dict(self, state_obj):
        parsed = robust_json_parse(state_obj)
        if isinstance(parsed, dict) and 'state' in parsed:
            return parsed
        return {'state': parsed}

    def _state_hash(self, state_obj):
        state = self._state_as_list(state_obj)
        try:
            hash(state)
            return state
        except TypeError:
            return json.dumps(state, sort_keys=True)

    def _call_heuristic(self, state_obj, goal_obj):
        try:
            state_dict = self._state_as_dict(state_obj)
        except Exception:
            state_dict = state_obj
        goal_list = self._state_as_list(goal_obj)
        # Try common call signatures
        try:
            value = self.heuristic_func(state_dict, goal_list)
        except TypeError:
            try:
                value = self.heuristic_func(self._state_as_list(state_obj), goal_list)
            except TypeError:
                try:
                    value = self.heuristic_func(state_dict)
                except Exception:
                    return 0
        except Exception:
            return 0
        if value is None:
            return 0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0

    def search_bfs(self, return_plan=False):
        from collections import deque
        visited = set()
        queue = deque()
        # Each element in queue: (state, path)
        queue.append((self.initial_state, []))
        visited.add(json.dumps(self.initial_state, sort_keys=True))
        start_time = time.time()
        while queue:
            if time.time() - start_time > self.timeout:
                logging.info("Search timed out.")
                if return_plan:
                    return False, [], {"reason": "timeout"}
                return False
            state, path = queue.popleft()
            # If state is a dict with 'state' key, extract the underlying state list
            state_for_goal = robust_json_parse(state)
            if isinstance(state_for_goal, dict) and 'state' in state_for_goal:
                state_for_goal = state_for_goal['state']
            goal_for_goal = robust_json_parse(self.goal_state)
            # Debug: print types and samples
            logging.info(f"DEBUG: state_for_goal type: {type(state_for_goal)}, sample: {str(state_for_goal)[:200]}")
            logging.info(f"DEBUG: goal_for_goal type: {type(goal_for_goal)}, sample: {str(goal_for_goal)[:200]}")
            try:
                result = self.is_goal_func(state_for_goal, goal_for_goal)
            except Exception as e:
                logging.error(f"Error in is_goal function: {e}")
                result = False
                # raise
            if result:
                logging.info("Goal found!")
                if return_plan:
                    return True, path, {"reason": "success"}
                return True
            try:
                # For successor, always pass the underlying state list
                successors = self.succ_func(state_for_goal)
                # Sort successors for deterministic behavior
                successors.sort(key=lambda x: json.dumps(x, sort_keys=True))
            except Exception as e:
                logging.error(f"Error in successor function: {e}")
                continue
            for succ in successors:
                succ_parsed = robust_json_parse(succ)
                # Handle (action, state) tuples from new succ.py
                if isinstance(succ_parsed, (list, tuple)) and len(succ_parsed) == 2 and isinstance(succ_parsed[0], str):
                    succ_parsed = {'action': succ_parsed[0], 'state': succ_parsed[1]}
                # If successor is a dict with 'state', use the dict for queueing (to preserve action info), but hash on the state list
                if isinstance(succ_parsed, dict) and 'state' in succ_parsed:
                    hashable_state = succ_parsed['state']
                else:
                    hashable_state = succ_parsed
                succ_str = json.dumps(hashable_state, sort_keys=True)
                if succ_str not in visited:
                    visited.add(succ_str)
                    # If action is present, add to path
                    if isinstance(succ_parsed, dict) and 'action' in succ_parsed:
                        queue.append((succ_parsed, path + [succ_parsed['action']]))
                    else:
                        queue.append((succ_parsed, path))
        logging.info("Goal not found. All states exhausted.")
        if return_plan:
            return False, [], {"reason": "exhausted"}
        return False

    def search_A_star(self, return_plan=False):
        import heapq

        goal_list = self._state_as_list(self.goal_state)
        visited_costs = {self._state_hash(self.initial_state): 0}
        counter = 0
        heap = []
        start_time = time.time()

        initial_h = self._call_heuristic(self.initial_state, self.goal_state)
        heapq.heappush(heap, (initial_h, counter, self.initial_state, [], 0))

        while heap:
            if time.time() - start_time > self.timeout:
                logging.info("Search timed out.")
                if return_plan:
                    return False, [], {"reason": "timeout"}
                return False

            priority, _, current_state, path, g_cost = heapq.heappop(heap)
            current_hash = self._state_hash(current_state)
            if g_cost > visited_costs.get(current_hash, float('inf')):
                continue

            state_list = self._state_as_list(current_state)
            try:
                if self.is_goal_func(state_list, goal_list):
                    print("Goal found!")
                    if return_plan:
                        return True, path, {"reason": "success"}
                    return True
            except Exception as e:
                logging.error(f"Error in is_goal function: {e}")
                # raise

            try:
                successors = self.succ_func(state_list)
                # Sort successors for deterministic behavior
                successors.sort(key=lambda x: json.dumps(x, sort_keys=True))
            except Exception as e:
                print(f"Error in successor function: {e}")
                continue

            for succ in successors:
                succ_parsed = robust_json_parse(succ)
                # Handle (action, state) tuples from new succ.py
                if isinstance(succ_parsed, (list, tuple)) and len(succ_parsed) == 2 and isinstance(succ_parsed[0], str):
                    succ_parsed = {'action': succ_parsed[0], 'state': succ_parsed[1]}
                succ_hash = self._state_hash(succ_parsed)
                new_path = path.copy()
                if isinstance(succ_parsed, dict) and 'action' in succ_parsed:
                    new_path = new_path + [succ_parsed['action']]
                new_g = g_cost + 1  # assuming unit step cost

                if new_g < visited_costs.get(succ_hash, float('inf')):
                    visited_costs[succ_hash] = new_g
                    h_value = self._call_heuristic(succ_parsed, self.goal_state)
                    counter += 1
                    heapq.heappush(heap, (new_g + h_value, counter, succ_parsed, new_path, new_g))

        logging.info("Goal not found. All states exhausted.")
        if return_plan:
            return False, [], {"reason": "exhausted"}
        return False

    def search_greedy_bfs(self, return_plan=False):
        import heapq

        goal_list = self._state_as_list(self.goal_state)
        # For Greedy BFS, we just need to track visited states, not necessarily costs.
        # But using a set of hashes is enough.
        visited = set()
        visited.add(self._state_hash(self.initial_state))
        
        counter = 0
        heap = []
        start_time = time.time()

        initial_h = self._call_heuristic(self.initial_state, self.goal_state)
        # Priority is just h_value
        heapq.heappush(heap, (initial_h, counter, self.initial_state, []))

        while heap:
            if time.time() - start_time > self.timeout:
                logging.info("Search timed out.")
                if return_plan:
                    return False, [], {"reason": "timeout"}
                return False

            priority, _, current_state, path = heapq.heappop(heap)

            state_list = self._state_as_list(current_state)
            try:
                if self.is_goal_func(state_list, goal_list):
                    print("Goal found!")
                    if return_plan:
                        return True, path, {"reason": "success"}
                    return True
            except Exception as e:
                logging.error(f"Error in is_goal function: {e}")
                # Treat as not a goal instead of crashing
                pass

            try:
                successors = self.succ_func(state_list)
                # Sort successors for deterministic behavior
                successors.sort(key=lambda x: json.dumps(x, sort_keys=True))
            except Exception as e:
                print(f"Error in successor function: {e}")
                continue

            for succ in successors:
                succ_parsed = robust_json_parse(succ)
                # Handle (action, state) tuples from domain successor functions
                if (
                    isinstance(succ_parsed, (list, tuple))
                    and len(succ_parsed) == 2
                    and isinstance(succ_parsed[0], str)
                ):
                    succ_parsed = {"action": succ_parsed[0], "state": succ_parsed[1]}

                succ_hash = self._state_hash(succ_parsed)

                if succ_hash in visited:
                    continue

                visited.add(succ_hash)

                new_path = path.copy()
                if isinstance(succ_parsed, dict) and "action" in succ_parsed:
                    new_path = new_path + [succ_parsed["action"]]

                h_value = self._call_heuristic(succ_parsed, self.goal_state)
                counter += 1
                heapq.heappush(heap, (h_value, counter, succ_parsed, new_path))

        logging.info("Goal not found. All states exhausted.")
        if return_plan:
            return False, [], {"reason": "exhausted"}
        return False

    def search(self, strategy='bfs', return_plan=True):
        strategy = strategy.lower()
        if strategy == 'bfs':
            return self.search_bfs(return_plan=return_plan)
        if strategy in {'a*', 'astar', 'a-star'}:
            return self.search_A_star(return_plan=return_plan)
        if strategy in {'greedy', 'greedy_bfs', 'greedy-bfs'}:
            return self.search_greedy_bfs(return_plan=return_plan)
        raise ValueError(f"Unknown search strategy '{strategy}'")

def main():
    parser = argparse.ArgumentParser(description="Search for a goal state in a planning domain using generated successor and is_goal functions.")
    parser.add_argument('--domain', type=str, required=True, help='Domain name (e.g., blocksworld, ferry, etc.)')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout for search in seconds (default: 600)')
    parser.add_argument('--strategy', type=str, default='astar', choices=['bfs', 'a*', 'astar', 'a-star', 'greedy', 'greedy_bfs', 'greedy-bfs'],
                        help='Search strategy to use (default: bfs)')
    parser.add_argument('--initial_state', type=str, help='Initial state as JSON string')
    parser.add_argument('--goal_state', type=str, help='Goal state as JSON string')
    parser.add_argument('--src', default='./src',type=str, help='Path to nl2state_result.json for batch search')
    parser.add_argument('--nl2state', default='nl2state_result.json', type=str, help='Input nl2state_result.json filename')
    parser.add_argument('--out', default='search_result.json', type=str, help='Path to save search results')
    args = parser.parse_args()

    setup_logging()
    
    nl2state_result_path = os.path.join(args.src, args.domain ,args.nl2state)
    search_result_path = os.path.join(args.src, args.domain, args.out)
    if os.path.exists(nl2state_result_path):
        # Batch mode: run search for each datapoint in nl2state_result.json
        with open(nl2state_result_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Running search for all datapoints in {nl2state_result_path}...")
        results = []
        for item in data.get('train_results', []):
            initial_state = item['predicted_initial_state']
            goal_state = item['predicted_goal_state']
            searcher = Search(args.domain, initial_state, goal_state, timeout=args.timeout)
            found, plan, info = searcher.search(strategy=args.strategy, return_plan=True)
            results.append({
                'example_id': item.get('example_id'),
                'initial_state': initial_state,
                'goal_state': goal_state,
                'plan': plan,
                'final_answer': 'yes' if found else 'no',
                'failure_reason': info.get('reason') if not found else None
            })
        with open(search_result_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Batch search results saved to {search_result_path}")
        for r in results:
            logging.info(r)
    else:
        # Single datapoint mode
        if not args.initial_state or not args.goal_state:
            logging.error("You must provide --initial_state and --goal_state as JSON strings if not using --nl2state_result.")
            return
        searcher = Search(args.domain, args.initial_state, args.goal_state, timeout=args.timeout)
        found, plan, info = searcher.search(strategy=args.strategy, return_plan=True)
        result = {
            'initial_state': args.initial_state,
            'goal_state': args.goal_state,
            'plan': plan,
            'final_answer': 'yes' if found else 'no',
            'failure_reason': info.get('reason') if not found else None
        }
        with open(search_result_path, 'w') as f:
            json.dump([result], f, indent=2)
        logging.info(f"Search result saved to {search_result_path}")
        logging.info(result)

if __name__ == "__main__":
    exit(main())
