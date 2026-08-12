import os
import re
import sys
import json
import logging
from openai import OpenAI

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL')

if base_url:
    client = OpenAI(api_key=openai_api_key, base_url=base_url)
else:
    client = OpenAI(api_key=openai_api_key)

token_usage = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "api_calls": 0
}

def calculate_cost(prompt_tokens, completion_tokens):
    # GPT-5.4 Pricing: $2.50 / 1M input tokens, $15.00 / 1M output tokens
    return (prompt_tokens * 2.50 + completion_tokens * 15.00) / 1000000.0

def query_llm(prompt, system_prompt=None, temperature=0.0):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=messages,
            temperature=temperature
        )
        token_usage["api_calls"] += 1
        usage = getattr(response, "usage", None)
        if usage:
            token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error querying OpenAI model gpt-5.4: {e}")
        raise e

def extract_content(text, start_tag, end_tag):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern_ic = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL | re.IGNORECASE)
    match_ic = pattern_ic.search(text)
    if match_ic:
        return match_ic.group(1).strip()
    return ""

def clean_code(code_str):
    code_str = code_str.strip()
    if code_str.startswith("```python"):
        code_str = code_str[len("```python"):].strip()
    elif code_str.startswith("```"):
        code_str = code_str[3:].strip()
    if code_str.endswith("```"):
        code_str = code_str[:-3].strip()
    return code_str

def safe_json_parse(json_str):
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):].strip()
    elif json_str.startswith("```"):
        json_str = json_str[3:].strip()
    if json_str.endswith("```"):
        json_str = json_str[:-3].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return eval(json_str)
        except Exception:
            return None

def extract_json_robust(text, start_tag, end_tag):
    content = extract_content(text, start_tag, end_tag)
    if content.strip():
        parsed = safe_json_parse(content)
        if parsed is not None:
            return parsed
            
    match_code = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match_code:
        parsed = safe_json_parse(match_code.group(1))
        if parsed is not None:
            return parsed
            
    match_braces = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_braces:
        parsed = safe_json_parse(match_braces.group(1))
        if parsed is not None:
            return parsed
            
    return {}

def extract_code_robust(text, start_tag, end_tag, func_name):
    code = extract_content(text, start_tag, end_tag)
    c = ""
    if code.strip():
        c = clean_code(code)
    else:
        match_code = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match_code:
            c = match_code.group(1).strip()
        else:
            match_any = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
            if match_any:
                c = match_any.group(1).strip()
            else:
                pattern = rf"(def\s+{func_name}\b.*?)(?=\n\S|\Z)"
                match_def = re.search(pattern, text, re.DOTALL)
                if match_def:
                    c = match_def.group(1).strip()
                elif f"def {func_name}" in text:
                    c = clean_code(text)
                else:
                    c = ""
                    
    # Normalize escaped newlines and quotes
    if "\\n" in c:
        if c.count("\\n") > c.count("\n"):
            c = c.replace("\\n", "\n")
    if '\\"' in c:
        c = c.replace('\\"', '"')
    return c

class SCOPESolverBuilder:
    def __init__(self, domain, q_ex, s_ex):
        self.domain = domain
        self.q_ex = q_ex
        self.s_ex = s_ex
        self.s_struc = None
        self.combinations_schema = None
        self.constraints_schema = None
        self.planning_instructions = None
        self.c_ex = None
        self.k_ex = None
        self.merged_ex = None
        self.combinations_func_code = None
        self.plan_func_code = None
        self.deliver_func_code = None

    def build(self):
        logging.info(f"🏗️ Starting Standalone SCOPE Solver construction for Natural Plan domain: {self.domain}...")
        start_prompt = token_usage["prompt_tokens"]
        start_comp = token_usage["completion_tokens"]
        start_api = token_usage.get("api_calls", 0)
        self.run_stage_i()
        self.run_stage_ii()
        logging.info("🎉 SCOPE Solver construction completed successfully!")
        end_prompt = token_usage["prompt_tokens"]
        end_comp = token_usage["completion_tokens"]
        end_api = token_usage.get("api_calls", 0)
        build_prompt_tokens = end_prompt - start_prompt
        build_completion_tokens = end_comp - start_comp
        build_api_calls = end_api - start_api
        build_cost = calculate_cost(build_prompt_tokens, build_completion_tokens)
        return {
            "combinations_func_code": self.combinations_func_code,
            "plan_func_code": self.plan_func_code,
            "deliver_func_code": self.deliver_func_code,
            "c_ex": self.c_ex,
            "k_ex": self.k_ex,
            "merged_ex": self.merged_ex,
            "q_ex": self.q_ex,
            "s_ex": self.s_ex,
            "s_struc": self.s_struc,
            "planning_instructions": self.planning_instructions,
            "build_prompt_tokens": build_prompt_tokens,
            "build_completion_tokens": build_completion_tokens,
            "build_api_calls": build_api_calls,
            "build_cost": build_cost
        }

    def run_stage_i(self):
        logging.info("--- Stage I: Query-Specific Problem Reasoning ---")
        
        # 1. Solution Agent
        logging.info("Running Solution Agent...")
        sol_prompt = f"""You are a Solution Agent.
You are given a natural language query and its corresponding answer solution (the correct plan or schedule).
Based on the answer solution, you have to create a structured JSON representation for the solution.
Please use as few keys as possible to represent the solution, but ensure that the structured format contains all necessary details to fully reconstruct the original Answer Solution.

Query:
{self.q_ex}

Answer Solution:
{self.s_ex}

Your output must contain only the structured solution representation inside <start_of_structured_output> and <end_of_structured_output>. Do not use '...' to represent values; output the complete JSON representation of the solution.

Output format:
<start_of_structured_output>
{{
  "solutions": <structured solution representation, e.g., a dictionary, list, or tuple representing the plan/schedule>,
  "solutions_description": <A text description of the structured solution format>
}}
<end_of_structured_output>"""
        
        sol_out = query_llm(sol_prompt)
        self.s_struc = extract_json_robust(sol_out, "<start_of_structured_output>", "<end_of_structured_output>")
        logging.info(f"Structured Solution Representation: {self.s_struc}")

        # 2. Planning Agent
        logging.info("Running Planning Agent...")
        plan_prompt = f"""You are a combinations and constraints planning agent.
You are given an exemplar query and the structured representation of its solution (an ordered list of actions, cities, flights, or meeting times).
Based on the query and solution representation, identify:
1. "combinations": The set of parameters (representing the initial state context/relations of all objects, dates/days, distances, durations, or starting statuses) required to build a transition model or search space and exhaustively generate all potential candidate plan/schedule sequences. You must NOT omit the initial state configurations, since plan generation is impossible without knowing travel distance matrices, available slots, or cities to visit.
2. "constraints": The set of constraints (representing the goal query conditions and preferences) used to filter these candidate plan sequences.

CRITICAL RULE: Any parameter that specifies the dimensions, duration, or structure of the actions in the plan (such as meeting durations, city stay lengths, flight durations, start locations, and the list of entities/friends to meet) MUST be placed in "combinations" (not "constraints"). This is because plan generation (combinations_func) is mathematically impossible without knowing the target durations, stay lengths, or participants to generate steps for. Placing these in "constraints" will result in a solver that cannot generate candidates for queries with different durations or stay lengths.

CRITICAL SCHEMA RULE: The "constraints" schema must ALWAYS include a "preferences" key (which will hold any participant preferences or constraints like "rather not meet after/before time", even if they are empty in the exemplar query) so that the generated solver is general enough to handle queries with preferences.

Exemplar Query:
{self.q_ex}

Structured Solution Format:
{self.s_struc}

Output format:
<start_of_COT>
<COT analysis of parameters and how to solve the domain>
<end_of_COT>
<start_of_structured_output>
{{
  "combinations": <dictionary specifying the combination parameter structures>,
  "constraints": <dictionary specifying the constraint parameter structures>,
  "combinations_description": <description of combination parameters>,
  "constraints_description": <description of constraint parameters>
}}
<end_of_structured_output>
<start_of_planning>
{{
  "Input Agent": <instructions for Input Agent to extract combination and constraint parameter values from a query>,
  "Combination Function Generator Agent": <instructions for Combination Function Generator Agent to generate combinations_func() code>,
  "Filter Function Generator Agent": <instructions for Filter Function Generator Agent to generate plan_func() code>
}}
<end_of_planning>"""

        plan_out = query_llm(plan_prompt)
        struc_out = extract_json_robust(plan_out, "<start_of_structured_output>", "<end_of_structured_output>")
        self.combinations_schema = struc_out.get("combinations")
        self.constraints_schema = struc_out.get("constraints")
        self.planning_instructions = extract_json_robust(plan_out, "<start_of_planning>", "<end_of_planning>")

        # 3. Optimization Agent 1: Filter Unnecessary Parameters
        logging.info("Running Optimization Agent 1 (Filter Unnecessary)...")
        opt1_prompt = f"""You are an Optimization Agent. Your job is to filter out unnecessary parameters of the output of Planning Agent.
We define parameter as each key in the "combinations" and "constraints" from structured output. If a parameter has no discriminative power (i.e. it applies uniformly across all candidate plans, causing either all plans to pass or all plans to fail, and is not represented in list or JSON), it must be removed.
IMPORTANT: You must NOT remove or filter out parameters that represent the initial state configuration of the world (like flight connectivity, distance matrix, durations, time ranges, calendars, etc.). Although these parameters might look constant for a single instance, they vary across instances and are absolutely necessary for simulating state transitions and generating valid plan candidates. Only remove parameters that are completely redundant or have no role in either plan generation or goal filtering.

Here is the structured output of Planning Agent:
{json.dumps(struc_out, indent=2)}

Here is the planning instructions of Planning Agent:
{json.dumps(self.planning_instructions, indent=2)}

Output format:
<start_of_COT>
<COT analysis>
<end_of_COT>
<start_of_structured_output>
<optimized structured output JSON>
<end_of_structured_output>
<start_of_planning>
<optimized planning instructions JSON>
<end_of_planning>"""

        opt1_out = query_llm(opt1_prompt)
        struc_out = extract_json_robust(opt1_out, "<start_of_structured_output>", "<end_of_structured_output>")
        self.planning_instructions = extract_json_robust(opt1_out, "<start_of_planning>", "<end_of_planning>")

        # 4. Optimization Agent 2: Fix Combinations Mistaken for Constraints
        logging.info("Running Optimization Agent 2 (Fix combinations/constraints)...")
        opt2_prompt = f"""You are an Optimization Agent. Your job is to check if any constraints in "constraints" are actually combination parameters and should be moved to "combinations" (e.g. travel matrices, schedules, list of participants that must be met in all candidate plan sequences).

CRITICAL RULE: Verify that all parameters specifying target durations, stay lengths, meeting durations, flight connections, and participants are placed in "combinations". If any of these are in "constraints", they must be moved to "combinations" because combinations_func must generate plans matching these structural properties dynamically.

Here is the structured output:
{json.dumps(struc_out, indent=2)}

Here is the planning instructions:
{json.dumps(self.planning_instructions, indent=2)}

Output format: Same as Optimization Agent 1."""

        opt2_out = query_llm(opt2_prompt)
        struc_out = extract_json_robust(opt2_out, "<start_of_structured_output>", "<end_of_structured_output>")
        self.planning_instructions = extract_json_robust(opt2_out, "<start_of_planning>", "<end_of_planning>")

        # 5. Optimization Agent 3: Validates and Expands Parameters
        logging.info("Running Optimization Agent 3 (Validate/Expand)...")
        opt3_prompt = f"""You are an Optimization Agent. Your job is to check if the parameters are expandable and need expansion (e.g. flight connectivity symmetry, calendar time ranges, objects, etc. to make plan generation complete).

Here is the structured output:
{json.dumps(struc_out, indent=2)}

Here is the planning instructions:
{json.dumps(self.planning_instructions, indent=2)}

Output format: Same as Optimization Agent 1."""

        opt3_out = query_llm(opt3_prompt)
        struc_out = extract_json_robust(opt3_out, "<start_of_structured_output>", "<end_of_structured_output>")
        self.planning_instructions = extract_json_robust(opt3_out, "<start_of_planning>", "<end_of_planning>")
        
        self.combinations_schema = struc_out.get("combinations")
        self.constraints_schema = struc_out.get("constraints")

        # 6. Extract exemplar combinations and constraints values
        logging.info("Extracting combination and constraint values for the exemplar query...")
        input_agent_prompt = f"""You are the Input Agent.
You are given a query and instructions to extract combination and constraint parameter values in structured JSON.

Query:
{self.q_ex}

Instructions:
{self.planning_instructions.get("Input Agent", "")}

Output format:
<start_of_structured_output>
{{
  "combinations": <extracted combinations parameter values>,
  "constraints": <extracted constraints parameter values>
}}
<end_of_structured_output>"""

        input_out = query_llm(input_agent_prompt)
        input_json = extract_json_robust(input_out, "<start_of_structured_output>", "<end_of_structured_output>")
        self.c_ex = input_json.get("combinations")
        self.k_ex = input_json.get("constraints")
        self.merged_ex = {**(self.c_ex or {}), **(self.k_ex or {})}
        logging.info(f"Exemplar combinations (C_ex): {self.c_ex}")
        logging.info(f"Exemplar constraints (K_ex): {self.k_ex}")

    def run_stage_ii(self):
        logging.info("--- Stage II: Generic Solver Generation ---")
        
        # 1. Combinations Function Generator Agent
        logging.info("Generating combinations_func(data)...")
        comb_gen_prompt = f"""You are the Combination Function Generator Agent.
You are given the problem parameters (both combinations and constraints) of the exemplar query:
{json.dumps(self.merged_ex, indent=2)}

You are also given the expected structured solution format S:
{json.dumps(self.s_struc.get("solutions"), indent=2)}

Planning Instructions:
{self.planning_instructions.get("Combination Function Generator Agent", "")}

CRITICAL GENERALIZATION RULE: Your code must NEVER hardcode values or constants from the exemplar query parameters (such as hardcoding meeting_duration = 60, stay duration = 2, specific days, names, or list values). You must read all variables (such as meeting durations, stay durations, flight rules, or lists of participants) dynamically from the input dictionary 'data' (e.g., data.get("meeting_duration_minutes") or data.get("stay_days") or data.get("city_day_requirements")). If you hardcode exemplar constants, the solver will fail on other queries with different values!
Your combinations_func must explore the complete search space of ALL possible durations, layouts, permutations, or allocations (e.g., for stay day allocations, generate all integer partitions of total_days that sum up to total_days and satisfy minimal city stay requirements, rather than assuming a single even split) to guarantee that a matching candidate plan is generated for any query.

Your job is to write a generic Python function combinations_func(data) that takes the problem parameters 'data' (a dictionary containing combinations and constraints) and returns a list of candidate plans/schedules.
Each candidate plan must follow the format of S.
The function should exhaustively explore the search space of potential solutions (e.g., all valid flight/trip itinerary routes, all possible meeting time slots, or all possible permutations of travel/waiting/meeting actions) to produce the list of candidate plan/schedule sequences.
Use product, permutations, or other tools from python's 'itertools' if needed. Do not print or include example usage.

Output format:
<start_of_COT>
...
<end_of_COT>
<start_of_code>
<def combinations_func(data) Python code only>
<end_of_code>"""

        comb_out = query_llm(comb_gen_prompt)
        self.combinations_func_code = extract_code_robust(comb_out, "<start_of_code>", "<end_of_code>", "combinations_func")

        # Refinement Loop for combinations_func
        patience = 3
        Xi_sample = []
        for step in range(patience):
            logging.info(f"Verifying combinations_func (Refinement Step {step+1}/{patience})...")
            try:
                namespace = {}
                import itertools
                import collections
                import math
                import datetime
                namespace['itertools'] = itertools
                namespace['collections'] = collections
                namespace['math'] = math
                namespace['datetime'] = datetime
                exec(self.combinations_func_code, namespace)
                comb_func = namespace.get("combinations_func")
                if not comb_func:
                    raise NameError("combinations_func is not defined in generated code")
                
                Xi_sample = comb_func(self.merged_ex)
                target_sol = self.s_struc.get("solutions")
                
                # Check if target solution plan exists in candidate list
                found = False
                for plan in Xi_sample:
                    if plan == target_sol:
                        found = True
                        break
                
                if found:
                    logging.info("✅ combinations_func generates target plan successfully!")
                    break
                else:
                    error_msg = f"Generated candidate plans do not contain the target solution: {target_sol}. Total generated: {len(Xi_sample)} plans."
                    logging.warning(f"Refinement required: {error_msg}")
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                logging.error(f"Execution error in combinations_func:\n{error_msg}")
                Xi_sample = []

            # Call Reflection Agent
            logging.info("Running Combination Function Reflection Agent...")
            refl_prompt = f"""You are a Reflection Agent. Your job is to observe the mistake of the output of the Combination Function Generator Agent.
Here is the problem parameters (combinations and constraints):
{json.dumps(self.merged_ex, indent=2)}

Here is the expected ground-truth solution S:
{json.dumps(self.s_struc.get("solutions"), indent=2)}

Here is the generated combinations_func code:
{self.combinations_func_code}

Here is the error or output when running combinations_func(data):
{error_msg}

Please correct the thought process and the code to ensure combinations_func(data) correctly generates candidate plans containing the ground-truth solution.

CRITICAL GENERALIZATION RULE: Your code must NEVER hardcode values or constants from the exemplar query parameters (such as hardcoding meeting_duration = 60, stay duration = 2, specific days, names, or list values). You must read all variables (such as meeting durations, stay durations, flight rules, or lists of participants) dynamically from the input dictionary 'data' (e.g., data.get("meeting_duration_minutes") or data.get("stay_days") or data.get("city_day_requirements")). If you hardcode exemplar constants, the solver will fail on other queries with different values!
Your combinations_func must explore the complete search space of ALL possible durations, layouts, permutations, or allocations (e.g., for stay day allocations, generate all integer partitions of total_days that sum up to total_days and satisfy minimal city stay requirements, rather than assuming a single even split) to guarantee that a matching candidate plan is generated for any query.

Output format:
<start_of_COT_correction>
...
<end_of_COT_correction>
<start_of_code_correction>
<corrected combinations_func Python code>
<end_of_code_correction>"""
            
            refl_out = query_llm(refl_prompt)
            self.combinations_func_code = extract_code_robust(refl_out, "<start_of_code_correction>", "<end_of_code_correction>", "combinations_func")

        # 2. Filter Function Generator Agent
        logging.info("Generating plan_func(data, constraints)...")
        filter_gen_prompt = f"""You are the Filter Function Generator Agent.
You are given the constraints parameters format:
{json.dumps(self.k_ex, indent=2)}

You are also given the list of candidate plans generated by combinations_func (first 20 plans shown):
{json.dumps(Xi_sample[:20], indent=2)}

Planning Instructions:
{self.planning_instructions.get("Filter Function Generator Agent", "")}

Your job is to write a generic Python function plan_func(data, constraints) that takes a list of candidate plans 'data' (a list) and the constraints dictionary 'constraints' and returns the plan that satisfies all constraints (e.g. correct trip durations, direct flight paths, friend meeting windows, or participant availability and preferences), or None/empty if no plan satisfies them.
Do not print or include example usage.

CRITICAL CONSTRAINT HANDLING RULE: You must write code to parse and satisfy all keys defined in the constraints, including "preferences" (which may contain passenger preferences, city stay requirements, or participant availability limits like "would rather not meet after 15:30" or "avoid meetings before 13:00"). If a key like "preferences" is empty in the exemplar context, you must still write code that checks if it exists in the test constraints at runtime and filters candidate plans accordingly. If multiple plans satisfy all constraints, you must return the earliest valid plan (or the one that optimizes the objective if specified).

Output format:
<start_of_COT>
...
<end_of_COT>
<start_of_code>
<def plan_func(data, constraints) Python code only>
<end_of_code>"""

        filter_out = query_llm(filter_gen_prompt)
        self.plan_func_code = extract_code_robust(filter_out, "<start_of_code>", "<end_of_code>", "plan_func")

        # Refinement Loop for plan_func
        Yi_sample = None
        for step in range(patience):
            logging.info(f"Verifying plan_func (Refinement Step {step+1}/{patience})...")
            try:
                namespace = {}
                import itertools
                import collections
                import math
                import datetime
                namespace['itertools'] = itertools
                namespace['collections'] = collections
                namespace['math'] = math
                namespace['datetime'] = datetime
                exec(self.combinations_func_code, namespace)
                exec(self.plan_func_code, namespace)
                
                comb_func = namespace.get("combinations_func")
                plan_func = namespace.get("plan_func")
                if not plan_func:
                    raise NameError("plan_func is not defined in generated code")
                
                Xi_sample = comb_func(self.merged_ex)
                Yi_sample = plan_func(Xi_sample, self.k_ex)
                target_sol = self.s_struc.get("solutions")
                
                if Yi_sample == target_sol:
                    logging.info("✅ plan_func filtered correctly and returned target plan!")
                    break
                else:
                    error_msg = f"plan_func returned: {Yi_sample}, but expected: {target_sol}."
                    logging.warning(f"Refinement required: {error_msg}")
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                logging.error(f"Execution error in plan_func:\n{error_msg}")

            # Call Reflection Agent
            logging.info("Running Filter Function Reflection Agent...")
            refl_prompt = f"""You are a Reflection Agent. Your job is to observe the mistake of the Filter Function Generator Agent.
Here is the candidate plans list:
{json.dumps(Xi_sample[:20], indent=2)}

Here is the constraints dictionary:
{json.dumps(self.k_ex, indent=2)}

Here is the expected ground-truth solution S:
{json.dumps(self.s_struc.get("solutions"), indent=2)}

Here is the generated plan_func code:
{self.plan_func_code}

Here is the error or output when running plan_func:
{error_msg}

Please correct the code to ensure it filters candidate plans correctly to return the ground-truth solution.
Output format: Same as Combination Reflection."""
            
            refl_out = query_llm(refl_prompt)
            self.plan_func_code = extract_code_robust(refl_out, "<start_of_code_correction>", "<end_of_code_correction>", "plan_func")

        # 3. Deliver Function Generator Agent
        logging.info("Generating deliver_func(data)...")
        deliver_gen_prompt = f"""You are the Deliver Function Generator Agent.
Your job is to write a generic Python function deliver_func(data) that takes the plan returned by plan_func (which is the structured solution representation: {json.dumps(self.s_struc.get("solutions"), indent=2)}) and returns a natural-language answer.
The natural-language answer must be formatted EXACTLY like the Exemplar Answer Solution:
{json.dumps(self.s_ex, indent=2)}

Make sure to format it with exact matching punctuation, tags, capitalization, and structure.
Your function deliver_func(data) must return a value of the same type (e.g. string or list of strings) as the Exemplar Answer Solution.
Write a clean Python function deliver_func(data). Do not print or include example usage.

Output format:
<start_of_COT>
...
<end_of_COT>
<start_of_code>
<def deliver_func(data) Python code only>
<end_of_code>"""

        deliver_out = query_llm(deliver_gen_prompt)
        self.deliver_func_code = extract_code_robust(deliver_out, "<start_of_code>", "<end_of_code>", "deliver_func")

        # Refinement Loop for deliver_func
        for step in range(patience):
            logging.info(f"Verifying deliver_func (Refinement Step {step+1}/{patience})...")
            try:
                namespace = {}
                import itertools
                import collections
                import math
                import datetime
                namespace['itertools'] = itertools
                namespace['collections'] = collections
                namespace['math'] = math
                namespace['datetime'] = datetime
                exec(self.combinations_func_code, namespace)
                exec(self.plan_func_code, namespace)
                exec(self.deliver_func_code, namespace)
                
                comb_func = namespace.get("combinations_func")
                plan_func = namespace.get("plan_func")
                deliver_func = namespace.get("deliver_func")
                if not deliver_func:
                    raise NameError("deliver_func is not defined in generated code")
                
                Xi_sample = comb_func(self.merged_ex)
                Yi_sample = plan_func(Xi_sample, self.k_ex)
                formatted_ans = deliver_func(Yi_sample)
                
                # Compare formatted_ans to self.s_ex
                def normalize_compare(a, b):
                    if isinstance(a, list) and isinstance(b, list):
                        return [str(x).strip() for x in a] == [str(x).strip() for x in b]
                    return str(a).strip() == str(b).strip()
                
                if normalize_compare(formatted_ans, self.s_ex):
                    logging.info("✅ deliver_func formatted correctly and returned matching answer!")
                    break
                else:
                    error_msg = f"deliver_func returned: {formatted_ans} ({type(formatted_ans)}), but expected: {self.s_ex} ({type(self.s_ex)})."
                    logging.warning(f"Refinement required: {error_msg}")
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                logging.error(f"Execution error in deliver_func:\n{error_msg}")

            # Call Reflection Agent
            logging.info("Running Deliver Function Reflection Agent...")
            refl_prompt = f"""You are a Reflection Agent. Your job is to observe the mistake of the Deliver Function Generator Agent.
Here is the input to deliver_func (the plan returned by plan_func):
{json.dumps(self.s_struc.get("solutions"), indent=2)}

Here is the expected ground-truth solution:
{json.dumps(self.s_ex, indent=2)}

Here is the generated deliver_func code:
{self.deliver_func_code}

Here is the error or output when running deliver_func:
{error_msg}

Please correct the deliver_func(data) code so that it outputs the exact ground-truth solution format.
Output format:
<start_of_COT_correction>
...
<end_of_COT_correction>
<start_of_code_correction>
<corrected deliver_func Python code>
<end_of_code_correction>"""
            
            refl_out = query_llm(refl_prompt)
            self.deliver_func_code = extract_code_robust(refl_out, "<start_of_code_correction>", "<end_of_code_correction>", "deliver_func")


class StandaloneSCOPESolver:
    def __init__(self, solver_config):
        self.config = solver_config
        self.namespace = {}
        # Compile functions in the execution namespace
        import itertools
        import collections
        import math
        import datetime
        self.namespace['itertools'] = itertools
        self.namespace['collections'] = collections
        self.namespace['math'] = math
        self.namespace['datetime'] = datetime
        exec(self.config["combinations_func_code"], self.namespace)
        exec(self.config["plan_func_code"], self.namespace)
        exec(self.config["deliver_func_code"], self.namespace)
        self.combinations_func = self.namespace["combinations_func"]
        self.plan_func = self.namespace["plan_func"]
        self.deliver_func = self.namespace["deliver_func"]

    def solve(self, query):
        start_prompt = token_usage["prompt_tokens"]
        start_comp = token_usage["completion_tokens"]
        start_api = token_usage.get("api_calls", 0)
        
        # 1. Input Agent to extract combinations and constraints
        input_prompt = f"""You are the Input Agent.
You are given an exemplar query, its combinations and constraints, and a test query.
Extract the combinations and constraints for the test query using the exact same structure and keys as the exemplar combinations and constraints.

Exemplar Query:
{self.config["q_ex"]}

Exemplar Combinations (output of Input Agent):
{json.dumps(self.config["c_ex"], indent=2)}

Exemplar Constraints (output of Input Agent):
{json.dumps(self.config["k_ex"], indent=2)}

Instructions:
{self.config["planning_instructions"].get("Input Agent", "")}

Test Query:
{query}

Output format:
<start_of_structured_output>
{{
  "combinations": <extracted combinations parameter values matching exemplar format>,
  "constraints": <extracted constraints parameter values matching exemplar format>
}}
<end_of_structured_output>"""
        
        input_out = query_llm(input_prompt)
        input_json = extract_json_robust(input_out, "<start_of_structured_output>", "<end_of_structured_output>")
        c_i = input_json.get("combinations", {})
        k_i = input_json.get("constraints", {})
        
        plan = None
        final_answer = None
        failure_reason = None
        
        import signal
        
        class SolverTimeout(Exception):
            pass
            
        def timeout_handler(signum, frame):
            raise SolverTimeout()
            
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        # Set a 15-second alarm for generated code execution
        signal.alarm(15)
        
        try:
            # 2. combinations_func - merge combinations and constraints parameters to ensure
            # all information is available in 'data'
            merged_i = {**(c_i or {}), **(k_i or {})}
            Xi = self.combinations_func(merged_i)
            # 3. plan_func
            Yi = self.plan_func(Xi, k_i)
            plan = Yi
            # 4. deliver_func
            final_answer = self.deliver_func(Yi)
            signal.alarm(0)  # disable the alarm
        except SolverTimeout:
            failure_reason = "SCOPE Solver execution timed out (exceeded 15 seconds)!"
            logging.error(failure_reason)
            final_answer = None
        except Exception as e:
            import traceback
            failure_reason = traceback.format_exc()
            logging.error(f"Solver execution failed: {failure_reason}")
            final_answer = None
        finally:
            signal.alarm(0)  # always disable the alarm
            signal.signal(signal.SIGALRM, old_handler)  # restore original handler
            
        end_prompt = token_usage["prompt_tokens"]
        end_comp = token_usage["completion_tokens"]
        end_api = token_usage.get("api_calls", 0)
        query_prompt_tokens = end_prompt - start_prompt
        query_completion_tokens = end_comp - start_comp
        query_api_calls = end_api - start_api
        query_cost = calculate_cost(query_prompt_tokens, query_completion_tokens)
            
        return {
            "predicted_initial_state": c_i,
            "predicted_goal_state": k_i,
            "plan": plan,
            "final_answer": final_answer,
            "failure_reason": failure_reason,
            "prompt_tokens": query_prompt_tokens,
            "completion_tokens": query_completion_tokens,
            "api_calls": query_api_calls,
            "cost": query_cost
        }

def validate_plan_llm(query, predicted_answer):
    prompt = f"""You are a strict constraints validator for planning problems.
Your job is to determine whether a proposed plan satisfies ALL rules and constraints described in the problem query.

Problem Query:
{query}

Proposed Plan:
{predicted_answer}

Analyze the proposed plan step-by-step against every constraint in the query:
1. Does the plan start at the correct location and time?
2. Do all travel steps connect locations that have direct flights / travel times, and are travel times correct?
3. Do stay/meeting/visit durations match the requirements?
4. Are all participant availabilities and preferences met (e.g. rather not meet after/before, flight dates, city durations)?
5. Is the plan completely valid with no conflicts?

Your output must contain only a JSON object inside <start_of_validation> and <end_of_validation>.

Output format:
<start_of_validation>
{{
  "valid": <true or false>,
  "reason": "<explanation of any violated constraint or validation details>"
}}
<end_of_validation>"""
    try:
        out = query_llm(prompt)
        res = extract_json_robust(out, "<start_of_validation>", "<end_of_validation>")
        return res.get("valid") is True, res.get("reason", "")
    except Exception as e:
        logging.error(f"Error in validate_plan_llm: {e}")
        return False, str(e)
