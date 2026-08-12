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

def check_sat_solution_satisfies_clauses(assignment, clauses):
    """
    Check if a truth assignment (list/tuple of signed literals or dictionary)
    satisfies a list of CNF clauses.
    """
    if not assignment:
        return False
    if isinstance(assignment, dict):
        assignment_dict = {int(k): bool(v) for k, v in assignment.items()}
    else:
        assignment_dict = {}
        for lit in assignment:
            val = int(lit)
            assignment_dict[abs(val)] = (val > 0)
            
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            sign = (lit > 0)
            if var in assignment_dict:
                if assignment_dict[var] == sign:
                    clause_satisfied = True
                    break
        if not clause_satisfied:
            return False
    return True

class SCOPESolverBuilder:
    def __init__(self, domain, q_ex, s_ex):
        self.domain = domain
        self.q_ex = q_ex  # JSON string containing {"variables": x, "clauses": [...]}
        self.s_ex = s_ex  # list of signed literals or "unsat"
        self.s_struc = None
        self.combinations_schema = None
        self.constraints_schema = None
        self.planning_instructions = None
        self.c_ex = None
        self.k_ex = None
        self.combinations_func_code = None
        self.plan_func_code = None
        self.deliver_func_code = None

    def build(self):
        logging.info(f"🏗️ Starting Standalone SCOPE Solver construction for StructuredSAT: {self.domain}...")
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
        sol_prompt = f"""You are a Solution Agent for Boolean Satisfiability (SAT) problems.
You are given a JSON query describing a SAT instance and its corresponding satisfying assignment (the solution).
Based on the solution, create a structured JSON representation for the solution.
Please use a simple, robust representation of the truth assignment.

Query:
{self.q_ex}

Answer Solution:
{self.s_ex}

Your output must contain only the structured solution representation inside <start_of_structured_output> and <end_of_structured_output>. Do not use '...' to represent values; output the complete JSON representation of the solution.

Output format:
<start_of_structured_output>
{{
  "solutions": <structured solution representation, e.g., list of signed integers representing variable values, or dict of var -> bool>,
  "solutions_description": <A text description of the structured solution format>
}}
<end_of_structured_output>"""
        
        sol_out = query_llm(sol_prompt)
        self.s_struc = extract_json_robust(sol_out, "<start_of_structured_output>", "<end_of_structured_output>")
        logging.info(f"Structured Solution Representation: {self.s_struc}")

        # 2. Planning Agent
        logging.info("Running Planning Agent...")
        plan_prompt = f"""You are a combinations and constraints planning agent for SAT solving.
You are given an exemplar query and the structured representation of its satisfying assignment solution.
Based on the query and solution representation, identify:
1. "combinations": The set of parameters (representing the variables and search parameters) required to build a search model to explore potential truth assignment sequences.
2. "constraints": The set of constraints (representing the clauses that must evaluate to True) used to filter/validate assignments.

CRITICAL RULE: Any parameter that defines the search variables (such as the number of variables) must be placed in "combinations" (not "constraints"). This is because combinations_func needs to know the variables to generate assignment branches for. Placing this in "constraints" will result in a solver that cannot generate candidates.

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

        # 3. Extract exemplar combinations and constraints values
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
  "combinations": <extracted combinations parameter values matching schema>,
  "constraints": <extracted constraints parameter values matching schema>
}}
<end_of_structured_output>"""

        input_out = query_llm(input_agent_prompt)
        input_json = extract_json_robust(input_out, "<start_of_structured_output>", "<end_of_structured_output>")
        self.c_ex = input_json.get("combinations")
        self.k_ex = input_json.get("constraints")
        logging.info(f"Exemplar combinations (C_ex): {self.c_ex}")
        logging.info(f"Exemplar constraints (K_ex): {self.k_ex}")

    def run_stage_ii(self):
        logging.info("--- Stage II: Generic Solver Generation ---")
        
        # 1. Combinations Function Generator Agent
        logging.info("Generating combinations_func(data)...")
        comb_gen_prompt = f"""You are the Combination Function Generator Agent.
You are given the combinations parameters and descriptions of the exemplar query:
{json.dumps(self.c_ex, indent=2)}

You are also given the expected structured solution format S:
{json.dumps(self.s_struc.get("solutions"), indent=2)}

Planning Instructions:
{self.planning_instructions.get("Combination Function Generator Agent", "")}

Your job is to write a generic Python function combinations_func(data) that takes the combinations parameters 'data' (a dictionary containing variables details) and returns a candidate search space representation (e.g., list of variable indices, or a lazy generator of truth assignments).
For SAT instances with many variables, generating all $2^n$ assignments eagerly will raise a MemoryError. Therefore, combinations_func should return a search configuration or list of variables to assign, letting the solver backtrack in plan_func.
Do not print or include example usage.

Output format:
<start_of_COT>
...
<end_of_COT>
<start_of_code>
<def combinations_func(data) Python code only>
<end_of_code>"""

        comb_out = query_llm(comb_gen_prompt)
        self.combinations_func_code = extract_code_robust(comb_out, "<start_of_code>", "<end_of_code>", "combinations_func")

        # Compile combinations_func to get a sample output
        namespace = {}
        try:
            exec(self.combinations_func_code, namespace)
            comb_func = namespace.get("combinations_func")
            Xi_sample = comb_func(self.c_ex)
        except Exception as e:
            logging.error(f"Error compiling combinations_func: {e}")
            Xi_sample = self.c_ex

        # 2. Filter Function Generator Agent
        logging.info("Generating plan_func(data, constraints)...")
        filter_gen_prompt = f"""You are the Filter Function Generator Agent.
You are given the constraints parameters format (clauses):
{json.dumps(self.k_ex, indent=2)}

You are also given the candidate setup generated by combinations_func:
{json.dumps(Xi_sample, indent=2)}

Planning Instructions:
{self.planning_instructions.get("Filter Function Generator Agent", "")}

Your job is to write a generic Python function plan_func(data, constraints) that solves the boolean satisfiability problem.
It must return a satisfying truth assignment (e.g., list of signed integers like `[1, -2, 3]` where positive is True, negative is False) that satisfies all clauses in constraints, or return None if no satisfying assignment exists.
CRITICAL DESIGN: To handle queries with up to 100 variables and thousands of clauses efficiently, plan_func MUST implement a backtracking search (e.g. DPLL or standard recursive backtracking with unit propagation or simple backtracking) instead of trying to generate all $2^n$ assignments.
Do not print or include example usage.

Output format:
<start_of_COT>
...
<end_of_COT>
<start_of_code>
<def plan_func(data, constraints) Python code only>
<end_of_code>"""

        filter_out = query_llm(filter_gen_prompt)
        self.plan_func_code = extract_code_robust(filter_out, "<start_of_code>", "<end_of_code>", "plan_func")

        # Refinement Loop for combinations_func + plan_func
        patience = 3
        exec_success = False
        for step in range(patience):
            logging.info(f"Verifying combinations_func and plan_func (Refinement Step {step+1}/{patience})...")
            try:
                namespace = {}
                exec(self.combinations_func_code, namespace)
                exec(self.plan_func_code, namespace)
                
                comb_func = namespace.get("combinations_func")
                plan_func = namespace.get("plan_func")
                
                if not comb_func or not plan_func:
                    raise NameError("combinations_func or plan_func is not defined in generated code")
                
                Xi = comb_func(self.c_ex)
                plan = plan_func(Xi, self.k_ex)
                
                # Verify if it satisfies the clauses
                # Since the exemplar is satisfiable, plan must be a satisfying assignment
                clauses = self.k_ex if isinstance(self.k_ex, list) else self.k_ex.get("clauses", [])
                
                if plan is not None and check_sat_solution_satisfies_clauses(plan, clauses):
                    logging.info("✅ generated solver solves the exemplar SAT instance successfully!")
                    exec_success = True
                    break
                else:
                    error_msg = f"Generated plan does not satisfy exemplar clauses! Plan returned: {plan} for clauses {clauses}."
                    logging.warning(f"Refinement required: {error_msg}")
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                logging.error(f"Execution error in solver execution:\n{error_msg}")

            # Call Reflection Agent
            logging.info("Running Solver Reflection Agent...")
            refl_prompt = f"""You are a Reflection Agent. Your job is to observe the mistake of the combinations_func and plan_func code.
Here is the combinations data:
{json.dumps(self.c_ex, indent=2)}

Here is the constraints data:
{json.dumps(self.k_ex, indent=2)}

Here is the generated combinations_func code:
{self.combinations_func_code}

Here is the generated plan_func code:
{self.plan_func_code}

Here is the execution error or output:
{error_msg}

Please correct the codes to ensure plan_func(combinations_func(C_ex), K_ex) successfully performs backtracking search and returns a satisfying assignment (list of signed integers).
Ensure plan_func does NOT raise recursion limits or run out of memory.
Output format:
<start_of_COT_correction>
...
<end_of_COT_correction>
<start_of_code_correction>
<corrected combinations_func Python code>
<end_of_code_correction>
<start_of_plan_correction>
<corrected plan_func Python code>
<end_of_plan_correction>"""
            
            refl_out = query_llm(refl_prompt)
            self.combinations_func_code = extract_code_robust(refl_out, "<start_of_code_correction>", "<end_of_code_correction>", "combinations_func")
            self.plan_func_code = extract_code_robust(refl_out, "<start_of_plan_correction>", "<end_of_plan_correction>", "plan_func")

        # 3. Deliver Function Generator Agent
        logging.info("Generating deliver_func(plan)...")
        deliver_prompt = f"""You are the Deliver Function Generator Agent.
You are given the satisfying plan assignment structure:
{json.dumps(self.s_struc.get("solutions"), indent=2)}

Your job is to write a generic Python function deliver_func(plan) that takes the satisfying assignment plan (a list of literals, or None if unsatisfiable) and returns the final answer string:
- "sat" if plan is a valid assignment (not None)
- "unsat" if plan is None

Do not print or include example usage.

Output format:
<start_of_COT>
...
<end_of_COT>
<start_of_code>
<def deliver_func(plan) Python code only>
<end_of_code>"""

        deliver_out = query_llm(deliver_prompt)
        self.deliver_func_code = extract_code_robust(deliver_out, "<start_of_code>", "<end_of_code>", "deliver_func")

        # Refinement Loop for deliver_func
        for step in range(patience):
            logging.info(f"Verifying deliver_func (Refinement Step {step+1}/{patience})...")
            try:
                namespace = {}
                exec(self.deliver_func_code, namespace)
                deliver_func = namespace.get("deliver_func")
                if not deliver_func:
                    raise NameError("deliver_func is not defined in generated code")
                
                # Test with satisfiable plan
                res_sat = deliver_func(self.s_struc.get("solutions"))
                res_unsat = deliver_func(None)
                
                if res_sat == "sat" and res_unsat == "unsat":
                    logging.info("✅ deliver_func formats output correctly!")
                    break
                else:
                    error_msg = f"deliver_func output format mismatch: sat_input -> {res_sat} (expected 'sat'), unsat_input -> {res_unsat} (expected 'unsat')"
                    logging.warning(f"Refinement required: {error_msg}")
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                logging.error(f"Execution error in deliver_func:\n{error_msg}")

            logging.info("Running Deliver Function Reflection Agent...")
            refl_prompt = f"""You are a Reflection Agent. Your job is to correct deliver_func code.
Here is the generated deliver_func code:
{self.deliver_func_code}

Here is the error or output when running:
{error_msg}

Correct the code to ensure deliver_func(plan) returns "sat" if plan is not None and "unsat" if plan is None.
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
    def __init__(self, config):
        self.config = config
        self.combinations_func = None
        self.plan_func = None
        self.deliver_func = None
        self._compile_functions()

    def _compile_functions(self):
        namespace = {}
        # Import standard libraries for generated code
        import itertools
        import collections
        import math
        import sys
        namespace['itertools'] = itertools
        namespace['collections'] = collections
        namespace['math'] = math
        namespace['sys'] = sys
        # Increase recursion limit just in case backtracking search goes deep
        sys.setrecursionlimit(50000)

        try:
            exec(self.config["combinations_func_code"], namespace)
            exec(self.config["plan_func_code"], namespace)
            exec(self.config["deliver_func_code"], namespace)
            
            self.combinations_func = namespace.get("combinations_func")
            self.plan_func = namespace.get("plan_func")
            self.deliver_func = namespace.get("deliver_func")
        except Exception as e:
            logging.error(f"Failed to compile generated solver functions: {e}")
            raise e

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
        # Set a 30-second alarm for generated code execution on large SAT formulas
        signal.alarm(30)
        
        try:
            # 2. combinations_func
            Xi = self.combinations_func(c_i)
            # 3. plan_func
            Yi = self.plan_func(Xi, k_i)
            plan = Yi
            # 4. deliver_func
            final_answer = self.deliver_func(Yi)
            signal.alarm(0)  # disable the alarm
        except SolverTimeout:
            failure_reason = "SCOPE Solver execution timed out (exceeded 30 seconds)!"
            logging.error(failure_reason)
            final_answer = "unsat"  # fallback
        except Exception as e:
            import traceback
            failure_reason = traceback.format_exc()
            logging.error(f"Solver execution failed: {failure_reason}")
            final_answer = "unsat"  # fallback
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
