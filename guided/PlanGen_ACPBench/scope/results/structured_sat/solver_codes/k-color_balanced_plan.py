def plan_func(data, constraints):
    clauses = constraints.get("clauses", [])
    
    # Determine variables
    var_ids = set()
    for clause in clauses:
        for lit in clause:
            var_ids.add(abs(lit))
    
    if isinstance(data, dict):
        if "variable_order" in data and data["variable_order"]:
            try:
                ordered_vars = [int(v) for v in data["variable_order"]]
            except Exception:
                ordered_vars = sorted(var_ids)
        elif "count" in data and isinstance(data["count"], int):
            ordered_vars = list(range(1, data["count"] + 1))
        else:
            ordered_vars = sorted(var_ids)
    else:
        ordered_vars = sorted(var_ids)
    
    for v in ordered_vars:
        var_ids.add(v)
    ordered_vars = sorted(var_ids) if not ordered_vars else ordered_vars

    def simplify(current_clauses, assignment):
        simplified = []
        for clause in current_clauses:
            clause_satisfied = False
            new_clause = []
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    val = assignment[var]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        clause_satisfied = True
                        break
                else:
                    new_clause.append(lit)
            if clause_satisfied:
                continue
            if not new_clause:
                return None
            simplified.append(new_clause)
        return simplified

    def unit_propagate(current_clauses, assignment):
        while True:
            changed = False
            
            # Check for empty clause
            for clause in current_clauses:
                if len(clause) == 0:
                    return None, None
            
            # Unit clauses
            unit_literals = []
            for clause in current_clauses:
                if len(clause) == 1:
                    unit_literals.append(clause[0])
            
            if unit_literals:
                for lit in unit_literals:
                    var = abs(lit)
                    val = lit > 0
                    if var in assignment:
                        if assignment[var] != val:
                            return None, None
                    else:
                        assignment[var] = val
                        changed = True
                current_clauses = simplify(current_clauses, assignment)
                if current_clauses is None:
                    return None, None
            
            # Pure literal elimination
            literal_polarity = {}
            for clause in current_clauses:
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        continue
                    if var not in literal_polarity:
                        literal_polarity[var] = set()
                    literal_polarity[var].add(1 if lit > 0 else -1)
            
            pure_assignments = {}
            for var, pols in literal_polarity.items():
                if len(pols) == 1:
                    pure_assignments[var] = (1 in pols)
            
            if pure_assignments:
                for var, val in pure_assignments.items():
                    if var in assignment:
                        if assignment[var] != val:
                            return None, None
                    else:
                        assignment[var] = val
                        changed = True
                current_clauses = simplify(current_clauses, assignment)
                if current_clauses is None:
                    return None, None
            
            if not changed:
                break
        
        return current_clauses, assignment

    def choose_variable(current_clauses, assignment):
        freq = {}
        for clause in current_clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    freq[var] = freq.get(var, 0) + 1
        if freq:
            return max(freq, key=freq.get)
        for v in ordered_vars:
            if v not in assignment:
                return v
        return None

    def dpll(current_clauses, assignment):
        current_clauses = simplify(current_clauses, assignment)
        if current_clauses is None:
            return None
        if not current_clauses:
            return assignment
        
        current_clauses, assignment = unit_propagate(current_clauses, assignment.copy())
        if current_clauses is None:
            return None
        if not current_clauses:
            return assignment
        
        var = choose_variable(current_clauses, assignment)
        if var is None:
            return assignment
        
        for val in (True, False):
            new_assignment = assignment.copy()
            new_assignment[var] = val
            result = dpll(current_clauses, new_assignment)
            if result is not None:
                return result
        
        return None

    result = dpll(clauses, {})
    if result is None:
        return None
    
    final_vars = ordered_vars if ordered_vars else sorted(var_ids)
    return [v if result.get(v, False) else -v for v in final_vars]