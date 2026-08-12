def plan_func(data, constraints):
    def simplify(clauses, assignment):
        simplified = []
        for clause in clauses:
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

    def unit_propagate(clauses, assignment):
        while True:
            changed = False
            for clause in clauses:
                if len(clause) == 1:
                    lit = clause[0]
                    var = abs(lit)
                    val = lit > 0
                    if var in assignment:
                        if assignment[var] != val:
                            return None, None
                    else:
                        assignment[var] = val
                        changed = True
            if not changed:
                break
            clauses = simplify(clauses, assignment)
            if clauses is None:
                return None, None
        return clauses, assignment

    def pure_literal_assign(clauses, assignment):
        while True:
            polarity = {}
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        continue
                    if var not in polarity:
                        polarity[var] = set()
                    polarity[var].add(lit > 0)

            pure_assignments = {}
            for var, signs in polarity.items():
                if len(signs) == 1:
                    pure_assignments[var] = True in signs

            if not pure_assignments:
                break

            assignment.update(pure_assignments)
            clauses = simplify(clauses, assignment)
            if clauses is None:
                return None, None

        return clauses, assignment

    def choose_variable(clauses, assignment):
        freq = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    freq[var] = freq.get(var, 0) + 1
        if not freq:
            return None
        return max(freq, key=freq.get)

    def dpll(clauses, assignment):
        clauses = simplify(clauses, assignment)
        if clauses is None:
            return None
        if not clauses:
            return assignment

        clauses, assignment = unit_propagate(clauses, assignment.copy())
        if clauses is None:
            return None
        if not clauses:
            return assignment

        clauses, assignment = pure_literal_assign(clauses, assignment.copy())
        if clauses is None:
            return None
        if not clauses:
            return assignment

        var = choose_variable(clauses, assignment)
        if var is None:
            return assignment

        for val in (True, False):
            new_assignment = assignment.copy()
            new_assignment[var] = val
            result = dpll(clauses, new_assignment)
            if result is not None:
                return result

        return None

    all_vars = set()
    for clause in constraints:
        for lit in clause:
            all_vars.add(abs(lit))

    result = dpll(constraints, {})
    if result is None:
        return None

    for var in all_vars:
        if var not in result:
            result[var] = False

    return [var if result[var] else -var for var in sorted(all_vars)]