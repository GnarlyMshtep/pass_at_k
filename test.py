def _taller_is_correct(proposed_sol: str, ground_truth:str) -> bool: 
    proposed_sol= proposed_sol.strip().replace(" ", "")
    ground_truth=ground_truth.strip().replace(" ", "")

    proposed_sol_set = set()
    ground_truth_set = set()
    for i, char in enumerate(proposed_sol): 
        if i %2 == 0: 
            proposed_sol_set.add(char)
        elif char != ",":
            return False 
    
    for i, char in enumerate(ground_truth): 
        if i %2 == 0: 
            ground_truth_set.add(char)
        elif char != ",":
            print(f"DEBUG: grouth truth has unexpected format {ground_truth}")

    return proposed_sol_set == ground_truth_set  


print(_taller_is_correct("A, C, E", "E"), _taller_is_correct("WED", "W, E, D"), _taller_is_correct("W, E, D", "W,E, D G")) 