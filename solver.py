"""
Mathematical Optimization Solver Engine
Solves the Mixed-Integer Linear Programming (MILP) Model for Consulting Resource Allocation.
Supports PuLP (CBC), SciPy (HiGHS/milp), and Gurobi/CPLEX if available.
"""

import time
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import pulp

def validate_parameters(
    client_df: pd.DataFrame,
    capacities_df: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """
    Validates logical consistency of input parameters before solving.
    Returns (is_valid, list_of_warning_or_error_messages).
    """
    issues = []
    
    total_required = client_df["Required_Hours"].sum()
    total_capacity = capacities_df["Capacity_Hours"].sum()
    
    if total_required > total_capacity:
        issues.append(
            f"❌ Demand Exceeds Capacity: Total client requirement ({total_required:.1f} hrs) "
            f"exceeds total consultant capacity ({total_capacity:.1f} hrs) by {total_required - total_capacity:.1f} hrs."
        )
        
    for _, row in client_df.iterrows():
        c = row["Client"]
        req = row["Required_Hours"]
        sum_min = row["M_Min"] + row["A_Min"] + row["T_Min"] + row["AI_Min"]
        if sum_min > req:
            issues.append(
                f"❌ Inconsistent Expertise Minimums for {c}: Sum of minimums ({sum_min:.1f} hrs) "
                f"exceeds client required hours ({req:.1f} hrs)."
            )
        if row["PM_Min"] > req:
            issues.append(
                f"❌ Inconsistent PM Hours for {c}: Minimum PM requirement ({row['PM_Min']:.1f} hrs) "
                f"exceeds client required hours ({req:.1f} hrs)."
            )
            
    # Check consultant specific capacity vs total minimums required across clients
    for col, role in [("M_Min", "M"), ("A_Min", "A"), ("T_Min", "T"), ("AI_Min", "AI")]:
        req_min = client_df[col].sum()
        cap_val = capacities_df.loc[capacities_df["Consultant"] == role, "Capacity_Hours"].values[0]
        if req_min > cap_val:
            issues.append(
                f"❌ Consultant Overload ({role}): Total minimum required hours across clients ({req_min:.1f} hrs) "
                f"exceeds consultant capacity ({cap_val:.1f} hrs)."
            )
            
    is_valid = len([msg for msg in issues if msg.startswith("❌")]) == 0
    return is_valid, issues


def solve_consulting_allocation(
    client_df: pd.DataFrame,
    billing_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    capacities_df: pd.DataFrame,
    solver_name: str = "PuLP (CBC)"
) -> Dict[str, Any]:
    """
    Formulates and solves the consulting resource allocation optimization problem.
    """
    start_time = time.time()
    
    # Pre-validation
    is_valid, validation_errors = validate_parameters(client_df, capacities_df)
    if not is_valid:
        return {
            "status": "Infeasible Prior to Solve",
            "optimal": False,
            "objective_value": 0.0,
            "errors": validation_errors,
            "solve_time": time.time() - start_time
        }

    # Extract Sets
    consultants = capacities_df["Consultant"].tolist()
    clients = client_df["Client"].tolist()
    
    # Extract Parameters
    capacities = dict(zip(capacities_df["Consultant"], capacities_df["Capacity_Hours"]))
    
    req_hours = {}
    pm_min = {}
    expertise_min = {c: {} for c in clients}
    
    for _, row in client_df.iterrows():
        cid = row["Client"]
        req_hours[cid] = float(row["Required_Hours"])
        pm_min[cid] = float(row["PM_Min"])
        expertise_min[cid]["M"] = float(row["M_Min"])
        expertise_min[cid]["A"] = float(row["A_Min"])
        expertise_min[cid]["T"] = float(row["T_Min"])
        expertise_min[cid]["AI"] = float(row["AI_Min"])
        
    rates = {}
    for _, row in billing_df.iterrows():
        cid = row["Client"]
        rates[cid] = {col: float(row[col]) for col in consultants}
        
    costs = {}
    for _, row in costs_df.iterrows():
        cid = row["Client"]
        costs[cid] = {col: float(row[col]) for col in consultants}
        
    # Unit contribution margins: M_ij = R_ij - C_ij
    margins = {}
    for cid in clients:
        margins[cid] = {c: rates[cid][c] - costs[cid][c] for c in consultants}

    # --- PuLP Formulation ---
    prob = pulp.LpProblem("Consulting_Resource_Allocation", pulp.LpMaximize)
    
    # Decision Variables
    # x_ij >= 0: continuous hours consultant i works for client j
    x = pulp.LpVariable.dicts(
        "x",
        ((i, j) for i in consultants for j in clients),
        lowBound=0,
        cat="Continuous"
    )
    
    # y_ij in {0, 1}: binary indicator if consultant i works for client j
    y = pulp.LpVariable.dicts(
        "y",
        ((i, j) for i in consultants for j in clients),
        cat="Binary"
    )
    
    # p_ij in {0, 1}: binary indicator if consultant i is Principal PM for client j
    p = pulp.LpVariable.dicts(
        "p",
        ((i, j) for i in consultants for j in clients),
        cat="Binary"
    )
    
    # Objective: Maximize Z = sum_i sum_j (R_ij - C_ij) * x_ij
    prob += pulp.lpSum(margins[j][i] * x[(i, j)] for i in consultants for j in clients), "Total_Contribution_Margin"
    
    # 1. Consultant capacity: sum_j x_ij <= Cap_i
    for i in consultants:
        prob += pulp.lpSum(x[(i, j)] for j in clients) <= capacities[i], f"Capacity_{i}"
        
    # 2. Client requirements: sum_i x_ij = H_j
    for j in clients:
        prob += pulp.lpSum(x[(i, j)] for i in consultants) == req_hours[j], f"Demand_{j}"
        
    # 3. Required expertise: x_ij >= S_ij
    for i in consultants:
        for j in clients:
            if expertise_min[j][i] > 0:
                prob += x[(i, j)] >= expertise_min[j][i], f"ExpertiseMin_{i}_{j}"
                
    # 4. Assignment linkage: x_ij <= H_j * y_ij (Big-M with M = H_j)
    for i in consultants:
        for j in clients:
            prob += x[(i, j)] <= req_hours[j] * y[(i, j)], f"Linkage_{i}_{j}"
            
    # 5. Exactly one PM per client: sum_i p_ij = 1
    for j in clients:
        prob += pulp.lpSum(p[(i, j)] for i in consultants) == 1, f"OnePM_{j}"
        
    # 6. PM must work: p_ij <= y_ij
    for i in consultants:
        for j in clients:
            prob += p[(i, j)] <= y[(i, j)], f"PMWork_{i}_{j}"
            
    # 7. PM minimum hours: x_ij >= PM_j * p_ij
    for i in consultants:
        for j in clients:
            prob += x[(i, j)] >= pm_min[j] * p[(i, j)], f"PMMinHours_{i}_{j}"

    # Solver selection
    if "Gurobi" in solver_name:
        try:
            solver = pulp.GUROBI_CMD(msg=False)
            prob.solve(solver)
        except Exception:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
    elif "CPLEX" in solver_name:
        try:
            solver = pulp.CPLEX_CMD(msg=False)
            prob.solve(solver)
        except Exception:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
    else:
        # Default PuLP CBC solver
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        
    solve_duration = time.time() - start_time
    status_str = pulp.LpStatus[prob.status]
    
    if status_str != "Optimal":
        return {
            "status": status_str,
            "optimal": False,
            "objective_value": 0.0,
            "errors": [f"Solver concluded with status: {status_str}. Check capacity and constraint settings."],
            "solve_time": solve_duration
        }

    # Extract Results
    allocations = {}
    pm_assignments = {}
    assigned_flags = {}
    
    for j in clients:
        allocations[j] = {}
        assigned_flags[j] = {}
        for i in consultants:
            allocations[j][i] = round(float(pulp.value(x[(i, j)]) or 0.0), 2)
            assigned_flags[j][i] = int(pulp.value(y[(i, j)]) or 0)
            if pulp.value(p[(i, j)]) and pulp.value(p[(i, j)]) > 0.5:
                pm_assignments[j] = i
                
    # Build detailed allocation DataFrame
    alloc_rows = []
    for j in clients:
        row = {
            "Client": j,
            "Priority": client_df.loc[client_df["Client"] == j, "Priority"].values[0],
            "Req_Hours": req_hours[j],
            "PM_Assigned": pm_assignments.get(j, "N/A"),
            "PM_Min_Req": pm_min[j],
        }
        client_rev = 0.0
        client_cost = 0.0
        for i in consultants:
            hrs = allocations[j][i]
            row[f"{i}_Hours"] = hrs
            client_rev += hrs * rates[j][i]
            client_cost += hrs * costs[j][i]
            
        row["Client_Revenue"] = client_rev
        row["Client_Cost"] = client_cost
        row["Client_Margin"] = client_rev - client_cost
        row["Margin_Pct"] = (row["Client_Margin"] / client_rev * 100) if client_rev > 0 else 0.0
        alloc_rows.append(row)
        
    allocation_df = pd.DataFrame(alloc_rows)
    
    # Consultant Utilization DataFrame
    util_rows = []
    for i in consultants:
        allocated = sum(allocations[j][i] for j in clients)
        cap = capacities[i]
        slack = cap - allocated
        pm_count = sum(1 for j in clients if pm_assignments.get(j) == i)
        util_pct = (allocated / cap * 100) if cap > 0 else 0.0
        
        # Contribution margin earned by this consultant
        cons_margin = sum(allocations[j][i] * margins[j][i] for j in clients)
        cons_rev = sum(allocations[j][i] * rates[j][i] for j in clients)
        cons_cost = sum(allocations[j][i] * costs[j][i] for j in clients)
        
        util_rows.append({
            "Consultant": i,
            "Role": capacities_df.loc[capacities_df["Consultant"] == i, "Role"].values[0] if "Role" in capacities_df.columns else i,
            "Capacity": cap,
            "Allocated_Hours": allocated,
            "Slack_Hours": slack,
            "Utilization_Pct": round(util_pct, 1),
            "PM_Assignments_Count": pm_count,
            "Total_Revenue": cons_rev,
            "Total_Cost": cons_cost,
            "Contribution_Margin": cons_margin
        })
    utilization_df = pd.DataFrame(util_rows)
    
    total_margin = float(pulp.value(prob.objective))
    total_revenue = allocation_df["Client_Revenue"].sum()
    total_cost = allocation_df["Client_Cost"].sum()
    overall_margin_pct = (total_margin / total_revenue * 100) if total_revenue > 0 else 0.0
    
    return {
        "status": status_str,
        "optimal": True,
        "objective_value": total_margin,
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "margin_percentage": overall_margin_pct,
        "allocation_df": allocation_df,
        "utilization_df": utilization_df,
        "allocations_dict": allocations,
        "pm_assignments": pm_assignments,
        "solve_time": round(solve_duration, 4),
        "solver_used": solver_name
    }
