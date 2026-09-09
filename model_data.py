"""
Optimization Data Structures and Default Problem Instances
Based on: "Optimal Allocation of Consulting Resources Across Client Engagements"
"""

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

CONSULTANTS = ["M", "A", "T", "AI"]
CONSULTANT_NAMES = {
    "M": "Marketing Technology (M)",
    "A": "Advertising (A)",
    "T": "Technology (T)",
    "AI": "Artificial Intelligence (AI)"
}

CLIENT_PRIORITIES = {
    "C1": "Strategic", "C2": "Important", "C3": "Strategic", "C4": "Standard",
    "C5": "Important", "C6": "Standard", "C7": "Strategic", "C8": "Standard",
    "C9": "Important", "C10": "Standard", "C11": "Important", "C12": "Strategic"
}

CLIENT_REQUIREMENTS_DEFAULT = {
    "C1": {"Priority": "Strategic", "Required_Hours": 14, "M_Min": 4, "A_Min": 0, "T_Min": 3, "AI_Min": 0, "PM_Min": 6},
    "C2": {"Priority": "Important", "Required_Hours": 12, "M_Min": 0, "A_Min": 4, "T_Min": 0, "AI_Min": 3, "PM_Min": 4},
    "C3": {"Priority": "Strategic", "Required_Hours": 16, "M_Min": 0, "A_Min": 0, "T_Min": 6, "AI_Min": 4, "PM_Min": 6},
    "C4": {"Priority": "Standard",  "Required_Hours": 10, "M_Min": 5, "A_Min": 3, "T_Min": 0, "AI_Min": 0, "PM_Min": 3},
    "C5": {"Priority": "Important", "Required_Hours": 15, "M_Min": 4, "A_Min": 0, "T_Min": 5, "AI_Min": 0, "PM_Min": 4},
    "C6": {"Priority": "Standard",  "Required_Hours": 11, "M_Min": 0, "A_Min": 4, "T_Min": 0, "AI_Min": 3, "PM_Min": 3},
    "C7": {"Priority": "Strategic", "Required_Hours": 13, "M_Min": 0, "A_Min": 0, "T_Min": 5, "AI_Min": 4, "PM_Min": 6},
    "C8": {"Priority": "Standard",  "Required_Hours": 9,  "M_Min": 3, "A_Min": 3, "T_Min": 0, "AI_Min": 0, "PM_Min": 3},
    "C9": {"Priority": "Important", "Required_Hours": 14, "M_Min": 4, "A_Min": 0, "T_Min": 4, "AI_Min": 0, "PM_Min": 4},
    "C10": {"Priority": "Standard", "Required_Hours": 12, "M_Min": 0, "A_Min": 4, "T_Min": 3, "AI_Min": 0, "PM_Min": 3},
    "C11": {"Priority": "Important", "Required_Hours": 10, "M_Min": 2, "A_Min": 0, "T_Min": 0, "AI_Min": 4, "PM_Min": 4},
    "C12": {"Priority": "Strategic", "Required_Hours": 14, "M_Min": 4, "A_Min": 3, "T_Min": 0, "AI_Min": 3, "PM_Min": 6},
}

BILLING_RATES_DEFAULT = {
    "C1":  {"M": 5000, "A": 5500, "T": 6500, "AI": 8000},
    "C2":  {"M": 5200, "A": 5800, "T": 6200, "AI": 8500},
    "C3":  {"M": 4800, "A": 5500, "T": 7000, "AI": 9000},
    "C4":  {"M": 5500, "A": 6000, "T": 6000, "AI": 7500},
    "C5":  {"M": 5000, "A": 5700, "T": 7200, "AI": 8200},
    "C6":  {"M": 5300, "A": 6200, "T": 6300, "AI": 8800},
    "C7":  {"M": 4900, "A": 5600, "T": 7300, "AI": 9200},
    "C8":  {"M": 5600, "A": 6100, "T": 6100, "AI": 7800},
    "C9":  {"M": 5100, "A": 5900, "T": 7100, "AI": 8300},
    "C10": {"M": 5400, "A": 6300, "T": 6800, "AI": 8000},
    "C11": {"M": 5200, "A": 5700, "T": 6500, "AI": 8700},
    "C12": {"M": 5000, "A": 6000, "T": 6700, "AI": 9000},
}

DELIVERY_COSTS_DEFAULT = {
    "C1":  {"M": 2000, "A": 2400, "T": 3000, "AI": 4000},
    "C2":  {"M": 2100, "A": 2500, "T": 2900, "AI": 4200},
    "C3":  {"M": 2000, "A": 2300, "T": 3200, "AI": 4500},
    "C4":  {"M": 2300, "A": 2600, "T": 2800, "AI": 3900},
    "C5":  {"M": 2100, "A": 2400, "T": 3300, "AI": 4100},
    "C6":  {"M": 2200, "A": 2700, "T": 3000, "AI": 4300},
    "C7":  {"M": 2000, "A": 2400, "T": 3400, "AI": 4600},
    "C8":  {"M": 2400, "A": 2700, "T": 2900, "AI": 4000},
    "C9":  {"M": 2100, "A": 2500, "T": 3300, "AI": 4100},
    "C10": {"M": 2200, "A": 2800, "T": 3100, "AI": 4000},
    "C11": {"M": 2100, "A": 2400, "T": 3000, "AI": 4300},
    "C12": {"M": 2000, "A": 2600, "T": 3100, "AI": 4500},
}

CAPACITIES_DEFAULT = {
    "M": 40.0,
    "A": 40.0,
    "T": 40.0,
    "AI": 40.0
}

def get_default_client_df() -> pd.DataFrame:
    rows = []
    for cid, data in CLIENT_REQUIREMENTS_DEFAULT.items():
        rows.append({
            "Client": cid,
            "Priority": data["Priority"],
            "Required_Hours": data["Required_Hours"],
            "M_Min": data["M_Min"],
            "A_Min": data["A_Min"],
            "T_Min": data["T_Min"],
            "AI_Min": data["AI_Min"],
            "PM_Min": data["PM_Min"]
        })
    return pd.DataFrame(rows)

def get_default_billing_df() -> pd.DataFrame:
    rows = []
    for cid, rates in BILLING_RATES_DEFAULT.items():
        rows.append({
            "Client": cid,
            "M": rates["M"],
            "A": rates["A"],
            "T": rates["T"],
            "AI": rates["AI"]
        })
    return pd.DataFrame(rows)

def get_default_costs_df() -> pd.DataFrame:
    rows = []
    for cid, costs in DELIVERY_COSTS_DEFAULT.items():
        rows.append({
            "Client": cid,
            "M": costs["M"],
            "A": costs["A"],
            "T": costs["T"],
            "AI": costs["AI"]
        })
    return pd.DataFrame(rows)

def get_default_capacities_df() -> pd.DataFrame:
    rows = []
    for cid, cap in CAPACITIES_DEFAULT.items():
        rows.append({
            "Consultant": cid,
            "Role": CONSULTANT_NAMES[cid],
            "Capacity_Hours": cap
        })
    return pd.DataFrame(rows)
