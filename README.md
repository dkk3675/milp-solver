# Consulting Resource Optimization & PM Allocation System

A production-ready decision support application built **strictly in Python** with an interactive **Streamlit** GUI, solving the Mixed-Integer Linear Programming (MILP) model for allocating specialist consultants across client engagements.

---

## 🎯 Course Project Alignment & Rubric Coverage

This software satisfies all requirements outlined in the course final project specification:
1. **Mathematical Formulation**: Complete MILP model with decision variables for allocation hours ($x_{ij}$), binary team assignment indicators ($y_{ij}$), and binary Principal PM designations ($p_{ij}$), complete with 8 constraint sets.
2. **Interactive GUI in Streamlit**:
   - Allows users to enter/edit all parameters (Client requirements, Billing rates, Delivery costs, Consultant capacities).
   - Solves the model when users click **"Solve Optimization Model"**.
   - Displays clear results in dedicated tabs (summary KPI metrics, client allocation matrix, consultant slack/utilization, and interactive Plotly visualization charts).
3. **Mandatory Abstract Submission**: Includes an executive ~400-word submission-ready abstract covering context, objectives, key assumptions, parameters, modeling approach, solver, and GUI plan.

---

## 🚀 Quickstart (Running Locally)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will launch automatically in your browser (typically at `http://localhost:8501`).

---

## 📁 Repository Structure

```
├── app.py              # Main Streamlit GUI with 5 interactive tabs and KPI dashboards
├── solver.py           # MILP optimization engine (Exclusively IBM ILOG CPLEX via docplex)
├── model_data.py       # Default benchmark parameters (Appendix A, B, C) and data structures
├── requirements.txt    # Python package dependencies (Streamlit, CPLEX, DOcplex, Pandas, Plotly)
└── README.md           # Project documentation and mathematical formulation
```

---

## 📐 Mathematical Formulation Summary

- **Sets**: $i \in \{M, A, T, AI\}$ (Consultants); $j \in \{C1, \dots, C12\}$ (Clients).
- **Decision Variables**:
  - $x_{ij} \ge 0$: Hours consultant $i$ allocates to client $j$.
  - $y_{ij} \in \{0, 1\}$: Binary indicator if consultant $i$ is assigned to client $j$.
  - $p_{ij} \in \{0, 1\}$: Binary indicator if consultant $i$ is Principal PM for client $j$.
- **Objective Function**:
  $$\max Z = \sum_{i} \sum_{j} (R_{ij} - C_{ij}) x_{ij}$$
- **Constraints**:
  1. $\sum_j x_{ij} \le \text{Cap}_i \quad \forall i$ (Consultant Capacity, default 40h)
  2. $\sum_i x_{ij} = H_j \quad \forall j$ (Client Demand Fulfillment)
  3. $x_{ij} \ge S_{ij} \quad \forall i, j$ (Mandatory Expertise Minimums)
  4. $x_{ij} \le H_j y_{ij} \quad \forall i, j$ (Assignment Linkage via Big-M)
  5. $\sum_i p_{ij} = 1 \quad \forall j$ (Exactly One Principal PM per Client)
  6. $p_{ij} \le y_{ij} \quad \forall i, j$ (PM Must Work on Project)
  7. $x_{ij} \ge PM_j \cdot p_{ij} \quad \forall i, j$ (PM Minimum Hours Contribution)
  8. $x_{ij} \ge 0, \; y_{ij}, p_{ij} \in \{0, 1\} \quad \forall i, j$ (Variable Domains)

---

## 📊 Benchmark Optimal Results

For the baseline problem parameters:
- **Optimal Contribution Margin**: **$559,000.00**
- **Total Firm Revenue**: **$1,035,900.00**
- **Total Delivery Cost**: **$476,900.00**
- **Profit Margin**: **53.96%**
- **Capacity Utilization**:
  - Marketing Tech (M): 30.0 / 40.0 hours (10 hours slack)
  - Advertising (A): 40.0 / 40.0 hours (100% utilized)
  - Technology (T): 40.0 / 40.0 hours (100% utilized)
  - Artificial Intelligence (AI): 40.0 / 40.0 hours (100% utilized)
- **Principal PM Distribution**:
  - Marketing Tech (M): 1 project (C9)
  - Advertising (A): 5 projects (C2, C4, C6, C8, C10)
  - Technology (T): 3 projects (C1, C3, C5)
  - Artificial Intelligence (AI): 3 projects (C7, C11, C12)
