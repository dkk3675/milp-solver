"""
Optimal Allocation of Consulting Resources Across Client Engagements
Interactive Streamlit Application for Mixed-Integer Linear Programming (MILP) Decision Support.
Strictly in Python with Streamlit frontend. Solved exclusively with IBM ILOG CPLEX.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import model_data
import solver

# Page Configuration
st.set_page_config(
    page_title="Consulting Resource Optimization | MILP Decision Support",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sophisticated Custom CSS for Enterprise UI & Mobile Responsiveness
st.markdown("""
<style>
    /* 1. Completely hide Streamlit sidebar and collapse control */
    [data-testid="stSidebar"], [data-testid="collapsedControl"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* 2. Responsive Main Layout Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
        }
    }

    /* 3. Executive Typography */
    .main-header {
        font-size: calc(1.5rem + 0.8vw);
        font-weight: 800;
        color: #1E40AF;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    /* 4. Metadata Badge Pills */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }
    .badge-blue {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .badge-green {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .badge-purple {
        background-color: #F5F3FF;
        color: #6D28D9;
        border: 1px solid #DDD6FE;
    }
    .badge-slate {
        background-color: #F8FAFC;
        color: #334155;
        border: 1px solid #E2E8F0;
    }

    /* 5. Top Action Container Card */
    .top-action-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* 6. Executive KPI Metric Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: left;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .kpi-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 5px;
        display: inline-block;
    }
    .kpi-delta-positive {
        color: #059669;
    }
    .kpi-delta-neutral {
        color: #2563EB;
    }

    /* 7. Tabs Responsive Scroll & Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 2px;
        overflow-x: auto;
        white-space: nowrap;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 8px 8px 0 0;
        color: #475569;
        transition: color 0.15s ease, background-color 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #1E40AF !important;
        background-color: #EFF6FF !important;
        border-bottom: 2px solid #2563EB !important;
    }

    /* 8. Math and Content Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 1rem;
    }
    .content-box-slate {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 0.85rem;
    }

    /* 9. Responsive DataTables */
    .dataframe-container {
        overflow-x: auto;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }

    /* 10. Hide default footer and excess Streamlit badges */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "client_df" not in st.session_state:
    st.session_state.client_df = model_data.get_default_client_df()
if "billing_df" not in st.session_state:
    st.session_state.billing_df = model_data.get_default_billing_df()
if "costs_df" not in st.session_state:
    st.session_state.costs_df = model_data.get_default_costs_df()
if "capacities_df" not in st.session_state:
    st.session_state.capacities_df = model_data.get_default_capacities_df()
if "solution" not in st.session_state:
    st.session_state.solution = solver.solve_consulting_allocation(
        st.session_state.client_df,
        st.session_state.billing_df,
        st.session_state.costs_df,
        st.session_state.capacities_df
    )

sol = st.session_state.solution

# ==========================================
# EXECUTIVE HEADER & METADATA BADGES
# ==========================================
st.markdown('<div class="main-header">💼 Optimal Allocation of Consulting Resources</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Mixed-Integer Linear Programming (MILP) Decision Support System • Solved Exclusively with IBM ILOG CPLEX</div>', unsafe_allow_html=True)

# Metadata Pills
st.markdown("""
<div class="badge-container">
    <span class="badge-pill badge-blue">⚡ Engine: IBM ILOG CPLEX Optimizer</span>
    <span class="badge-pill badge-green">🎯 Status: Exact Global Optimal</span>
    <span class="badge-pill badge-purple">👥 Resources: 4 Specialists (160h Cap)</span>
    <span class="badge-pill badge-slate">📋 Demand: 12 Clients (150h Total)</span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# TOP ACTION BAR (ABOVE NAVIGATION TABS)
# ==========================================
with st.container():
    act_col1, act_col2, act_col3, act_col4 = st.columns([1.5, 1.2, 1.2, 1.1])
    
    with act_col1:
        solve_clicked = st.button("🚀 Solve Optimization Model", type="primary", use_container_width=True)
    with act_col2:
        if sol.get("optimal", False):
            st.markdown(f"""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:6px 12px; text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#15803D; text-transform:uppercase;">Contribution Margin</div>
                <div style="font-size:1.05rem; font-weight:800; color:#166534;">${sol['objective_value']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:8px; padding:6px 12px; text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#B91C1C; text-transform:uppercase;">Status</div>
                <div style="font-size:1rem; font-weight:800; color:#991B1B;">Infeasible</div>
            </div>
            """, unsafe_allow_html=True)
    with act_col3:
        if sol.get("optimal", False):
            st.markdown(f"""
            <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:6px 12px; text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#1D4ED8; text-transform:uppercase;">Firm Load Utilized</div>
                <div style="font-size:1.05rem; font-weight:800; color:#1E40AF;">{(st.session_state.client_df['Required_Hours'].sum() / st.session_state.capacities_df['Capacity_Hours'].sum() * 100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 12px; text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">Firm Load</div>
                <div style="font-size:1rem; font-weight:800; color:#334155;">--</div>
            </div>
            """, unsafe_allow_html=True)
    with act_col4:
        solve_time_ms = sol.get("solve_time", 0.0) * 1000
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 12px; text-align:center;">
            <div style="font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase;">CPLEX Time</div>
            <div style="font-size:1.05rem; font-weight:800; color:#334155;">{solve_time_ms:.1f} ms</div>
        </div>
        """, unsafe_allow_html=True)

if solve_clicked:
    with st.spinner("Formulating constraints and executing IBM ILOG CPLEX solver..."):
        st.session_state.solution = solver.solve_consulting_allocation(
            st.session_state.client_df,
            st.session_state.billing_df,
            st.session_state.costs_df,
            st.session_state.capacities_df
        )
    st.toast("Optimization solved successfully with IBM ILOG CPLEX!", icon="✅")
    st.rerun()

st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

# ==========================================
# REVERSED TABS INTERFACE
# ==========================================
tab_math, tab_params, tab_charts, tab_results = st.tabs([
    "📐 Mathematical Model",
    "🎛️ Parameters & Data Editor",
    "📊 Visual Analytics & Plots",
    "🚀 Optimization & Results"
])

# ==========================================
# SECTION 1: MATHEMATICAL MODEL & FORMULATION
# ==========================================
with tab_math:
    st.subheader("📐 Mixed-Integer Linear Programming (MILP) Formulation")
    st.caption("Industrial mathematical programming model formulated with IBM DOcplex for exact branch-and-cut optimization.")
    
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        st.markdown("""
        <div class="content-box">
            <h4 style="color:#1E40AF; margin-top:0; font-size:1rem; font-weight:700;">1. Sets and Indices</h4>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        \begin{aligned}
        & i \in I = \{M, A, T, AI\} \quad && \text{Specialist Consultants} \\
        & j \in J = \{1, 2, \dots, 12\} \quad && \text{Client Engagements}
        \end{aligned}
        """)

        st.markdown("""
        <div class="content-box">
            <h4 style="color:#1E40AF; margin-top:0; font-size:1rem; font-weight:700;">2. Decision Variables</h4>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        \begin{aligned}
        & x_{ij} \ge 0 \quad && \text{Continuous allocation of hours from consultant } i \text{ to client } j \\
        & y_{ij} \in \{0, 1\} \quad && \text{Binary assignment flag: 1 if consultant } i \text{ is assigned to } j, \, 0 \text{ otherwise} \\
        & p_{ij} \in \{0, 1\} \quad && \text{Binary Principal PM indicator: 1 if consultant } i \text{ leads project } j
        \end{aligned}
        """)

    with col_m2:
        st.markdown("""
        <div class="content-box">
            <h4 style="color:#1E40AF; margin-top:0; font-size:1rem; font-weight:700;">3. Model Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        \begin{aligned}
        & H_j \quad && \text{Total required delivery hours for client } j \\
        & R_{ij} \quad && \text{Hourly billing revenue rate for consultant } i \text{ on client } j \\
        & C_{ij} \quad && \text{Hourly consultant cost rate for consultant } i \text{ on client } j \\
        & S_{ij} \quad && \text{Mandatory minimum required hours of consultant } i \text{ on client } j \\
        & \text{PM}_j \quad && \text{Minimum supervisory hours required by the designated Principal PM} \\
        & \text{Cap}_i \quad && \text{Consultant weekly available capacity (default 40.0 hours)}
        \end{aligned}
        """)

        st.markdown("""
        <div class="content-box" style="border-left: 4px solid #2563EB;">
            <h4 style="color:#1E40AF; margin-top:0; font-size:1rem; font-weight:700;">4. Objective Function</h4>
            <div style="font-size:0.85rem; color:#475569;">Maximize total firm weekly contribution margin across all consultants and clients:</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        \max \quad Z = \sum_{i \in I} \sum_{j \in J} (R_{ij} - C_{ij}) \cdot x_{ij}
        """)

    st.markdown("""
    <div class="content-box">
        <h4 style="color:#1E40AF; margin-top:0; font-size:1rem; font-weight:700;">5. Constraint System (8 Rigorous Rule Sets)</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    \begin{aligned}
    & \text{(1) Capacity Limits:} & \sum_{j \in J} x_{ij} &\le \text{Cap}_i \quad &&\forall i \in I \\
    & \text{(2) Client Demand Fulfillment:} & \sum_{i \in I} x_{ij} &= H_j \quad &&\forall j \in J \\
    & \text{(3) Expertise Minimums:} & x_{ij} &\ge S_{ij} \quad &&\forall i \in I, \, j \in J \\
    & \text{(4) Assignment Linkage (Big-M):} & x_{ij} &\le H_j \cdot y_{ij} \quad &&\forall i \in I, \, j \in J \\
    & \text{(5) Exactly One Principal PM:} & \sum_{i \in I} p_{ij} &= 1 \quad &&\forall j \in J \\
    & \text{(6) PM Must Be Assigned:} & p_{ij} &\le y_{ij} \quad &&\forall i \in I, \, j \in J \\
    & \text{(7) Minimum PM Supervisory Hours:} & x_{ij} &\ge \text{PM}_j \cdot p_{ij} \quad &&\forall i \in I, \, j \in J \\
    & \text{(8) Variable Domains:} & x_{ij} &\ge 0, \; y_{ij}, p_{ij} \in \{0, 1\} \quad &&\forall i \in I, \, j \in J
    \end{aligned}
    """)

    st.markdown("""
    <div class="content-box-slate">
        <div style="font-size:0.85rem; font-weight:700; color:#1E40AF; text-transform:uppercase; letter-spacing:0.04em;">⚡ Optimization Technology & Solver Specifications</div>
        <div style="font-size:0.9rem; color:#334155; margin-top:4px;">
            The model is solved strictly using the <b>IBM ILOG CPLEX Optimizer</b> via the official Python DOcplex Mathematical Programming API (<code>docplex.mp.model.Model</code>). CPLEX employs advanced Branch-and-Cut, cutting plane generation, dynamic search, and presolve heuristics to guarantee exact global optimality in milliseconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SECTION 2: PARAMETERS & DATA EDITOR
# ==========================================
with tab_params:
    col_p_title, col_p_btn = st.columns([3, 1])
    with col_p_title:
        st.subheader("🎛️ Interactive Parameter Management & Data Editors")
        st.caption("Directly modify client demand, expertise minimums, billing rates, and delivery costs. Edits take effect immediately upon clicking Solve.")
    with col_p_btn:
        if st.button("🔄 Reset to Default Parameters", use_container_width=True):
            st.session_state.client_df = model_data.get_default_client_df()
            st.session_state.billing_df = model_data.get_default_billing_df()
            st.session_state.costs_df = model_data.get_default_costs_df()
            st.session_state.capacities_df = model_data.get_default_capacities_df()
            st.session_state.solution = solver.solve_consulting_allocation(
                st.session_state.client_df,
                st.session_state.billing_df,
                st.session_state.costs_df,
                st.session_state.capacities_df
            )
            st.toast("Reset all parameters to benchmark defaults!", icon="🔄")
            st.rerun()

    # Parameter Summary Metrics Ribbon
    tot_req = st.session_state.client_df["Required_Hours"].sum()
    tot_cap = st.session_state.capacities_df["Capacity_Hours"].sum()
    slack_diff = tot_cap - tot_req
    
    st.markdown(f"""
    <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:1rem;">
        <div style="flex:1; min-width:140px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 12px;">
            <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">Total Client Demand</div>
            <div style="font-size:1.15rem; font-weight:800; color:#0F172A;">{tot_req:.0f} hrs</div>
        </div>
        <div style="flex:1; min-width:140px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 12px;">
            <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">Total Firm Capacity</div>
            <div style="font-size:1.15rem; font-weight:800; color:#0F172A;">{tot_cap:.0f} hrs</div>
        </div>
        <div style="flex:1; min-width:140px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 12px;">
            <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">Firm Capacity Buffer</div>
            <div style="font-size:1.15rem; font-weight:800; color:#059669;">+{slack_diff:.0f} hrs ({slack_diff/tot_cap*100:.1f}%)</div>
        </div>
        <div style="flex:1; min-width:140px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 12px;">
            <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">Active Accounts</div>
            <div style="font-size:1.15rem; font-weight:800; color:#0F172A;">{len(st.session_state.client_df)} Clients</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs([
        "Client Requirements (Appendix A)",
        "Billing Rates (Appendix B)",
        "Delivery Costs (Appendix C)",
        "Consultant Capacities"
    ])
    
    with p_tab1:
        st.markdown("**Client Requirements, Priority & Minimum Hours Constraints**")
        client_col_config = {
            "Client": st.column_config.TextColumn(label="Client (j)", disabled=True),
            "Priority": st.column_config.SelectboxColumn(label="Priority", options=["Strategic", "Important", "Standard"]),
            "Required_Hours": st.column_config.NumberColumn(label="Required (Hⱼ)", min_value=1.0, step=1.0, format="%.0f hrs"),
            "M_Min": st.column_config.NumberColumn(label="Mₘᵢₙ (S_{M,j})", min_value=0.0, step=1.0, format="%.0f hrs"),
            "A_Min": st.column_config.NumberColumn(label="Aₘᵢₙ (S_{A,j})", min_value=0.0, step=1.0, format="%.0f hrs"),
            "T_Min": st.column_config.NumberColumn(label="Tₘᵢₙ (S_{T,j})", min_value=0.0, step=1.0, format="%.0f hrs"),
            "AI_Min": st.column_config.NumberColumn(label="AIₘᵢₙ (S_{AI,j})", min_value=0.0, step=1.0, format="%.0f hrs"),
            "PM_Min": st.column_config.NumberColumn(label="PMₘᵢₙ (PMⱼ)", min_value=0.0, step=1.0, format="%.0f hrs"),
        }
        edited_clients = st.data_editor(
            st.session_state.client_df,
            column_config=client_col_config,
            num_rows="dynamic",
            use_container_width=True,
            key="client_editor"
        )
        st.session_state.client_df = edited_clients
        
        c_req = edited_clients["Required_Hours"].sum()
        m_tot = edited_clients["M_Min"].sum()
        a_tot = edited_clients["A_Min"].sum()
        t_tot = edited_clients["T_Min"].sum()
        ai_tot = edited_clients["AI_Min"].sum()
        
        st.info(f"📊 **Current Totals:** Total Required Demand (Hⱼ) = **{c_req:.0f} hrs** | Mₘᵢₙ = **{m_tot:.0f}** | Aₘᵢₙ = **{a_tot:.0f}** | Tₘᵢₙ = **{t_tot:.0f}** | AIₘᵢₙ = **{ai_tot:.0f}**")

    with p_tab2:
        st.markdown("**Consultant Billing Rates ($ / hour)**")
        billing_col_config = {
            "Client": st.column_config.TextColumn(label="Client (j)", disabled=True),
            "M": st.column_config.NumberColumn(label="Marketing (R_{M,j})", min_value=0.0, step=100.0, format="$%d"),
            "A": st.column_config.NumberColumn(label="Advertising (R_{A,j})", min_value=0.0, step=100.0, format="$%d"),
            "T": st.column_config.NumberColumn(label="Technology (R_{T,j})", min_value=0.0, step=100.0, format="$%d"),
            "AI": st.column_config.NumberColumn(label="AI (R_{AI,j})", min_value=0.0, step=100.0, format="$%d"),
        }
        edited_billing = st.data_editor(
            st.session_state.billing_df,
            column_config=billing_col_config,
            use_container_width=True,
            key="billing_editor"
        )
        st.session_state.billing_df = edited_billing

    with p_tab3:
        st.markdown("**Consultant Delivery Costs ($ / hour)**")
        costs_col_config = {
            "Client": st.column_config.TextColumn(label="Client (j)", disabled=True),
            "M": st.column_config.NumberColumn(label="Marketing (C_{M,j})", min_value=0.0, step=100.0, format="$%d"),
            "A": st.column_config.NumberColumn(label="Advertising (C_{A,j})", min_value=0.0, step=100.0, format="$%d"),
            "T": st.column_config.NumberColumn(label="Technology (C_{T,j})", min_value=0.0, step=100.0, format="$%d"),
            "AI": st.column_config.NumberColumn(label="AI (C_{AI,j})", min_value=0.0, step=100.0, format="$%d"),
        }
        edited_costs = st.data_editor(
            st.session_state.costs_df,
            column_config=costs_col_config,
            use_container_width=True,
            key="costs_editor"
        )
        st.session_state.costs_df = edited_costs

    with p_tab4:
        st.markdown("**Consultant Weekly Available Capacities (hours/week)**")
        caps_col_config = {
            "Consultant": st.column_config.TextColumn(label="Consultant (i)", disabled=True),
            "Role": st.column_config.TextColumn(label="Role Specification", disabled=True),
            "Capacity_Hours": st.column_config.NumberColumn(label="Capacity (Capᵢ)", min_value=0.0, step=1.0, format="%.1f hrs"),
        }
        edited_caps = st.data_editor(
            st.session_state.capacities_df,
            column_config=caps_col_config,
            use_container_width=True,
            key="caps_editor"
        )
        st.session_state.capacities_df = edited_caps
        st.caption(f"Total Available Capacity (Capᵢ): **{edited_caps['Capacity_Hours'].sum():.1f} hrs/week**")

# ==========================================
# SECTION 3: VISUAL ANALYTICS & PLOTS
# ==========================================
with tab_charts:
    sol = st.session_state.solution
    if not sol.get("optimal", False):
        st.warning("Please solve an optimal instance first in the 'Optimization & Results' tab.")
    else:
        st.subheader("📊 Visual Optimization Analytics")
        st.caption("Interactive charts depicting allocation distribution, capacity utilization vs slack, client profitability ranking, and PM governance.")
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("##### 1. Consultant Work Allocation per Client (Hours)")
            alloc_raw = sol["allocation_df"]
            
            chart_data = []
            for _, r in alloc_raw.iterrows():
                for c in ["M", "A", "T", "AI"]:
                    hrs = r[f"{c}_Hours"]
                    if hrs > 0:
                        chart_data.append({
                            "Client": r["Client"],
                            "Consultant": model_data.CONSULTANT_NAMES[c],
                            "Hours": hrs,
                            "PM": "⭐️ PM" if r["PM_Assigned"] == c else "Team Member"
                        })
            cdf = pd.DataFrame(chart_data)
            if not cdf.empty:
                cdf["Hours_Label"] = cdf["Hours"].map(lambda h: f"{h:.0f}h" if h == int(h) else f"{h:.1f}h")
            
            fig_alloc = px.bar(
                cdf,
                x="Client",
                y="Hours",
                color="Consultant",
                text="Hours_Label" if not cdf.empty else None,
                color_discrete_map={
                    model_data.CONSULTANT_NAMES["M"]: "#2563EB",
                    model_data.CONSULTANT_NAMES["A"]: "#059669",
                    model_data.CONSULTANT_NAMES["T"]: "#D97706",
                    model_data.CONSULTANT_NAMES["AI"]: "#7C3AED"
                },
                barmode="stack",
                template="plotly_white"
            )
            fig_alloc.update_traces(
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#FFFFFF", size=11, family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif")
            )
            fig_alloc.update_layout(
                height=380,
                margin=dict(l=15, r=15, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11)
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

        with row1_col2:
            st.markdown("##### 2. Consultant Capacity Utilization vs Slack")
            udf = sol["utilization_df"]
            
            fig_util = go.Figure()
            fig_util.add_trace(go.Bar(
                name="Allocated Hours",
                x=udf["Consultant"],
                y=udf["Allocated_Hours"],
                marker_color="#2563EB",
                text=udf["Allocated_Hours"].map(lambda x: f"{x:.1f}h"),
                textposition="inside"
            ))
            fig_util.add_trace(go.Bar(
                name="Unallocated Slack Hours",
                x=udf["Consultant"],
                y=udf["Slack_Hours"],
                marker_color="#E2E8F0",
                text=udf["Slack_Hours"].map(lambda x: f"{x:.1f}h" if x > 0 else ""),
                textposition="inside"
            ))
            fig_util.update_layout(
                barmode="stack",
                template="plotly_white",
                height=380,
                margin=dict(l=15, r=15, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Hours / Week"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11)
            )
            st.plotly_chart(fig_util, use_container_width=True)

        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("##### 3. Client Contribution Margin Ranking ($)")
            sorted_clients = sol["allocation_df"].sort_values(by="Client_Margin", ascending=True)
            
            fig_margin = px.bar(
                sorted_clients,
                x="Client_Margin",
                y="Client",
                orientation="h",
                color="Priority",
                color_discrete_map={"Strategic": "#4F46E5", "Important": "#0284C7", "Standard": "#64748B"},
                text="Client_Margin",
                template="plotly_white"
            )
            fig_margin.update_traces(texttemplate="$%{text:,.0f}", textposition="inside")
            fig_margin.update_layout(
                height=400,
                margin=dict(l=15, r=15, t=30, b=20),
                xaxis_title="Contribution Margin ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11)
            )
            st.plotly_chart(fig_margin, use_container_width=True)

        with row2_col2:
            st.markdown("##### 4. Principal PM Leadership Distribution")
            pm_counts = sol["utilization_df"].copy()
            
            fig_pm = px.pie(
                pm_counts,
                values="PM_Assignments_Count",
                names="Role",
                hole=0.48,
                color="Role",
                color_discrete_map={
                    model_data.CONSULTANT_NAMES["M"]: "#2563EB",
                    model_data.CONSULTANT_NAMES["A"]: "#059669",
                    model_data.CONSULTANT_NAMES["T"]: "#D97706",
                    model_data.CONSULTANT_NAMES["AI"]: "#7C3AED"
                },
                template="plotly_white"
            )
            fig_pm.update_layout(
                height=400,
                margin=dict(l=15, r=15, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(text=f"12 PMs", x=0.5, y=0.5, font_size=15, font_family="-apple-system", showarrow=False)],
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11)
            )
            st.plotly_chart(fig_pm, use_container_width=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("##### 5. 3D Optimization Solution Space & Optimal Value Frontier")
        st.caption("Interactive 3D optimization manifold mapping Client Project Hours (Hⱼ), Total Delivery Cost (Cⱼ), and the resulting Optimal Contribution Margin (Zⱼ). Rotate, zoom, and hover to inspect individual optimal client coordinates and the fitted value frontier surface.")
        
        # 3D Solution Space Calculations
        adf = sol["allocation_df"].copy()
        x_vals = adf["Req_Hours"].values
        y_vals = adf["Client_Cost"].values
        z_vals = adf["Client_Margin"].values
        
        # Peak margin client
        best_idx = int(np.argmax(z_vals))
        best_client = adf.iloc[best_idx]
        
        # Stat cards for 3D view
        st.markdown(f"""
        <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:1rem;">
            <div style="flex:1; min-width:160px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 14px;">
                <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">Global Optimal Objective (Z*)</div>
                <div style="font-size:1.15rem; font-weight:800; color:#1E40AF;">${sol['objective_value']:,.2f}</div>
            </div>
            <div style="flex:1; min-width:160px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 14px;">
                <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">Highest Margin Client</div>
                <div style="font-size:1.15rem; font-weight:800; color:#059669;">{best_client['Client']} (${best_client['Client_Margin']:,.0f})</div>
            </div>
            <div style="flex:1; min-width:160px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 14px;">
                <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">Total Hours Allocated</div>
                <div style="font-size:1.15rem; font-weight:800; color:#0F172A;">{x_vals.sum():.0f} hrs across 12 Clients</div>
            </div>
            <div style="flex:1; min-width:160px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 14px;">
                <div style="font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;">3D Value Plane Model</div>
                <div style="font-size:1.15rem; font-weight:800; color:#6366F1;">Fitted Solution Frontier</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Fit optimal frontier surface plane: z = a*x + b*y + c
        A = np.c_[x_vals, y_vals, np.ones(x_vals.shape[0])]
        C_coeffs, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
        
        x_surf = np.linspace(float(x_vals.min()) - 1, float(x_vals.max()) + 1, 15)
        y_surf = np.linspace(float(y_vals.min()) - 2000, float(y_vals.max()) + 2000, 15)
        X_grid, Y_grid = np.meshgrid(x_surf, y_surf)
        Z_grid = C_coeffs[0]*X_grid + C_coeffs[1]*Y_grid + C_coeffs[2]
        
        fig_3d = go.Figure()
        
        # Add fitted value frontier plane
        fig_3d.add_trace(go.Surface(
            x=x_surf,
            y=y_surf,
            z=Z_grid,
            colorscale=[[0, "rgba(219, 234, 254, 0.35)"], [1, "rgba(191, 219, 254, 0.45)"]],
            showscale=False,
            name="Fitted Solution Frontier",
            hoverinfo="skip"
        ))
        
        # Add stem drop lines from each point down to z = 0
        for idx_pt, row_pt in adf.iterrows():
            xi = row_pt["Req_Hours"]
            yi = row_pt["Client_Cost"]
            zi = row_pt["Client_Margin"]
            fig_3d.add_trace(go.Scatter3d(
                x=[xi, xi],
                y=[yi, yi],
                z=[0, zi],
                mode="lines",
                line=dict(color="#CBD5E1", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            ))
            
        # Custom hover text list
        hover_texts = []
        for _, r in adf.iterrows():
            txt = (
                f"<b>Client: {r['Client']} ({r['Priority']})</b><br>"
                f"Required Hours: {r['Req_Hours']:.0f} hrs<br>"
                f"Delivery Cost: ${r['Client_Cost']:,.2f}<br>"
                f"<b>Optimal Margin: ${r['Client_Margin']:,.2f}</b><br>"
                f"Profit Margin: {r['Margin_Pct']:.1f}%<br>"
                f"Designated PM: {r['PM_Assigned']}"
            )
            hover_texts.append(txt)
            
        # Add optimal scatter points
        fig_3d.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="markers+text",
            text=adf["Client"].values,
            textposition="top center",
            textfont=dict(size=11, color="#0F172A", family="-apple-system, sans-serif"),
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=8,
                color=z_vals,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text="Optimal Margin ($)", font=dict(size=11, color="#334155")),
                    thickness=14,
                    len=0.75,
                    x=1.02
                ),
                showscale=True,
                line=dict(color="#FFFFFF", width=1.5)
            ),
            name="Client Optimal Points"
        ))
        
        # Highlight peak margin client
        fig_3d.add_trace(go.Scatter3d(
            x=[best_client["Req_Hours"]],
            y=[best_client["Client_Cost"]],
            z=[best_client["Client_Margin"]],
            mode="markers+text",
            text=[f"⭐ Peak: {best_client['Client']}"],
            textposition="bottom center",
            textfont=dict(size=12, color="#B45309", family="-apple-system, sans-serif"),
            hoverinfo="skip",
            marker=dict(
                size=12,
                color="#F59E0B",
                symbol="diamond",
                line=dict(color="#78350F", width=2)
            ),
            name="Max Margin Engagement"
        ))
        
        fig_3d.update_layout(
            title=dict(
                text="<b>3D Optimal Solution Space: Client Coordinates on the Contribution Frontier</b>",
                font=dict(size=13, color="#1E40AF")
            ),
            height=540,
            margin=dict(l=10, r=10, t=40, b=10),
            scene=dict(
                xaxis=dict(
                    title="Engagement Hours (Hⱼ)",
                    backgroundcolor="rgba(248, 250, 252, 0.6)",
                    gridcolor="#E2E8F0",
                    showbackground=True
                ),
                yaxis=dict(
                    title="Delivery Cost (Cⱼ)",
                    backgroundcolor="rgba(248, 250, 252, 0.6)",
                    gridcolor="#E2E8F0",
                    showbackground=True
                ),
                zaxis=dict(
                    title="Contribution Margin (Zⱼ)",
                    backgroundcolor="rgba(248, 250, 252, 0.6)",
                    gridcolor="#E2E8F0",
                    showbackground=True
                ),
                camera=dict(eye=dict(x=1.65, y=-1.65, z=1.15))
            ),
            legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# SECTION 4: OPTIMIZATION & RESULTS
# ==========================================
with tab_results:
    sol = st.session_state.solution
    
    if not sol.get("optimal", False):
        st.error(f"⚠️ Optimization Status: {sol.get('status', 'Infeasible')}")
        for err in sol.get("errors", []):
            st.warning(err)
        st.info("💡 Suggestion: Check if total consultant capacities are at least equal to total client required hours, or if specific expertise minimums exceed available hours.")
    else:
        # Top KPI Metric Grid (Responsive CSS Grid)
        total_req = st.session_state.client_df["Required_Hours"].sum()
        total_cap = st.session_state.capacities_df["Capacity_Hours"].sum()
        firm_load_pct = (total_req / total_cap * 100) if total_cap > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Contribution Margin (Z*)</div>
                <div class="kpi-value">${sol['objective_value']:,.2f}</div>
                <div class="kpi-delta kpi-delta-positive">+{sol['margin_percentage']:.1f}% Profit Margin</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Firm Revenue</div>
                <div class="kpi-value">${sol['total_revenue']:,.2f}</div>
                <div class="kpi-delta kpi-delta-neutral">Gross Client Billing</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Delivery Cost</div>
                <div class="kpi-value">${sol['total_cost']:,.2f}</div>
                <div class="kpi-delta kpi-delta-neutral">Specialist Delivery Cost</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Capacity Utilized</div>
                <div class="kpi-value">{total_req:.0f} / {total_cap:.0f} <span style="font-size:1rem; font-weight:600; color:#64748B;">hrs</span></div>
                <div class="kpi-delta kpi-delta-positive">{firm_load_pct:.1f}% Firm Load</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">CPLEX Runtime</div>
                <div class="kpi-value">{sol['solve_time']*1000:.1f} <span style="font-size:1rem; font-weight:600; color:#64748B;">ms</span></div>
                <div class="kpi-delta kpi-delta-positive">Integer Optimal (B&C)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        # Detailed Client Allocation Table
        st.subheader("📋 Client Allocation & Principal Project Manager Assignments")
        st.caption("Detailed breakdown showing hours allocated per consultant, designated Principal PM, and financial outcome per client engagement.")
        
        alloc_df = sol["allocation_df"].copy()
        
        # Format currency columns for display
        display_df = alloc_df.copy()
        display_df["Client_Revenue"] = display_df["Client_Revenue"].map("${:,.2f}".format)
        display_df["Client_Cost"] = display_df["Client_Cost"].map("${:,.2f}".format)
        display_df["Client_Margin"] = display_df["Client_Margin"].map("${:,.2f}".format)
        display_df["Margin_Pct"] = display_df["Margin_Pct"].map("{:.1f}%".format)
        display_df["PM_Assigned"] = display_df["PM_Assigned"].map(lambda x: f"⭐️ {x}")
        
        # Subscript RHS of variables with underscore in column headers
        alloc_col_renames = {
            "Req_Hours": "Required (Hⱼ)",
            "PM_Assigned": "Designated PM (pᵢⱼ)",
            "PM_Min_Req": "PM Min (PMⱼ)",
            "M_Hours": "M Hours (x_{M,j})",
            "A_Hours": "A Hours (x_{A,j})",
            "T_Hours": "T Hours (x_{T,j})",
            "AI_Hours": "AI Hours (x_{AI,j})",
            "Client_Revenue": "Revenue (Rⱼ)",
            "Client_Cost": "Cost (Cⱼ)",
            "Client_Margin": "Margin (Zⱼ)",
            "Margin_Pct": "Profit Margin (%)"
        }
        display_df = display_df.rename(columns=alloc_col_renames)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Consultant Utilization Summary
        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
        st.subheader("👥 Consultant Utilization, Slack & Margin Contribution")
        
        util_df = sol["utilization_df"].copy()
        display_util = util_df.copy()
        display_util["Total_Revenue"] = display_util["Total_Revenue"].map("${:,.2f}".format)
        display_util["Total_Cost"] = display_util["Total_Cost"].map("${:,.2f}".format)
        display_util["Contribution_Margin"] = display_util["Contribution_Margin"].map("${:,.2f}".format)
        display_util["Utilization_Pct"] = display_util["Utilization_Pct"].map("{:.1f}%".format)
        
        # Subscript RHS of variables with underscore in utilization column headers
        util_col_renames = {
            "Capacity": "Capacity (Capᵢ)",
            "Allocated_Hours": "Allocated Hours (xᵢ)",
            "Slack_Hours": "Slack (sᵢ)",
            "Utilization_Pct": "Utilization (%)",
            "PM_Assignments_Count": "Projects Led (pᵢⱼ)",
            "Total_Revenue": "Total Revenue (Rᵢ)",
            "Total_Cost": "Total Cost (Cᵢ)",
            "Contribution_Margin": "Contribution Margin (Zᵢ)"
        }
        display_util = display_util.rename(columns=util_col_renames)
        
        st.dataframe(display_util, use_container_width=True, hide_index=True)
        
        # CSV Download Option
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        csv_data = alloc_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Optimal Allocation Results (CSV)",
            data=csv_data,
            file_name="optimal_consulting_allocation.csv",
            mime="text/csv",
            use_container_width=True
        )
