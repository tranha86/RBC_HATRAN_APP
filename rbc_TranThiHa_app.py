# rbc_TranThiHa_app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import base64
from pathlib import Path

st.set_page_config(page_title="Real Business Cycle Model Simulation", layout="wide")

# ======================= HEADER (no stray whitespace) =======================
def _img_b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def add_header_simple(logo_path: str, title: str, subtitle: str):
    """Header with one left logo and centered title/subtitle; uses inline HTML safely."""
    logo = _img_b64(logo_path) if logo_path else ""
    logo_html = f'<img src="{logo}" alt="Logo" style="height:90px;margin-right:16px;">' if logo else ""
    html = (
        "<div style='display:flex;align-items:center;justify-content:flex-start;"
        "padding:15px 40px;background:linear-gradient(90deg,#2b5876,#4e4376);"
        "border-bottom:3px solid #e0e0e0;box-shadow:0 4px 12px rgba(0,0,0,.15);"
        "color:#fff;border-radius:6px;'>"
        f"{logo_html}"
        "<div style='flex:1;text-align:center;'>"
        f"<div style='font-size:28px;font-weight:700;margin-bottom:8px;'>{title}</div>"
        f"<div style='font-size:20px;font-weight:600;color:#ffd966;'>{subtitle}</div>"
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# --- Call header ---
add_header_simple(
    logo_path="PNG1.png",  # đặt "" nếu không muốn hiện logo
    title="Real Business Cycle Model Simulation (Full RBC)",
    subtitle="National Economics University"
)

# ======================= SIDEBAR: Parameters =======================
st.sidebar.header("Model Parameters")
alpha = st.sidebar.number_input("Capital Share (α)", 0.05, 0.95, 0.35, 0.01, format="%.3f")
beta  = st.sidebar.number_input("Discount Factor (β)", 0.90, 0.999, 0.985, 0.001, format="%.3f")
eta   = st.sidebar.number_input("Consumption Elasticity (η)", 0.1, 10.0, 1.0, 0.1, format="%.1f")
phi   = st.sidebar.number_input("Labor Elasticity (φ)", 0.1, 10.0, 1.7, 0.1, format="%.1f")
theta = st.sidebar.number_input("Leisure Weight (θ)", 0.01, 50.0, 11.0, 0.1, format="%.1f")
delta = st.sidebar.number_input("Depreciation Rate (δ)", 0.001, 0.20, 0.025, 0.001, format="%.3f")
rho_a = st.sidebar.number_input("TFP Persistence (ρₐ)", 0.0, 0.999, 0.95, 0.005, format="%.3f")

st.sidebar.markdown("---")
impulse = st.sidebar.number_input("Impulse size ε (log-deviation)", 0.0, 0.10, 0.01, 0.001, format="%.3f")
T_irf   = st.sidebar.slider("IRF horizon (periods)", 10, 200, 80, 1)

st.sidebar.markdown("---")
stoch_on = st.sidebar.checkbox("Enable Stochastic Simulation", True)
T_sim    = st.sidebar.slider("Stochastic periods", 100, 5000, 500, 50)
sigma_e  = st.sidebar.number_input("Shock s.d. σₑ", 0.0, 0.10, 0.01, 0.001, format="%.3f")
seed_on  = st.sidebar.checkbox("Set random seed", True)
seed     = st.sidebar.number_input("Seed", 0, 10000, 42, 1)

# ======================= Helpers =======================
def steady_state(alpha, beta, eta, phi, theta, delta):
    r_ss = 1.0 / beta - (1.0 - delta)
    if r_ss <= 0:
        raise ValueError("Invalid (β, δ): r_ss ≤ 0.")
    # r = α k^{α-1} n^{1-α}  ⇒  k = [α/r]^{1/(1-α)} · n
    Cn = (alpha / r_ss) ** (1.0 / (1.0 - alpha))

    def resid(n):
        if n <= 1e-6 or n >= 0.99:
            return 1e6
        k = Cn * n
        y = (k ** alpha) * (n ** (1 - alpha))
        i = delta * k
        c = y - i
        w = (1 - alpha) * (k ** alpha) * (n ** (-alpha))
        lhs = theta * (c ** eta) * (n ** phi)
        return lhs - w

    a, b = 1e-4, 0.95
    fa, fb = resid(a), resid(b)
    if np.sign(fa) == np.sign(fb):
        # fallback an toàn
        n = 1.0 / 3.0
        k = Cn * n
        y = (k ** alpha) * (n ** (1 - alpha))
        i = delta * k
        c = y - i
        w = (1 - alpha) * (k ** alpha) * (n ** (-alpha))
        return k, n, y, c, i, r_ss, w

    for _ in range(120):
        m = 0.5 * (a + b)
        fm = resid(m)
        if np.sign(fm) == np.sign(resid(a)):
            a = m
        else:
            b = m
        if abs(fm) < 1e-12 or (b - a) < 1e-10:
            break

    n = 0.5 * (a + b)
    k = Cn * n
    y = (k ** alpha) * (n ** (1 - alpha))
    i = delta * k
    c = y - i
    w = (1 - alpha) * (k ** alpha) * (n ** (-alpha))
    return k, n, y, c, i, r_ss, w

def build_mats(alpha, beta, eta, phi, delta, y_ss, c_ss, i_ss, r_ss):
    A = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    B = np.array([[-(1.0 - delta)], [-alpha], [0.0], [0.0], [1.0], [0.0]])
    C = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, -delta],
        [1.0, 0.0, -(1.0 - alpha), 0.0, 0.0, 0.0],
        [0.0, eta, phi, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-y_ss, c_ss, 0.0, 0.0, 0.0, i_ss]
    ])
    D = np.array([[0.0], [-1.0], [0.0], [0.0], [0.0], [0.0]])
    F = 0.0; G = 0.0; H = 0.0; L = 0.0; M = 0.0
    J = np.array([[0.0, eta / beta, 0.0, 0.0, -r_ss, 0.0]])
    K = np.array([[0.0, -eta / beta, 0.0, 0.0, 0.0, 0.0]])
    return A, B, C, D, F, G, H, J, K, L, M

def uhlig(alpha, beta, eta, phi, delta, rho_a, y_ss, c_ss, i_ss, r_ss):
    A, B, C, D, F, G, H, J, K, L, M = build_mats(alpha, beta, eta, phi, delta, y_ss, c_ss, i_ss, r_ss)
    solveC = lambda X: np.linalg.solve(C, X)

    # P từ phương trình bậc hai
    a = float(F - J @ solveC(A))
    b = float(-(J @ solveC(B) - G + K @ solveC(A)))
    c = float(-K @ solveC(B) + H)
    disc = b*b - 4*a*c
    if disc < 0:
        disc = np.real(disc)
    P1 = (-b + np.sqrt(disc)) / (2*a) if a != 0 else 0.0
    P2 = (-b - np.sqrt(disc)) / (2*a) if a != 0 else 0.0
    P  = float(P1 if abs(P1) <= abs(P2) else P2)

    # R
    R = -solveC(A * P + B)

    # Q (scalar)
    JC_A = float(J @ solveC(A))
    KC_A = float(K @ solveC(A))
    JC_D = float(J @ solveC(D))
    KC_D = float(K @ solveC(D))
    JR   = float(J @ R)
    LHS = rho_a * (F - JC_A) + (JR + F*P + G - KC_A)
    RHS = (JC_D - L) * rho_a + KC_D - M
    Q = float(RHS / LHS)

    # S
    S = -solveC(A * Q + D)
    return P, Q, R, S

def irf(P, Q, R, S, rho_a, eps, T, percent=True):
    T = int(T)
    Atil = np.zeros(T + 1)
    Ktil = np.zeros(T + 1)
    Y = np.zeros(T + 1); Cc = np.zeros(T + 1); L = np.zeros(T + 1)
    W = np.zeros(T + 1); Rr = np.zeros(T + 1); I = np.zeros(T + 1)
    Atil[0] = eps
    for t in range(T):
        Y[t]  = R[0, 0]*Ktil[t] + S[0, 0]*Atil[t]
        Cc[t] = R[1, 0]*Ktil[t] + S[1, 0]*Atil[t]
        L[t]  = R[2, 0]*Ktil[t] + S[2, 0]*Atil[t]
        W[t]  = R[3, 0]*Ktil[t] + S[3, 0]*Atil[t]
        Rr[t] = R[4, 0]*Ktil[t] + S[4, 0]*Atil[t]
        I[t]  = R[5, 0]*Ktil[t] + S[5, 0]*Atil[t]
        Ktil[t + 1] = P*Ktil[t] + Q*Atil[t]
        Atil[t + 1] = rho_a*Atil[t]
    Y[T]  = R[0, 0]*Ktil[T] + S[0, 0]*Atil[T]
    Cc[T] = R[1, 0]*Ktil[T] + S[1, 0]*Atil[T]
    L[T]  = R[2, 0]*Ktil[T] + S[2, 0]*Atil[T]
    W[T]  = R[3, 0]*Ktil[T] + S[3, 0]*Atil[T]
    Rr[T] = R[4, 0]*Ktil[T] + S[4, 0]*Atil[T]
    I[T]  = R[5, 0]*Ktil[T] + S[5, 0]*Atil[T]
    series = {"Y": Y, "C": Cc, "L": L, "W": W, "R": Rr, "I": I, "K": Ktil, "A": Atil}
    if percent:
        for k in series:
            series[k] = 100.0 * series[k]
    return series

def plot_grid(series, T, title):
    labels = ["Y", "C", "L", "W", "R", "I", "K", "A"]
    fig = plt.figure(figsize=(10, 9))
    for i, lab in enumerate(labels, start=1):
        ax = fig.add_subplot(3, 3, i)
        ax.plot(np.arange(T + 1), series[lab][:T + 1], linewidth=1.2)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(lab)
        ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(title)

# ======================= Model Overview =======================
with st.expander("Model Overview (Full RBC) / Tổng quan mô hình", expanded=True):
    st.markdown("### Households")
    st.latex(r"\max_{\{c_t,k_{t+1},n_t\}} \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t u(c_t,n_t)")
    st.latex(r"u(c_t,n_t)=\frac{c_t^{1-\eta}-1}{1-\eta}-\theta\frac{n_t^{1+\phi}}{1+\phi}")
    st.latex(r"c_t+i_t=r_t k_t + w_t n_t,\qquad k_{t+1}=(1-\delta)k_t+i_t")
    st.markdown("### Production")
    st.latex(r"y_t=A_t k_t^{\alpha} n_t^{1-\alpha}")
    st.markdown("### Stochastic process")
    st.latex(r"\log A_t=(1-\rho_A)\log(A^{ss})+\rho_A\log(A_{t-1})+\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,\sigma_e^2)")
    st.markdown("### Equilibrium condition")
    st.latex(r"y_t = c_t + i_t")

# ======================= Compute Steady State & Solution =======================
col1, col2 = st.columns([1, 2], gap="large")
with col2:
    st.markdown("### Steady State & Linear Solution")
    try:
        k_ss, n_ss, y_ss, c_ss, i_ss, r_ss, w_ss = steady_state(alpha, beta, eta, phi, theta, delta)
        P, Q, R, S = uhlig(alpha, beta, eta, phi, delta, rho_a, y_ss, c_ss, i_ss, r_ss)
        st.write(f"**P = {P:.6f}**, **Q = {Q:.6f}**")
        st.write("**R'** =", np.round(R.flatten(), 6))
        st.write("**S'** =", np.round(S.flatten(), 6))
    except Exception as e:
        st.error(f"Steady state / solution error: {e}")
        st.stop()

# ======================= Tabs =======================
# ---------- Model Overview (chi tiết, render LaTeX chuẩn) ----------
with st.expander("Model Overview (Full RBC) / Tổng quan mô hình", expanded=True):
    # ===== Households =====
    st.markdown("## Households / Hộ gia đình đại diện")
    st.markdown(
        "- The representative household maximizes lifetime utility / "
        "Hộ gia đình đại diện tối đa hóa hữu dụng trọn đời:"
    )
    st.latex(r"\max_{\{c_t,k_{t+1},n_t\}} \; \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \, u(c_t,n_t)")
    st.latex(r"u(c_t,n_t)=\frac{c_t^{1-\eta}-1}{1-\eta} \;-\; \theta \frac{n_t^{1+\phi}}{1+\phi}")
    st.markdown(
        "- Parameters: $0<\beta<1$ (discount), $\eta>0$ (inv. IES), "
        "$\phi>0$ (Frisch elasticity inverse), $\\theta>0$ (leisure weight)."
    )

    # ===== Budget & Capital =====
    st.markdown("### Budget & Capital Accumulation / Ràng buộc ngân sách & tích lũy vốn")
    st.latex(r"c_t + i_t \;=\; r_t\,k_t \;+\; w_t\,n_t")
    st.latex(r"k_{t+1} \;=\; (1-\delta)\,k_t \;+\; i_t")
    st.markdown(
        "- $c_t$: consumption; $i_t$: investment; $k_t$: capital; $n_t$: labor; "
        "$r_t$: rental rate of capital; $w_t$: wage; $0<\delta<1$: depreciation."
    )

    # ===== Production =====
    st.markdown("## Production / Sản xuất")
    st.latex(r"y_t \;=\; A_t\,k_t^{\alpha}\,n_t^{\,1-\alpha}, \qquad 0<\alpha<1")
    st.markdown(
        "- Competitive firm with Cobb–Douglas technology / Công ty cạnh tranh với công nghệ Cobb–Douglas."
    )

    # ===== Stochastic process =====
    st.markdown("## Stochastic Process for TFP / Quá trình ngẫu nhiên của TFP")
    st.latex(r"\log A_t \;=\; (1-\rho_A)\log A^{ss} \;+\; \rho_A \log A_{t-1} \;+\; \varepsilon_t")
    st.latex(r"\varepsilon_t \sim \mathcal{N}(0,\sigma_{\varepsilon}^{2}), \qquad 0\le \rho_A<1")
    st.markdown("- $A^{ss}$ is steady-state TFP / $A^{ss}$ là mức TFP trạng thái dừng.")

    # ===== Equilibrium conditions =====
    st.markdown("## Market-Clearing / Cân bằng thị trường")
    st.latex(r"y_t \;=\; c_t \;+\; i_t")

    # ===== First-order conditions (nonlinear) =====
    st.markdown("## Non-linear First-Order Conditions / Điều kiện bậc nhất (phi tuyến)")
    st.latex(r"\theta\, c_t^{\eta}\, n_t^{\phi} \;=\; w_t \quad \text{(intratemporal labor-leisure)}")
    st.latex(r"c_t^{-\eta} \;=\; \beta\,\mathbb{E}_t\!\Big[c_{t+1}^{-\eta}\,\big(1+r_{t+1}-\delta\big)\Big] \quad \text{(Euler)}")
    st.latex(r"k_{t+1} \;=\; (1-\delta)k_t \;+\; i_t")
    st.latex(r"y_t \;=\; A_t k_t^{\alpha} n_t^{\,1-\alpha}")
    st.latex(r"r_t \;=\; A_t \alpha k_t^{\alpha-1} n_t^{\,1-\alpha}, \qquad w_t \;=\; A_t(1-\alpha)k_t^{\alpha} n_t^{-\alpha}")

    # ===== Log-linearization =====
    st.markdown("## Log-linearization (Uhlig / Christiano) / Tuyến tính hóa log")
    st.markdown(
        "Let tildes denote log-deviations from steady state: "
        "$\\tilde x_t = \\log x_t - \\log x^{ss}$. Then the linearized system:"
    )
    st.latex(r"\eta\,\tilde c_t \;+\; \phi\,\tilde n_t \;=\; \tilde w_t")
    st.latex(r"\mathbb{E}_t(\tilde c_{t+1}) - \tilde c_t \;=\; \beta r^{ss}\eta\,\mathbb{E}_t(\tilde r_{t+1})")
    st.latex(r"\tilde k_{t+1} \;=\; (1-\delta)\tilde k_t \;+\; \delta\,\tilde i_t")
    st.latex(r"\tilde y_t \;=\; \tilde A_t \;+\; \alpha\,\tilde k_t \;+\; (1-\alpha)\,\tilde n_t")
    st.latex(r"\tilde r_t \;=\; \tilde y_t - \tilde k_t, \qquad \tilde w_t \;=\; \tilde y_t - \tilde n_t")
    st.latex(r"y^{ss}\,\tilde y_t \;=\; c^{ss}\,\tilde c_t \;+\; i^{ss}\,\tilde i_t, \qquad \tilde A_t \;=\; \rho_A \tilde A_{t-1} + \varepsilon_t")

    # ===== Solution method =====
    st.markdown("## Solution Method / Phương pháp nghiệm")
    st.markdown(
        "- We use the **Method of Undetermined Coefficients** (Christiano, 2002; Uhlig, 1999) "
        "to solve the linearized system for policy rules "
        "$k_{t+1} = P\\,k_t + Q\\,A_t$, $x_t = R\\,k_t + S\\,A_t$ for "
        "$x_t\\in\\{y_t,c_t,n_t,w_t,r_t,i_t\\}$. "
        "These matrices $(P,Q,R,S)$ được tính trong phần mã bên dưới và dùng để tạo IRF & mô phỏng."
    )
