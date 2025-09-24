# rbc_TranThiHa_app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Real Business Cycle Model Simulation", layout="wide")

# ---------- Header with (optional) logos ----------
def _img_to_data_url(p: Path) -> str:
    if not p or not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def add_header(left_logo: str,
               title: str,
               subtitle_top: str,
               subtitle_bottom: str = "",
               right_logo: str = ""):
    """
    Header: left logo + centered titles.
    Tự động bỏ các thẻ trống nếu không có subtitle_bottom hoặc right_logo.
    """
    def img_tag(path):
        if path and Path(path).exists():
            src = _img_to_data_url(Path(path))
            return f'<img src="{src}" alt="Logo">'
        return ""

    left_html  = img_tag(left_logo)
    right_html = img_tag(right_logo)

    bottom_html = f'<div class="app-subtitle-bottom">{subtitle_bottom}</div>' if subtitle_bottom else ""

    st.markdown(
        f"""
        <style>
        .app-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 15px 40px;
            background: linear-gradient(90deg, #2b5876, #4e4376);
            border-bottom: 3px solid #e0e0e0;
            box-shadow: 0 4px 12px rgba(0,0,0,.15);
            color: white;
        }}
        .app-header img {{ height: 90px; }}
        .app-center {{ text-align: center; flex-grow: 1; margin: 0 30px; }}
        .app-title {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
        .app-subtitle-top {{ font-size: 20px; font-weight: 600; color: #ffd966; }}
        .app-subtitle-bottom {{ font-size: 18px; font-weight: 500; color: #ffecb3; }}
        </style>
        <div class="app-header">
            {left_html}
            <div class="app-center">
                <div class="app-title">{title}</div>
                <div class="app-subtitle-top">{subtitle_top}</div>
                {bottom_html}
            </div>
            {right_html}
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Call header (only NEU logo & name; no faculty line, no right logo) ---
add_header(
    left_logo="PNG1.png",                         # logo trường (đặt trong repo)
    title="Real Business Cycle Model Simulation (Full RBC)",
    subtitle_top="National Economics University", # chỉ tên trường
    subtitle_bottom="",                            # ẩn dòng 'Faculty'
    right_logo=""                                  # không hiện logo phải
)

# ---------- Sidebar: parameters ----------
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

# ---------- Helpers ----------
def steady_state(alpha, beta, eta, phi, theta, delta):
    """Solve steady state for (k, n, y, c, i, r, w) with A_ss = 1."""
    r_ss = 1.0 / beta - (1.0 - delta)
    if r_ss <= 0:
        raise ValueError("Invalid (β, δ): r_ss ≤ 0.")

    # From r = α k^{α-1} n^{1-α}  ⇒  k = [α/r]^{1/(1-α)} · n
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

    # bisection for n
    a, b = 1e-4, 0.95
    fa, fb = resid(a), resid(b)
    if np.sign(fa) == np.sign(fb):
        found = False
        for mult in [0.5, 1, 2, 5, 10]:
            def resid2(n):
                if n <= 1e-6 or n >= 0.99:
                    return 1e6
                k = Cn * n
                y = (k ** alpha) * (n ** (1 - alpha))
                i = delta * k
                c = y - i
                w = (1 - alpha) * (k ** alpha) * (n ** (-alpha))
                lhs = (theta * mult) * (c ** eta) * (n ** phi)
                return lhs - w
            fa2, fb2 = resid2(a), resid2(b)
            if np.sign(fa2) != np.sign(fb2):
                resid = resid2
                found = True
                break
        if not found:
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
    """A,B,C,D,F,G,H,J,K,L,M for Uhlig form; controls order = [y,c,n,w,r,i]."""
    A = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    B = np.array([[-(1.0 - delta)], [-alpha], [0.0], [0.0], [1.0], [0.0]])
    C = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, -delta],           # k_{t+1}-(1-δ)k_t-δ i_t=0
        [1.0, 0.0, -(1.0 - alpha), 0.0, 0.0, 0.0],   # y = A + αk + (1-α)n
        [0.0, eta, phi, -1.0, 0.0, 0.0],             # η c + φ n = w
        [-1.0, 0.0, 1.0, 1.0, 0.0, 0.0],             # w = y - n
        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],             # r = y - k
        [-y_ss, c_ss, 0.0, 0.0, 0.0, i_ss]           # y_ss y = c_ss c + i_ss i
    ])
    D = np.array([[0.0], [-1.0], [0.0], [0.0], [0.0], [0.0]])   # A_t only in production
    F = 0.0; G = 0.0; H = 0.0; L = 0.0; M = 0.0                  # Expectational block (Euler)
    J = np.array([[0.0, eta / beta, 0.0, 0.0, -r_ss, 0.0]])
    K = np.array([[0.0, -eta / beta, 0.0, 0.0, 0.0, 0.0]])
    return A, B, C, D, F, G, H, J, K, L, M

def uhlig(alpha, beta, eta, phi, delta, rho_a, y_ss, c_ss, i_ss, r_ss):
    """Solve for P,Q,R,S (1 shock ⇒ Q scalar)."""
    A, B, C, D, F, G, H, J, K, L, M = build_mats(alpha, beta, eta, phi, delta, y_ss, c_ss, i_ss, r_ss)
    Csolve = lambda X: np.linalg.solve(C, X)

    # Quadratic for P
    a = float(F - J @ Csolve(A))
    b = float(-(J @ Csolve(B) - G + K @ Csolve(A)))
    c = float(-K @ Csolve(B) + H)
    disc = b * b - 4 * a * c
    if disc < 0:
        disc = np.real(disc)
    P1 = (-b + np.sqrt(disc)) / (2 * a)
    P2 = (-b - np.sqrt(disc)) / (2 * a)
    P  = float(P1 if abs(P1) <= abs(P2) else P2)

    # R
    R = - Csolve(A * P + B)

    # Q (scalar)
    JC_A = float(J @ Csolve(A))
    KC_A = float(K @ Csolve(A))
    JC_D = float(J @ Csolve(D))
    KC_D = float(K @ Csolve(D))
    JR   = float(J @ R)
    LHS = rho_a * (F - JC_A) + (JR + F * P + G - KC_A)
    RHS = (JC_D - L) * rho_a + KC_D - M
    Q = float(RHS / LHS)

    # S
    S = - Csolve(A * Q + D)
    return P, Q, R, S

def irf(P, Q, R, S, rho_a, eps, T, percent=True):
    """IRF for a one-time shock ε at t=0 (store 0..T)."""
    T = int(T)
    Atil = np.zeros(T + 1)
    Ktil = np.zeros(T + 1)
    Y = np.zeros(T + 1); C = np.zeros(T + 1); L = np.zeros(T + 1)
    W = np.zeros(T + 1); Rr = np.zeros(T + 1); I = np.zeros(T + 1)

    Atil[0] = eps
    for t in range(T):
        Y[t]  = R[0, 0] * Ktil[t] + S[0, 0] * Atil[t]
        C[t]  = R[1, 0] * Ktil[t] + S[1, 0] * Atil[t]
        L[t]  = R[2, 0] * Ktil[t] + S[2, 0] * Atil[t]
        W[t]  = R[3, 0] * Ktil[t] + S[3, 0] * Atil[t]
        Rr[t] = R[4, 0] * Ktil[t] + S[4, 0] * Atil[t]
        I[t]  = R[5, 0] * Ktil[t] + S[5, 0] * Atil[t]
        Ktil[t + 1] = P * Ktil[t] + Q * Atil[t]
        Atil[t + 1] = rho_a * Atil[t]

    Y[T]  = R[0, 0] * Ktil[T] + S[0, 0] * Atil[T]
    C[T]  = R[1, 0] * Ktil[T] + S[1, 0] * Atil[T]
    L[T]  = R[2, 0] * Ktil[T] + S[2, 0] * Atil[T]
    W[T]  = R[3, 0] * Ktil[T] + S[3, 0] * Atil[T]
    Rr[T] = R[4, 0] * Ktil[T] + S[4, 0] * Atil[T]
    I[T]  = R[5, 0] * Ktil[T] + S[5, 0] * Atil[T]

    series = {"Y": Y, "C": C, "L": L, "W": W, "R": Rr, "I": I, "K": Ktil, "A": Atil}
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

# ---------- Top: Model Overview ----------
with st.expander("Model Overview (Full RBC) / Tổng quan mô hình", expanded=True):
    st.markdown("### Households")
    st.markdown("The representative household maximizes lifetime utility:")
    st.latex(r"""
    \max_{\{c_t,k_{t+1},n_t\}} \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t u(c_t,n_t),\quad
    u(c_t,n_t)=\frac{c_t^{1-\eta}-1}{1-\eta}-\theta\frac{n_t^{1+\phi}}{1+\phi}
    """)
    st.markdown("**Budget and capital accumulation:**")
    st.latex(r"c_t+i_t=r_tk_t+w_t n_t,\qquad k_{t+1}=(1-\delta)k_t+i_t")

    st.markdown("### Production")
    st.latex(r"y_t=A_t k_t^{\alpha} n_t^{1-\alpha}")

    st.markdown("### Stochastic process")
    st.latex(r"\log A_t=(1-\rho_A)\log(A^{ss})+\rho_A\log(A_{t-1})+\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,\sigma_e^2)")

    st.markdown("### Equilibrium condition")
    st.latex(r"y_t=c_t+i_t")

    st.markdown("### Non-linear Equations")
    st.latex(r"\theta c_t^{\eta}n_t^{\phi}=w_t")
    st.latex(r"c_t^{-\eta}=\beta\,\mathbb{E}_t\!\left[c_{t+1}^{-\eta}(1+r_{t+1}-\delta)\right]")
    st.latex(r"k_{t+1}=(1-\delta)k_t+i_t")
    st.latex(r"y_t=A_t k_t^{\alpha} n_t^{1-\alpha}")
    st.latex(r"r_t=A_t \alpha k_t^{\alpha-1} n_t^{1-\alpha}")
    st.latex(r"w_t=A_t(1-\alpha)k_t^{\alpha}n_t^{-\alpha}")

    st.markdown("### Log-linearized (Uhlig rules; tildes = log-deviation)")
    st.latex(r"\eta \tilde c_t + \phi \tilde n_t = \tilde w_t")
    st.latex(r"\mathbb{E}_t(\tilde c_{t+1})-\tilde c_t = \beta r^{ss}\eta\,\mathbb{E}_t(\tilde r_{t+1})")
    st.latex(r"\tilde k_{t+1}=(1-\delta)\tilde k_t+\delta \tilde i_t")
    st.latex(r"\tilde y_t=\tilde A_t+\alpha \tilde k_t+(1-\alpha)\tilde n_t")
    st.latex(r"\tilde r_t=\tilde y_t-\tilde k_t,\qquad \tilde w_t=\tilde y_t-\tilde n_t")
    st.latex(r"y^{ss}\tilde y_t=c^{ss}\tilde c_t+i^{ss}\tilde i_t,\qquad \tilde A_t=\rho_A \tilde A_{t-1}+\varepsilon_t")

# ---------- Compute SS & Solution ----------
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### Model Overview (quick recap)")
    st.write("Adjust parameters in the sidebar and explore the IRFs / simulations.")

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

# ---------- Tabs ----------
tabs = st.tabs([
    "Model Overview",
    "Steady-State Values",
    "Impulse Response Functions",
    "Stochastic Simulation"
])

with tabs[0]:
    st.write("This tab summarizes the model blocks and linear solution. See above.")

with tabs[1]:
    st.markdown("### Steady-State Values")
    df = pd.DataFrame({
        "Variable": ["Interest Rate (r)", "Wage (w)", "Output (y)", "Investment (i)",
                     "Consumption (c)", "Labor (n)", "Capital (k)"],
        "Value": [r_ss, w_ss, y_ss, i_ss, c_ss, n_ss, k_ss]
    })
    df["Value"] = df["Value"].map(float)
    st.table(df)

with tabs[2]:
    st.markdown("### Impulse Response Functions")
    ser = irf(P, Q, R, S, rho_a, impulse, T_irf, percent=True)
    plot_grid(ser, T_irf, "IRFs to a one-time TFP shock ε at t=0 (log-deviation, %)")

with tabs[3]:
    st.markdown("### Stochastic Simulation (AR(1) TFP shocks)")
    if stoch_on:
        if seed_on:
            np.random.seed(int(seed))
        eps = np.random.normal(0.0, sigma_e, size=T_sim)
        A = np.zeros(T_sim + 1); K = np.zeros(T_sim + 1)
        Y = np.zeros(T_sim + 1); C = np.zeros(T_sim + 1); L = np.zeros(T_sim + 1)
        W = np.zeros(T_sim + 1); Rr = np.zeros(T_sim + 1); I = np.zeros(T_sim + 1)

        for t in range(T_sim):
            Y[t]  = R[0, 0] * K[t] + S[0, 0] * A[t]
            C[t]  = R[1, 0] * K[t] + S[1, 0] * A[t]
            L[t]  = R[2, 0] * K[t] + S[2, 0] * A[t]
            W[t]  = R[3, 0] * K[t] + S[3, 0] * A[t]
            Rr[t] = R[4, 0] * K[t] + S[4, 0] * A[t]
            I[t]  = R[5, 0] * K[t] + S[5, 0] * A[t]
            K[t + 1] = P * K[t] + Q * A[t]
            A[t + 1] = rho_a * A[t] + eps[t]

        ser2 = {"Y": Y, "C": C, "L": L, "W": W, "R": Rr, "I": I, "K": K, "A": A}
        for k in ser2:
            ser2[k] = 100.0 * np.array(ser2[k])
        T_show = min(200, T_sim)
        plot_grid({kk: vv[:T_show + 1] for kk, vv in ser2.items()}, T_show,
                  "Sample path (first 200 periods shown), units: % log-deviation")
    else:
        st.info("Tick **Enable Stochastic Simulation** in the sidebar to run.")


