import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_impedance(R, L, C, f):
    ω = 2 * np.pi * f
    X_L = ω * L
    X_C = 1 / (ω * C) if C > 0 else 0
    X = X_L - X_C
    Z = np.sqrt(R**2 + X**2)
    φ = np.arctan2(X, R)
    return ω, X_L, X_C, X, Z, φ

def power_values(V, R, X):
    Z_sq = R**2 + X**2
    I = V / np.sqrt(Z_sq)
    P = V**2 * R / Z_sq
    Q = V**2 * X / Z_sq
    PF = R / np.sqrt(Z_sq)
    φ = np.arctan2(X, R)
    return P, Q, I, PF, φ

def capacitor_correction(P, φ_initial, PF_target, V, ω):
    φ_target = np.arccos(PF_target)
    Qc = P * (np.tan(φ_initial) - np.tan(φ_target))
    if Qc <= 0:
        return 0, 0, φ_target
    C = Qc / (V**2 * ω)
    return C, Qc, φ_target

st.set_page_config(page_title="Power Factor Correction", layout="wide")
st.title("Power Factor Improvement Simulator")

with st.sidebar:
    st.header("Inputs")
    V = st.number_input("Supply Voltage (V)", 230.0)
    f = st.number_input("Frequency (Hz)", 50.0)
    R = st.number_input("Resistance R (Ω)", 10.0)
    L = st.number_input("Inductance L (H)", 0.05)
    C = st.number_input("Series Capacitance C (F)", 0.0, format="%.6f")
    PF_target = st.slider("Target Power Factor", 0.7, 0.999, 0.95, 0.01)

ω, X_L, X_C, X, Z, φ = compute_impedance(R, L, C, f)
P, Q, I, PF, φ = power_values(V, R, X)
C_shunt, Qc, φ_target = capacitor_correction(P, φ, PF_target, V, ω)

st.subheader("Results Summary")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Circuit Parameters**")
    st.write(f"R = {R:.3f} Ω, L = {L:.4f} H, C = {C:.6f} F")
    st.write(f"X_L = {X_L:.3f} Ω, X_C = {X_C:.3f} Ω, X_net = {X:.3f} Ω")
    st.write(f"|Z| = {Z:.3f} Ω,  φ = {np.degrees(φ):.2f}°")

with col2:
    st.markdown("**Power Values**")
    st.write(f"P = {P:.3f} W,  Q = {Q:.3f} VAR")
    st.write(f"I = {I:.4f} A,  PF = {PF:.4f}")

st.markdown("---")
st.subheader("Power Triangle Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.plot([0, P], [0, 0], 'k-', lw=3)
    ax1.plot([P, P], [0, Q], 'k-', lw=3)
    ax1.plot([0, P], [0, Q], 'k--', lw=2)
    ax1.text(P/2, -0.1*abs(Q), f"P={P:.2f}", color='blue', ha='center')
    ax1.text(P, Q/2, f"Q={Q:.2f}", color='red', va='center')
    ax1.set_title("Before Correction")
    ax1.set_xlabel("Active Power (W)")
    ax1.set_ylabel("Reactive Power (VAR)")
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    if C_shunt > 0:
        Q_new = Q - Qc
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.plot([0, P], [0, 0], 'k-', lw=3)
        ax2.plot([P, P], [0, Q_new], 'k-', lw=3)
        ax2.plot([0, P], [0, Q_new], 'k--', lw=2)
        ax2.text(P/2, -0.1*abs(Q_new), f"P={P:.2f}", color='blue', ha='center')
        ax2.text(P, Q_new/2, f"Q={Q_new:.2f}", color='red', va='center')
        ax2.set_title("After Correction")
        ax2.set_xlabel("Active Power (W)")
        ax2.set_ylabel("Reactive Power (VAR)")
        ax2.grid(True)
        st.pyplot(fig2)
    else:
        st.info("No correction needed — PF already near target.")

st.subheader("Capacitor Suggestion")
if C_shunt > 0:
    st.success(f"Add capacitor: **C = {C_shunt*1e6:.3f} µF**, Qc = {Qc:.3f} VAR")
else:
    st.info("No capacitor required.")

data = {
    "Before": {"PF": PF, "P (W)": P, "Q (VAR)": Q, "φ (deg)": np.degrees(φ)},
}
if C_shunt > 0:
    data["After"] = {"PF": np.cos(φ_target), "P (W)": P, "Q (VAR)": Q - Qc, "φ (deg)": np.degrees(φ_target)}

df = pd.DataFrame(data)
st.dataframe(df.T)
