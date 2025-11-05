import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_series_impedance(R, L, C_series, f):
    """Compute series reactances and impedance (Z = R + j(XL - XC))."""
    omega = 2 * np.pi * f
    X_L = omega * L
    X_C_series = 1/(omega * C_series) if C_series and C_series > 0 else 0.0
    X_net = X_L - X_C_series
    Z_mag = np.sqrt(R**2 + X_net**2)
    phi = np.arctan2(X_net, R)
    return {'omega': omega, 'X_L': X_L, 'X_C_series': X_C_series, 'X_net': X_net, 'Z_mag': Z_mag, 'phi': phi}

def powers_from_voltage(R, X_net, V_rms):
    """Given R and X_net for series load at voltage V_rms, compute P, Q, S, PF, I."""
    Z_sq = R**2 + X_net**2
    S_apparent = V_rms**2 / np.sqrt(Z_sq)
    I_rms = V_rms / np.sqrt(Z_sq)
    P_active = V_rms**2 * R / Z_sq
    Q_reactive = V_rms**2 * X_net / Z_sq
    phi = np.arctan2(X_net, R)
    PF = np.cos(phi)
    return {'I_rms': I_rms, 'P': P_active, 'Q': Q_reactive, 'S': S_apparent, 'phi': phi, 'PF': PF}

def classify_load(phi, tol_deg=1.0):
    """Return string classification based on phase angle (radians)."""
    deg = np.degrees(phi)
    if abs(deg) <= tol_deg:
        return 'Resistive (unity)'
    elif deg > 0:
        return f'Inductive (lagging), φ = {deg:.2f}°'
    else:
        return f'Capacitive (leading), φ = {deg:.2f}°'

def required_shunt_capacitor(P, phi_initial, PF_target, V_rms, omega):
    """Compute shunt capacitor value needed for PF correction."""
    tan1 = np.tan(phi_initial)
    PF_target = np.clip(PF_target, 0.01, 0.9999)
    phi_target = np.arccos(PF_target)
    tan2 = np.tan(phi_target) * np.sign(phi_initial)
    Qc = P * (tan1 - tan2)
    if Qc <= 0:
        return 0.0, Qc, phi_target
    C = Qc / (V_rms**2 * omega)
    return C, Qc, phi_target

# Streamlit app layout
st.set_page_config(page_title="Power Factor Improvement Simulator", layout="wide")
st.title("Smart Power Factor Improvement & Load Analysis Simulator")

with st.sidebar:
    st.header("Inputs")
    V_rms = st.number_input("Supply RMS Voltage (V)", value=230.0, min_value=1.0)
    f = st.number_input("Frequency (Hz)", value=50.0, min_value=1.0)
    R = st.number_input("Resistance R (Ω)", value=10.0, min_value=0.0)
    L = st.number_input("Inductance L (H)", value=0.05, min_value=0.0)
    C_series = st.number_input("Series Capacitance C (F)", value=0.0, min_value=0.0, format="%.6f")
    pf_target = st.slider("Target Power Factor", 0.7, 0.999, 0.95, 0.01)
    st.markdown("---")

imp = compute_series_impedance(R, L, C_series, f)
pw = powers_from_voltage(R, imp['X_net'], V_rms)
classification = classify_load(imp['phi'])
C_shunt, Qc_required, phi_target = required_shunt_capacitor(pw['P'], imp['phi'], pf_target, V_rms, imp['omega'])
C_shunt_uF = C_shunt * 1e6

st.subheader("Results")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Load / Impedance Details**")
    st.write(f"R = {R:.4g} Ω")
    st.write(f"L = {L:.6g} H → X_L = {imp['X_L']:.4f} Ω")
    if C_series > 0:
        st.write(f"Series C = {C_series:.6g} F → X_C = {imp['X_C_series']:.4f} Ω")
    st.write(f"Net Reactance X = {imp['X_net']:.4f} Ω")
    st.write(f"|Z| = {imp['Z_mag']:.4f} Ω")
    st.write(f"Phase angle φ = {np.degrees(imp['phi']):.3f}°")

with col2:
    st.markdown("**Power Quantities**")
    st.write(f"Current I = {pw['I_rms']:.6f} A")
    st.write(f"Active Power P = {pw['P']:.6f} W")
    st.write(f"Reactive Power Q = {pw['Q']:.6f} VAR")
    st.write(f"Apparent Power |S| = {pw['S']:.6f} VA")
    st.write(f"Power Factor = {pw['PF']:.6f}")
    st.write(f"Load Classification: **{classification}**")

st.subheader("Capacitor Suggestion (Shunt Correction)")
if C_shunt <= 0:
    st.info("No capacitor required (PF already near target or cannot be corrected by adding capacitance).")
else:
    st.success(f"Add shunt capacitor: **C = {C_shunt_uF:.3f} µF** (Qc = {Qc_required:.3f} VAR)")
    st.write(f"Target PF angle = {np.degrees(phi_target):.2f}°")

st.subheader("Power Triangle Visualization")

# Create two side-by-side columns
col1, col2 = st.columns(2)

# ---------- Power Triangle (Before Correction) ----------
with col1:
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    P, Q = pw['P'], pw['Q']

    # Draw the triangle
    ax1.plot([0, P], [0, 0], 'k-', linewidth=3)        # Base (P)
    ax1.plot([P, P], [0, Q], 'k-', linewidth=3)        # Vertical (Q)
    ax1.plot([0, P], [0, Q], 'k--', linewidth=2)       # Hypotenuse (S)

    # Labels
    ax1.text(P / 2, -0.05 * abs(Q + 1), f"P = {P:.2f} W", ha='center', fontsize=9, color='blue')
    ax1.text(P + 0.02 * abs(P), Q / 2, f"Q = {Q:.2f} VAR", va='center', fontsize=9, color='red')

    # Format
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('Active Power (W)')
    ax1.set_ylabel('Reactive Power (VAR)')
    ax1.set_title('Power Triangle (Before Correction)')
    ax1.grid(True)
    ax1.set_xlim(0, max(P * 1.3, 1))
    ax1.set_ylim(0, max(Q * 1.3, 1))
    st.pyplot(fig1)


# ---------- Power Triangle (After Correction) ----------
with col2:
    if C_shunt > 0:
        Q_after = Q - Qc_required
        fig2, ax2 = plt.subplots(figsize=(4, 4))

        # Draw the triangle
        ax2.plot([0, P], [0, 0], 'k-', linewidth=3)          # Base (P)
        ax2.plot([P, P], [0, Q_after], 'k-', linewidth=3)    # Vertical (Q_after)
        ax2.plot([0, P], [0, Q_after], 'k--', linewidth=2)   # Hypotenuse (S_after)

        # Labels
        ax2.text(P / 2, -0.05 * abs(Q_after + 1), f"P = {P:.2f} W", ha='center', fontsize=9, color='blue')
        ax2.text(P + 0.02 * abs(P), Q_after / 2, f"Q = {Q_after:.2f} VAR", va='center', fontsize=9, color='red')

        # Format
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('Active Power (W)')
        ax2.set_ylabel('Reactive Power (VAR)')
        ax2.set_title('Power Triangle (After Correction)')
        ax2.grid(True)
        ax2.set_xlim(0, max(P * 1.3, 1))
        ax2.set_ylim(0, max(max(Q, Q_after) * 1.3, 1))
        st.pyplot(fig2)
    else:
        st.info("No capacitor correction applied — only before-correction triangle shown.")

st.subheader("Summary Table")
before = {'PF': pw['PF'], 'P (W)': pw['P'], 'Q (VAR)': pw['Q'], 'I (A)': pw['I_rms'], 'φ (deg)': np.degrees(pw['phi'])}
if C_shunt > 0:
    after = {'PF': np.cos(phi_target), 'P (W)': pw['P'], 'Q (VAR)': pw['Q'] - Qc_required,
             'I (A)': pw['I_rms'], 'φ (deg)': np.degrees(phi_target)}
    df = pd.DataFrame({'Before': before, 'After': after})
else:
    df = pd.DataFrame({'Before': before})
st.dataframe(df.T)
st.markdown("---")