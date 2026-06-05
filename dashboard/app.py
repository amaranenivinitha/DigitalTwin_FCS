import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from streamlit_autorefresh import st_autorefresh

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Aircraft Digital Twin",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("✈️ Aircraft Digital Twin Dashboard")

# =========================================================
# AUTO REFRESH
# =========================================================

st.sidebar.header("Live Refresh")

refresh_rate = st.sidebar.slider(
    "Refresh Interval (sec)",
    1,
    10,
    2
)

st_autorefresh(
    interval=refresh_rate * 1000,
    key="live_dashboard"
)

# =========================================================
# LIVE TELEMETRY SECTION
# =========================================================

st.header("🛰️ Real-Time Telemetry Stream")

LIVE_FILE = "data/live/live_telemetry.csv"

# =========================================================
# CHECK FILE
# =========================================================

if not os.path.exists(LIVE_FILE):

    st.warning(
        "⚠️ Live telemetry file not found."
    )

    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

try:

    live_df = pd.read_csv(LIVE_FILE)

except Exception as e:

    st.error(f"Error loading telemetry: {e}")

    st.stop()

# =========================================================
# EMPTY DATA CHECK
# =========================================================

if len(live_df) < 5:

    st.warning(
        "Waiting for telemetry data..."
    )

    st.stop()

# =========================================================
# LAST SAMPLES
# =========================================================

latest = live_df.tail(300)

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

latest_nominal = latest["gyro_nominal"].iloc[-1]
latest_faulty = latest["gyro_faulty"].iloc[-1]

fault_count = latest["fault_label"].sum()

current_time = latest["time_s"].iloc[-1]

col1.metric(
    "Current Time",
    f"{current_time:.2f} s"
)

col2.metric(
    "Nominal Gyro",
    f"{latest_nominal:.3f}"
)

col3.metric(
    "Faulty Gyro",
    f"{latest_faulty:.3f}"
)

col4.metric(
    "Fault Count",
    int(fault_count)
)

# =========================================================
# LIVE PLOT
# =========================================================

fig_live, ax_live = plt.subplots(
    figsize=(14, 6)
)

ax_live.plot(
    latest["time_s"],
    latest["gyro_nominal"],
    label="Nominal Signal",
    linewidth=2
)

ax_live.plot(
    latest["time_s"],
    latest["gyro_faulty"],
    label="Faulty Signal",
    linewidth=2
)

# =========================================================
# FAULT REGION
# =========================================================

fault_region = latest[
    latest["fault_label"] == 1
]

if len(fault_region) > 0:

    ax_live.scatter(
        fault_region["time_s"],
        fault_region["gyro_faulty"],
        s=30,
        label="Detected Fault"
    )

# =========================================================
# PLOT FORMATTING
# =========================================================

ax_live.set_title(
    "Real-Time Aircraft Telemetry"
)

ax_live.set_xlabel(
    "Time (s)"
)

ax_live.set_ylabel(
    "Gyro Signal"
)

ax_live.legend()

ax_live.grid(True)

st.pyplot(fig_live)

# =========================================================
# LIVE STATUS
# =========================================================

latest_fault = latest["fault_label"].iloc[-1]

if latest_fault == 1:

    st.error(
        "🚨 LIVE FAULT DETECTED"
    )

else:

    st.success(
        "✅ Aircraft Operating Normally"
    )

# =========================================================
# DATA TABLE
# =========================================================

st.subheader("Latest Telemetry Samples")

st.dataframe(
    latest.tail(15),
    use_container_width=True
)

# =========================================================
# SYSTEM INFO
# =========================================================

st.markdown("---")

st.info(
    "Real-Time AI Aircraft Digital Twin Monitoring Active"
)