import numpy as np
import matplotlib.pyplot as plt

from cell_models import protocols
from cell_models.ohara_rudy import OharaRudyModel


# ============================================================
# 1. RUN O'HARA-RUDY-CIPA MODEL
# ============================================================

print("Initializing O'Hara-Rudy-CiPA model...")

model = OharaRudyModel()

protocol = protocols.PacedProtocol(
    model_name="OR"
)

print("Running simulation...")

trace = model.generate_response(
    protocol,
    is_no_ion_selective=False
)

time = np.asarray(trace.t)
voltage = np.asarray(trace.y)

print(f"Simulation points: {len(time)}")
print(f"Time range: {time[0]:.2f} - {time[-1]:.2f} ms")


# ============================================================
# 2. ACTION POTENTIAL MEASUREMENTS
# ============================================================

# Find first major depolarization
baseline_mask = time < 10

if np.any(baseline_mask):
    resting_voltage = np.mean(voltage[baseline_mask])
else:
    resting_voltage = voltage[0]

peak_index = np.argmax(voltage)
peak_voltage = voltage[peak_index]
peak_time = time[peak_index]

amplitude = peak_voltage - resting_voltage

apd50_voltage = resting_voltage + 0.50 * amplitude
apd90_voltage = resting_voltage + 0.10 * amplitude


def find_crossing_after_peak(target_voltage):
    after_peak = np.where(
        (time > peak_time) &
        (voltage <= target_voltage)
    )[0]

    if len(after_peak) == 0:
        return None

    return time[after_peak[0]] - peak_time


apd50 = find_crossing_after_peak(apd50_voltage)
apd90 = find_crossing_after_peak(apd90_voltage)


print("\n================ ACTION POTENTIAL =================")
print(f"Resting membrane potential : {resting_voltage:.2f} mV")
print(f"Peak voltage               : {peak_voltage:.2f} mV")
print(f"Peak time                  : {peak_time:.2f} ms")
print(f"Amplitude                  : {amplitude:.2f} mV")
print(f"APD50                      : {apd50:.2f} ms")
print(f"APD90                      : {apd90:.2f} ms")


# ============================================================
# 3. GET IONIC CURRENTS
# ============================================================

current_info = trace.current_response_info

if current_info is None:
    raise RuntimeError(
        "Current response information was not generated."
    )


current_names = [
    "I_Na",
    "I_NaL",
    "I_To",
    "I_CaL",
    "I_CaNa",
    "I_CaK",
    "I_Kr",
    "I_Ks",
    "I_K1",
    "I_NaCa",
    "I_NaCa_ss",
    "I_NaK",
    "I_pCa",
    "I_bNa",
    "I_bCa",
    "I_Cab",
    "I_Kb",
]


currents = {}

for name in current_names:
    try:
        currents[name] = np.asarray(
            current_info.get_current(name)
        )
    except Exception:
        print(f"Warning: could not retrieve {name}")


print("\n================ IONIC CURRENTS =================")

for name, values in currents.items():
    print(
        f"{name:10s} "
        f"min={np.min(values):8.3f} "
        f"max={np.max(values):8.3f}"
    )


# ============================================================
# 4. FULL 6-SECOND ACTION POTENTIAL
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(time, voltage)

plt.xlabel("Time (ms)")
plt.ylabel("Membrane Voltage (mV)")
plt.title("O'Hara-Rudy-CiPA — Ventricular Action Potentials")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "ord_full_action_potential.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. FIRST ACTION POTENTIAL — PHASE ANALYSIS
# ============================================================

first_ap_mask = (time >= 0) & (time <= 350)

t_ap = time[first_ap_mask]
v_ap = voltage[first_ap_mask]

plt.figure(figsize=(14, 6))

plt.plot(t_ap, v_ap, linewidth=2)

# Phase labels
plt.axvspan(
    peak_time - 2,
    peak_time + 3,
    alpha=0.15
)

plt.axhline(
    apd50_voltage,
    linestyle="--",
    alpha=0.5
)

plt.axhline(
    apd90_voltage,
    linestyle="--",
    alpha=0.5
)

plt.text(
    peak_time + 3,
    peak_voltage - 10,
    "Phase 0\nRapid depolarization",
    fontsize=10
)

plt.text(
    25,
    25,
    "Phase 1",
    fontsize=10
)

plt.text(
    80,
    10,
    "Phase 2\nPlateau / slow repolarization",
    fontsize=10
)

plt.text(
    220,
    -50,
    "Phase 3\nRepolarization",
    fontsize=10
)

plt.text(
    300,
    -82,
    "Phase 4\nResting state",
    fontsize=10
)

plt.xlabel("Time (ms)")
plt.ylabel("Membrane Voltage (mV)")
plt.title(
    "O'Hara-Rudy-CiPA — Action Potential Phases"
)

plt.xlim(0, 350)
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "ord_action_potential_phases.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. MAJOR CURRENTS — FULL SCALE
# ============================================================

major_currents = [
    "I_Na",
    "I_CaL",
    "I_Kr",
    "I_Ks",
    "I_K1",
]

plt.figure(figsize=(14, 6))

for name in major_currents:
    if name in currents:
        plt.plot(
            time,
            currents[name],
            label=name
        )

plt.xlabel("Time (ms)")
plt.ylabel("Current")
plt.title(
    "O'Hara-Rudy-CiPA — Major Ionic Currents"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "ord_ionic_currents_full.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. ZOOMED IONIC CURRENTS
# ============================================================

zoom_mask = (time >= 0) & (time <= 350)

plt.figure(figsize=(14, 7))

for name in major_currents:

    if name in currents:

        plt.plot(
            time[zoom_mask],
            currents[name][zoom_mask],
            label=name,
            linewidth=1.8
        )

plt.xlabel("Time (ms)")
plt.ylabel("Current")
plt.title(
    "O'Hara-Rudy-CiPA — Ionic Currents During One Action Potential"
)

plt.xlim(0, 350)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "ord_ionic_currents_zoomed.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. INDIVIDUAL CURRENT PANELS
# ============================================================

fig, axes = plt.subplots(
    len(major_currents),
    1,
    figsize=(14, 14),
    sharex=True
)

for ax, name in zip(axes, major_currents):

    if name in currents:

        ax.plot(
            time[zoom_mask],
            currents[name][zoom_mask],
            linewidth=1.5
        )

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (ms)")

fig.suptitle(
    "O'Hara-Rudy-CiPA — Individual Major Ionic Currents",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "ord_individual_currents.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. ALL IONIC CURRENTS
# ============================================================

plt.figure(figsize=(15, 9))

for name, values in currents.items():

    plt.plot(
        time[zoom_mask],
        values[zoom_mask],
        label=name,
        linewidth=1
    )

plt.xlabel("Time (ms)")
plt.ylabel("Current")
plt.title(
    "O'Hara-Rudy-CiPA — All Recorded Ionic Currents"
)

plt.xlim(0, 350)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "ord_all_ionic_currents.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n================================================")
print("O'HARA-RUDY-CiPA BASELINE ANALYSIS COMPLETE")
print("================================================")

print("\nGenerated files:")

print("1. ord_full_action_potential.png")
print("2. ord_action_potential_phases.png")
print("3. ord_ionic_currents_full.png")
print("4. ord_ionic_currents_zoomed.png")
print("5. ord_individual_currents.png")
print("6. ord_all_ionic_currents.png")

print("\nBaseline biomarkers:")

print(f"RMP   = {resting_voltage:.2f} mV")
print(f"Peak  = {peak_voltage:.2f} mV")
print(f"APD50 = {apd50:.2f} ms")
print(f"APD90 = {apd90:.2f} ms")

print("\nAnalysis finished.")