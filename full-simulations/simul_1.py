import numpy as np
import matplotlib.pyplot as plt

# Constants and Parameters
rate_constants = [1.5e-4, 2.0e-4, 2.5e-4]  # Example k for low, medium, high UV intensity
no2_concentration = 50e-6  # Initial NO2 concentration (mol/L)
voc_list = {
    "Acetone": [10e-6, 20e-6, 30e-6],  # VOC concentrations (mol/L)
    "Ethanol": [15e-6, 25e-6, 35e-6],
    "Isopropanol": [12e-6, 22e-6, 32e-6]
}
time = np.linspace(0, 3600, 500)  # Simulate up to 1 hour (3600 seconds) with 500 intervals


# Function to calculate ozone production
def ozone_production(k, no2, voc, t):
    return k * no2 * voc * t


# Large-Scale Simulation
results = {}  # Dictionary to store simulation results

for uv_index, k in enumerate(rate_constants):
    uv_label = ["Low UV", "Medium UV", "High UV"][uv_index]
    results[uv_label] = {}

    for voc_name, concentrations in voc_list.items():
        results[uv_label][voc_name] = []

        for concentration in concentrations:
            ozone = ozone_production(k, no2_concentration, concentration, time)
            results[uv_label][voc_name].append((concentration, ozone))

# Plot Results
plt.figure(figsize=(16, 10))

for uv_label, voc_data in results.items():
    for voc_name, simulations in voc_data.items():
        for concentration, ozone in simulations:
            plt.plot(
                time, ozone,
                label=f"{voc_name} ({concentration * 1e6:.1f} ppm, {uv_label})"
            )

plt.title("Large-Scale Ozone Production Simulation")
plt.xlabel("Time (s)")
plt.ylabel("Ozone Concentration (mol/L)")
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Conditions", fontsize='small')
plt.tight_layout()
plt.show()
