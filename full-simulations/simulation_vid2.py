import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# ---------------------------------
# KINETIC MODEL SETUP
# ---------------------------------
# Simple photochemical mechanism (example rates, not real data):
# NO2 --(hv)--> NO + O
# O + O2 -> O3
# VOC + O -> products + O3 (simplified)

# Assume O2 is constant in large excess
O2_conc = 0.21

# Rate constants (fictitious for demonstration)
k_ph = 1e-3     # Photolysis rate of NO2 [s^-1]
k_o3 = 1e6       # O3 formation from O + O2
k_voc = 5e5      # VOC reaction with O leading to O3 formation

# Initial concentrations (arbitrary units)
NO2_0 = 1e-6
NO_0 = 0.0
O_0 = 0.0
O3_0 = 0.0
VOC_0 = 1e-6

y0 = [NO2_0, NO_0, O_0, O3_0, VOC_0]

# Time handling
time_passed = 0.0
dt = 5.0  # seconds per animation frame (example)
time_data = []
NO2_data = []
NO_data = []
O_data = []
O3_data = []
VOC_data = []

def reactions(y, t, k_ph, k_o3, k_voc, O2):
    NO2, NO, O, O3, VOC = y
    r_ph = k_ph * NO2
    r_o3_form = k_o3 * O * O2
    r_voc_rxn = k_voc * VOC * O

    dNO2_dt = -r_ph
    dNO_dt = r_ph
    dO_dt = r_ph - r_o3_form - r_voc_rxn
    dO3_dt = r_o3_form + r_voc_rxn
    dVOC_dt = -r_voc_rxn
    return [dNO2_dt, dNO_dt, dO_dt, dO3_dt, dVOC_dt]


# ---------------------------------
# VISUALIZATION SETUP
# ---------------------------------
# We'll represent each species with a set of particles in a 2D box.
# The number of particles is scaled to concentration for illustration.

box_size = 10
particle_scale = 5e7  # scale factor to convert concentration to particle count

def concentrations_to_particle_counts(NO2_c, NO_c, O_c, O3_c, VOC_c):
    # Convert concentration to approximate particle counts
    # You can tune particle_scale for best visibility.
    NO2_count = max(int(NO2_c * particle_scale), 0)
    NO_count  = max(int(NO_c * particle_scale), 0)
    O_count   = max(int(O_c * particle_scale), 0)
    O3_count  = max(int(O3_c * particle_scale), 0)
    VOC_count = max(int(VOC_c * particle_scale), 0)
    return NO2_count, NO_count, O_count, O3_count, VOC_count

def init_positions(count):
    if count <= 0:
        return np.empty((0,2))
    return np.random.rand(count, 2) * box_size

def init_velocities(count):
    if count <= 0:
        return np.empty((0,2))
    return (np.random.rand(count, 2) - 0.5) * 0.05

# Initially, determine particle counts and positions
initial_counts = concentrations_to_particle_counts(NO2_0, NO_0, O_0, O3_0, VOC_0)
positions = {
    "NO2": init_positions(initial_counts[0]),
    "NO": init_positions(initial_counts[1]),
    "O": init_positions(initial_counts[2]),
    "O3": init_positions(initial_counts[3]),
    "VOC": init_positions(initial_counts[4])
}

velocities = {
    "NO2": init_velocities(initial_counts[0]),
    "NO": init_velocities(initial_counts[1]),
    "O": init_velocities(initial_counts[2]),
    "O3": init_velocities(initial_counts[3]),
    "VOC": init_velocities(initial_counts[4])
}

# Set up figure with two subplots
fig = plt.figure(figsize=(8, 10))

# Top subplot: particles
ax_particles = fig.add_subplot(2,1,1)
ax_particles.set_xlim(0, box_size)
ax_particles.set_ylim(0, box_size)
ax_particles.set_xlabel('X position')
ax_particles.set_ylabel('Y position')
ax_particles.set_title("Particle Simulation (Spatial)")

scatter_NO2 = ax_particles.scatter(positions["NO2"][:,0], positions["NO2"][:,1], c='red', s=10, label='NO2')
scatter_NO = ax_particles.scatter(positions["NO"][:,0] if len(positions["NO"])>0 else [],
                                  positions["NO"][:,1] if len(positions["NO"])>0 else [],
                                  c='blue', s=10, label='NO')
scatter_O = ax_particles.scatter(positions["O"][:,0] if len(positions["O"])>0 else [],
                                 positions["O"][:,1] if len(positions["O"])>0 else [],
                                 c='green', s=10, label='O')
scatter_O3 = ax_particles.scatter(positions["O3"][:,0] if len(positions["O3"])>0 else [],
                                  positions["O3"][:,1] if len(positions["O3"])>0 else [],
                                  c='purple', s=10, label='O3')
scatter_VOC = ax_particles.scatter(positions["VOC"][:,0] if len(positions["VOC"])>0 else [],
                                   positions["VOC"][:,1] if len(positions["VOC"])>0 else [],
                                   c='orange', s=10, label='VOC')
ax_particles.legend(loc='upper right')

count_text = ax_particles.text(0.05, 0.95, '', transform=ax_particles.transAxes, fontsize=12, verticalalignment='top')

# Bottom subplot: concentrations over time
ax_conc = fig.add_subplot(2,1,2)
ax_conc.set_xlabel('Time (s)')
ax_conc.set_ylabel('Concentration (arb. units)')
ax_conc.set_title("Concentration vs Time")

line_NO2, = ax_conc.plot([], [], 'r-', label='NO2')
line_NO,  = ax_conc.plot([], [], 'b-', label='NO')
line_O,   = ax_conc.plot([], [], 'g-', label='O')
line_O3,  = ax_conc.plot([], [], 'm-', label='O3')
line_VOC, = ax_conc.plot([], [], 'y-', label='VOC')
ax_conc.legend(loc='upper right')
ax_conc.set_ylim(0, max(NO2_0, VOC_0)*1.1)

current_state = np.array(y0)

def update_particles(species, new_count):
    old_count = len(positions[species])
    if new_count > old_count:
        # Add more particles
        add_count = new_count - old_count
        new_positions = init_positions(add_count)
        new_vels = init_velocities(add_count)
        positions[species] = np.vstack([positions[species], new_positions]) if old_count>0 else new_positions
        velocities[species] = np.vstack([velocities[species], new_vels]) if old_count>0 else new_vels
    elif new_count < old_count:
        # Remove some particles
        remove_count = old_count - new_count
        if remove_count < old_count:
            positions[species] = positions[species][:-remove_count]
            velocities[species] = velocities[species][:-remove_count]
        else:
            positions[species] = np.empty((0,2))
            velocities[species] = np.empty((0,2))

def update_positions():
    # Move particles and apply boundary conditions
    for species in positions.keys():
        if len(positions[species]) > 0:
            positions[species] += velocities[species]
            # Reflective boundaries
            mask_x_low = positions[species][:,0] < 0
            mask_x_high = positions[species][:,0] > box_size
            mask_y_low = positions[species][:,1] < 0
            mask_y_high = positions[species][:,1] > box_size

            positions[species][mask_x_low,0] = -positions[species][mask_x_low,0]
            velocities[species][mask_x_low,0] = -velocities[species][mask_x_low,0]

            positions[species][mask_x_high,0] = 2*box_size - positions[species][mask_x_high,0]
            velocities[species][mask_x_high,0] = -velocities[species][mask_x_high,0]

            positions[species][mask_y_low,1] = -positions[species][mask_y_low,1]
            velocities[species][mask_y_low,1] = -velocities[species][mask_y_low,1]

            positions[species][mask_y_high,1] = 2*box_size - positions[species][mask_y_high,1]
            velocities[species][mask_y_high,1] = -velocities[species][mask_y_high,1]

def update(frame):
    global current_state, time_passed

    # Integrate ODEs for dt
    t_span = [time_passed, time_passed+dt]
    sol = odeint(reactions, current_state, t_span, args=(k_ph, k_o3, k_voc, O2_conc))
    new_state = sol[-1]

    # Update current_state
    current_state = new_state
    time_passed += dt

    NO2_c, NO_c, O_c, O3_c, VOC_c = current_state
    # Append to data arrays
    time_data.append(time_passed)
    NO2_data.append(NO2_c)
    NO_data.append(NO_c)
    O_data.append(O_c)
    O3_data.append(O3_c)
    VOC_data.append(VOC_c)

    # Convert concentrations to particle counts
    NO2_count, NO_count, O_count, O3_count, VOC_count = concentrations_to_particle_counts(NO2_c, NO_c, O_c, O3_c, VOC_c)

    # Update particles to reflect new counts
    update_particles("NO2", NO2_count)
    update_particles("NO", NO_count)
    update_particles("O", O_count)
    update_particles("O3", O3_count)
    update_particles("VOC", VOC_count)

    # Update particle positions
    update_positions()

    # Update scatter plots
    scatter_NO2.set_offsets(positions["NO2"])
    scatter_NO.set_offsets(positions["NO"])
    scatter_O.set_offsets(positions["O"])
    scatter_O3.set_offsets(positions["O3"])
    scatter_VOC.set_offsets(positions["VOC"])

    # Update concentration lines
    line_NO2.set_data(time_data, NO2_data)
    line_NO.set_data(time_data, NO_data)
    line_O.set_data(time_data, O_data)
    line_O3.set_data(time_data, O3_data)
    line_VOC.set_data(time_data, VOC_data)
    ax_conc.set_xlim(0, time_passed+10)

    # Update count text to show what's happening
    # e.g., "Photolysis reducing NO2, forming NO and O. O then forms O3."
    reaction_description = ("UV light breaks down NO2 into NO and O.\n"
                            "O combines with O2 to form O3.\n"
                            "VOC presence also enhances O3 formation.\n"
                            f"Current Counts: NO2={NO2_count}, NO={NO_count}, O={O_count}, O3={O3_count}, VOC={VOC_count}")
    count_text.set_text(reaction_description)

    return (scatter_NO2, scatter_NO, scatter_O, scatter_O3, scatter_VOC,
            line_NO2, line_NO, line_O, line_O3, line_VOC, count_text)

anim = FuncAnimation(fig, update, frames=200, interval=200, blit=False)
plt.tight_layout()
plt.show()
