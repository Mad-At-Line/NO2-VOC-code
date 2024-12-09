import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator

# VOC: Volatile Organic Compounds
# They enhance ozone formation when reacting with NOx under UV.

# ---------------------------------
# KINETIC MODEL SETUP
# ---------------------------------
O2_conc = 0.21
k_ph_base = 1e-3   # Base photolysis rate (to be scaled by UV intensity)
k_o3 = 1e6
k_voc = 5e5

NO2_0 = 1e-6
NO_0 = 0.0
O_0 = 0.0
O3_0 = 0.0
VOC_0 = 1e-6
y0 = [NO2_0, NO_0, O_0, O3_0, VOC_0]

time_passed = 0.0
dt = 5.0

time_data = []
NO2_data = []
NO_data = []
O_data = []
O3_data = []
VOC_data = []

# Box Setup
box_size = 10
particle_scale = 5e7

# Create UV Intensity Map
res = 50
X_arr = np.linspace(0, box_size, res)
Y_arr = np.linspace(0, box_size, res)
X, Y = np.meshgrid(X_arr, Y_arr)
UV_intensity = np.exp(-((X - box_size/2)**2 + (Y - box_size/2)**2)/10)
uv_intensity_max = UV_intensity.max()

# Interpolator for UV intensity
UV_interpolator = RegularGridInterpolator((X_arr, Y_arr), UV_intensity.T)
# Note: RegularGridInterpolator expects data in order (x,y) and
# by default x corresponds to the first dimension of the data, and y the second.
# Our UV_intensity is shaped (res, res) with X, Y from meshgrid.
# X is increasing along X_arr (horizontal), Y along Y_arr (vertical).
# By using UV_intensity.T, we match (X_arr, Y_arr) ordering.

def get_uv_intensity_for_positions(positions):
    # positions is Nx3. We only use x,y since UV map is 2D.
    if len(positions) == 0:
        return 0.0
    pts = positions[:, :2]  # (x,y)
    # Ensure points are within bounds [0, box_size], clamp if needed
    pts = np.clip(pts, 0, box_size)
    return UV_interpolator(pts)

def reactions(y, t, k_ph_current, k_o3, k_voc, O2):
    NO2, NO, O, O3, VOC = y
    r_ph = k_ph_current * NO2
    r_o3_form = k_o3 * O * O2
    r_voc_rxn = k_voc * VOC * O

    dNO2_dt = -r_ph
    dNO_dt = r_ph
    dO_dt = r_ph - r_o3_form - r_voc_rxn
    dO3_dt = r_o3_form + r_voc_rxn
    dVOC_dt = -r_voc_rxn
    return [dNO2_dt, dNO_dt, dO_dt, dO3_dt, dVOC_dt]


def concentrations_to_particle_counts(NO2_c, NO_c, O_c, O3_c, VOC_c):
    NO2_count = max(int(NO2_c * particle_scale), 0)
    NO_count  = max(int(NO_c * particle_scale), 0)
    O_count   = max(int(O_c * particle_scale), 0)
    O3_count  = max(int(O3_c * particle_scale), 0)
    VOC_count = max(int(VOC_c * particle_scale), 0)
    return NO2_count, NO_count, O_count, O3_count, VOC_count

def init_positions(count):
    if count <= 0:
        return np.empty((0,3))
    return np.random.rand(count, 3) * box_size

def init_velocities(count):
    if count <= 0:
        return np.empty((0,3))
    return (np.random.rand(count, 3) - 0.5) * 0.05

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

fig = plt.figure(figsize=(8,10))
ax_particles = fig.add_subplot(2,1,1, projection='3d')
ax_particles.set_xlim(0, box_size)
ax_particles.set_ylim(0, box_size)
ax_particles.set_zlim(0, box_size)
ax_particles.set_xlabel('X position')
ax_particles.set_ylabel('Y position')
ax_particles.set_zlabel('Z position')
ax_particles.set_title("3D Particle Simulation with UV Field")

# Plot UV surface at z=0
surf = ax_particles.plot_surface(
    X, Y, np.zeros_like(X),
    facecolors=plt.cm.viridis((UV_intensity - UV_intensity.min())/(UV_intensity.max()-UV_intensity.min())),
    rstride=1, cstride=1, shade=False
)
surf.set_edgecolor('none')

scatter_NO2 = ax_particles.scatter(positions["NO2"][:,0], positions["NO2"][:,1], positions["NO2"][:,2], c='red', s=10, label='NO2')
scatter_NO = ax_particles.scatter([], [], [], c='blue', s=10, label='NO')
scatter_O = ax_particles.scatter([], [], [], c='green', s=10, label='O')
scatter_O3 = ax_particles.scatter([], [], [], c='purple', s=10, label='O3')
scatter_VOC = ax_particles.scatter([], [], [], c='orange', s=10, label='VOC')
ax_particles.legend(loc='upper right')

# Place reaction text at top-left, counts slightly lower
reaction_text = ax_particles.text2D(0.05, 0.95, '', transform=ax_particles.transAxes, fontsize=10, verticalalignment='top')
count_text = ax_particles.text2D(0.05, 0.88, '', transform=ax_particles.transAxes, fontsize=10, verticalalignment='top')

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
        add_count = new_count - old_count
        new_positions = init_positions(add_count)
        new_vels = init_velocities(add_count)
        if old_count > 0:
            positions[species] = np.vstack([positions[species], new_positions])
            velocities[species] = np.vstack([velocities[species], new_vels])
        else:
            positions[species] = new_positions
            velocities[species] = new_vels
    elif new_count < old_count:
        remove_count = old_count - new_count
        if remove_count < old_count:
            positions[species] = positions[species][:-remove_count]
            velocities[species] = velocities[species][:-remove_count]
        else:
            positions[species] = np.empty((0,3))
            velocities[species] = np.empty((0,3))

def update_positions():
    for species in positions.keys():
        if len(positions[species]) > 0:
            positions[species] += velocities[species]
            for dim in [0,1,2]:
                mask_low = positions[species][:,dim] < 0
                mask_high = positions[species][:,dim] > box_size

                positions[species][mask_low,dim] = -positions[species][mask_low,dim]
                velocities[species][mask_low,dim] = -velocities[species][mask_low,dim]

                positions[species][mask_high,dim] = 2*box_size - positions[species][mask_high,dim]
                velocities[species][mask_high,dim] = -velocities[species][mask_high,dim]

def update(frame):
    global current_state, time_passed

    # Determine current UV influence on k_ph
    # Approximate: use NO2 particle locations to find mean UV intensity
    if len(positions["NO2"]) > 0:
        uv_values = get_uv_intensity_for_positions(positions["NO2"])
        mean_uv = np.mean(uv_values)
    else:
        # If no NO2, just default to some baseline UV (e.g., center intensity)
        mean_uv = uv_intensity_max * 0.5

    # Scale k_ph by mean_uv relative to maximum
    k_ph_current = k_ph_base * (mean_uv / uv_intensity_max)

    # Integrate ODEs
    t_span = [time_passed, time_passed+dt]
    sol = odeint(reactions, current_state, t_span, args=(k_ph_current, k_o3, k_voc, O2_conc))
    new_state = sol[-1]

    current_state = new_state
    time_passed += dt

    NO2_c, NO_c, O_c, O3_c, VOC_c = current_state
    time_data.append(time_passed)
    NO2_data.append(NO2_c)
    NO_data.append(NO_c)
    O_data.append(O_c)
    O3_data.append(O3_c)
    VOC_data.append(VOC_c)

    NO2_count, NO_count, O_count, O3_count, VOC_count = concentrations_to_particle_counts(NO2_c, NO_c, O_c, O3_c, VOC_c)

    update_particles("NO2", NO2_count)
    update_particles("NO", NO_count)
    update_particles("O", O_count)
    update_particles("O3", O3_count)
    update_particles("VOC", VOC_count)

    update_positions()

    # Update 3D scatter data
    scatter_NO2._offsets3d = (positions["NO2"][:,0], positions["NO2"][:,1], positions["NO2"][:,2]) if len(positions["NO2"])>0 else ([],[],[])
    scatter_NO._offsets3d = (positions["NO"][:,0], positions["NO"][:,1], positions["NO"][:,2]) if len(positions["NO"])>0 else ([],[],[])
    scatter_O._offsets3d = (positions["O"][:,0], positions["O"][:,1], positions["O"][:,2]) if len(positions["O"])>0 else ([],[],[])
    scatter_O3._offsets3d = (positions["O3"][:,0], positions["O3"][:,1], positions["O3"][:,2]) if len(positions["O3"])>0 else ([],[],[])
    scatter_VOC._offsets3d = (positions["VOC"][:,0], positions["VOC"][:,1], positions["VOC"][:,2]) if len(positions["VOC"])>0 else ([],[],[])

    # Update concentration lines
    line_NO2.set_data(time_data, NO2_data)
    line_NO.set_data(time_data, NO_data)
    line_O.set_data(time_data, O_data)
    line_O3.set_data(time_data, O3_data)
    line_VOC.set_data(time_data, VOC_data)
    ax_conc.set_xlim(0, time_passed+10)

    # Update text
    # Moved counts lower to avoid overlap with reaction text
    reaction_description = (
        "UV-driven reactions:\n"
        "NO2 + hv -> NO + O\n"
        "O + O2 -> O3\n"
        "VOC + O -> O3"
    )
    reaction_text.set_text(reaction_description)

    count_info = (
        f"Counts: NO2={NO2_count}, NO={NO_count}, O={O_count}, O3={O3_count}, VOC={VOC_count}\n"
        f"UV scaling: mean_uv={mean_uv:.3f}, k_ph={k_ph_current:.2e}"
    )
    count_text.set_text(count_info)

    return (scatter_NO2, scatter_NO, scatter_O, scatter_O3, scatter_VOC,
            line_NO2, line_NO, line_O, line_O3, line_VOC, reaction_text, count_text)

# Increase FPS by reducing interval
# For ~20 FPS: interval=50 ms
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.tight_layout()
plt.show()
