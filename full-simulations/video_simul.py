import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_particles = {
    "NO2": 50,
    "NO": 0,
    "O": 0,
    "O3": 0,
    "VOC": 50
}

# Particle properties: positions and velocities
# For simplicity, we'll just randomly scatter them
box_size = 10  # Arbitrary box dimension


def init_positions(count):
    return np.random.rand(count, 2) * box_size


positions = {
    "NO2": init_positions(num_particles["NO2"]),
    "NO": init_positions(num_particles["NO"]),
    "O": init_positions(num_particles["O"]),
    "O3": init_positions(num_particles["O3"]),
    "VOC": init_positions(num_particles["VOC"])
}


# Assign random velocities to particles
def init_velocities(count):
    # Random velocities in both x and y directions
    return (np.random.rand(count, 2) - 0.5) * 0.1


velocities = {
    "NO2": init_velocities(num_particles["NO2"]),
    "NO": init_velocities(num_particles["NO"]),
    "O": init_velocities(num_particles["O"]),
    "O3": init_velocities(num_particles["O3"]),
    "VOC": init_velocities(num_particles["VOC"])
}

# Plot Setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_title("Particle Simulation")

# Scatter plots for each species
scatters = {
    "NO2": ax.scatter(positions["NO2"][:, 0], positions["NO2"][:, 1], c='red', label='NO2'),
    "NO": ax.scatter([], [], c='blue', label='NO'),
    "O": ax.scatter([], [], c='green', label='O'),
    "O3": ax.scatter([], [], c='purple', label='O3'),
    "VOC": ax.scatter(positions["VOC"][:, 0], positions["VOC"][:, 1], c='orange', label='VOC')
}
ax.legend()

# Placeholder reaction: Over time, NO2 gets photolyzed, creating NO and O,
# then O + O2 (assumed abundant) forms O3. VOC affects O3 formation rate.
# In a real simulation, you'll integrate actual reaction kinetics.
time_step = 0
max_time_steps = 200


def update(frame):
    global time_step
    time_step += 1

    # Example reaction logic (highly simplified and not chemically accurate):
    # Every 10 steps, convert some NO2 to NO and O
    if time_step % 10 == 0 and len(positions["NO2"]) > 5:
        # Convert 5 NO2 to NO and O
        idx_to_convert = np.random.choice(len(positions["NO2"]), 5, replace=False)

        # NO2 particles chosen for conversion
        chosen_NO2 = positions["NO2"][idx_to_convert]

        # Remove them from NO2
        positions["NO2"] = np.delete(positions["NO2"], idx_to_convert, axis=0)
        velocities["NO2"] = np.delete(velocities["NO2"], idx_to_convert, axis=0)

        # Add equivalent NO and O particles
        positions["NO"] = np.vstack([positions["NO"], chosen_NO2])
        velocities["NO"] = np.vstack([velocities["NO"], init_velocities(5)])

        positions["O"] = np.vstack([positions["O"], chosen_NO2])
        velocities["O"] = np.vstack([velocities["O"], init_velocities(5)])

    # Every 10 steps, some O converts to O3 in presence of VOC (just as a placeholder)
    # This reduces O and increases O3 counts.
    if time_step % 15 == 0 and len(positions["O"]) > 5 and len(positions["VOC"]) > 0:
        idx_to_convert = np.random.choice(len(positions["O"]), 5, replace=False)
        chosen_O = positions["O"][idx_to_convert]

        # Remove O
        positions["O"] = np.delete(positions["O"], idx_to_convert, axis=0)
        velocities["O"] = np.delete(velocities["O"], idx_to_convert, axis=0)

        # Add O3
        positions["O3"] = np.vstack([positions["O3"], chosen_O])
        velocities["O3"] = np.vstack([velocities["O3"], init_velocities(5)])

        # Maybe also reduce VOC slightly over time to simulate consumption
        if len(positions["VOC"]) > 5:
            positions["VOC"] = positions["VOC"][:-5]
            velocities["VOC"] = velocities["VOC"][:-5]

    # Update positions based on velocities
    for species in positions.keys():
        if len(positions[species]) > 0:
            positions[species] += velocities[species]
            # Reflective boundary conditions
            for i in range(len(positions[species])):
                for dim in [0, 1]:
                    if positions[species][i, dim] < 0:
                        positions[species][i, dim] = -positions[species][i, dim]
                        velocities[species][i, dim] = -velocities[species][i, dim]
                    elif positions[species][i, dim] > box_size:
                        positions[species][i, dim] = 2 * box_size - positions[species][i, dim]
                        velocities[species][i, dim] = -velocities[species][i, dim]

    # Update scatter plots
    scatters["NO2"].set_offsets(positions["NO2"] if len(positions["NO2"]) > 0 else np.empty((0, 2)))
    scatters["NO"].set_offsets(positions["NO"] if len(positions["NO"]) > 0 else np.empty((0, 2)))
    scatters["O"].set_offsets(positions["O"] if len(positions["O"]) > 0 else np.empty((0, 2)))
    scatters["O3"].set_offsets(positions["O3"] if len(positions["O3"]) > 0 else np.empty((0, 2)))
    scatters["VOC"].set_offsets(positions["VOC"] if len(positions["VOC"]) > 0 else np.empty((0, 2)))

    ax.set_title(f"Particle Simulation - Time Step: {time_step}")

    if time_step > max_time_steps:
        anim.event_source.stop()  # stop the animation after a certain time

    return scatters.values()


anim = FuncAnimation(fig, update, frames=range(max_time_steps), interval=100, blit=False)
plt.show()
