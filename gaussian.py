# ***************************************
# *										*
# *			GAUSSIAN DISTRIBUTION		*
# *										*
# ***************************************

import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from functions import *

# Number of components and radii for spheres
N_part = 200
N_obstacles = 530

# Box
len_box = 40
left_border = - len_box/2
right_border = len_box/2
floor = 0
start_y = 10 + floor

# Particles
particles = np.zeros( (N_part, 3, 2) )		# 1st dim: particle; 2nd dim: position, velocity or acceleration; 3rd dimension: 2D spatial coord 
R_part = 0.1
# Positions (Particles initially in x = 0 and y = start_y )
particles[:, 0, 0] = 0
particles[:, 0, 1] = start_y
# Velocities already set to zero
# Acceleration (gravity along y)
particles[ : , 2, 1] = - 9.81

# Obstacles
rows_obst = 10
num_obst_per_row = N_obstacles / rows_obst
R_obst = 0.2
spacing = (start_y - floor - 2) / (rows_obst - 1)
obstacles = np.zeros( (rows_obst, int(num_obst_per_row), 2) )
# For all the obstacles, the y depends on their row
for i in range(rows_obst) :
	obstacles[i, :, 1] =  start_y - 1 - i * spacing
# The obstacles position is alternated every row
for i in range(rows_obst) :
	if i % 2 == 0 :
		obstacles[i, :, 0] = np.linspace(left_border + 3*R_obst + 0.1 + len_box/num_obst_per_row/2, right_border - 3*R_obst - 0.1, num=int(num_obst_per_row))
	else :
		obstacles[i, :, 0] = np.linspace(left_border + 3*R_obst + 0.1, right_border - 3*R_obst - 0.1 - len_box / num_obst_per_row/2, num=int(num_obst_per_row))

# Simulation
dt = 0.003
t_sample = 15
traj = []
arrived = np.zeros(N_part)			# 0 : not arrived / 1 : arrived
num_arrived = arrived.sum()			# Number of particles arrived at floor
actual_arrived = 0					# Actual number of particles used in the progress bar

print("\nStarting calculation...\n")

with open("data.xyz", "w") as data_file:

	s = 0
	while( num_arrived < N_part - 0.5) :
		
		# Evolve with velocity-Verlet
		particles = velocity_verlet(particles, dt)
		
		# Check collisions with obstacles
		particles = check_collision(particles, obstacles, R_part, R_obst, arrived)

		# Save trajectory every t_sample
		if s % t_sample == 0 :
			traj.append( particles[:, 0, :].copy() )
		
		# Write on .xyz file
		if s % int(t_sample/3) == 0 :
			# Print number of particles
			data_file.write(f"{N_part + obstacles.shape[0]*obstacles.shape[1]}\n")
			data_file.write(f"Frame {int(s/10)}\n")

			# Data
			for i in range(particles.shape[0]) :
				data_file.write(f"P\t{particles[i][0][0]*10:.3f}\t{particles[i][0][1]*10:.3f}\t0\n")
			
			for i in range(obstacles.shape[0]) :
				for j in range(obstacles.shape[1]) :
					data_file.write(f"O\t{obstacles[i][j][0]*10:.3f}\t{obstacles[i][j][1]*10:.3f}\t0\n")
		
		# Check for arrived particles and set their acceleration and velocity to zero
		arrived, particles = check_finish(particles, floor, arrived)
		num_arrived = arrived.sum()

		# Progress bar
		if actual_arrived != int(num_arrived) or s == 0 :
			prog_percentage = np.floor(100 * num_arrived / N_part).astype(int)
			sys.stdout.write("\r" + "Status\t" + "[" + "#" * prog_percentage + "." * (100 - prog_percentage) + "]\t" + f"{prog_percentage}%" + f"  ({actual_arrived} / {N_part})" )
			actual_arrived = int(num_arrived)
		
		s += 1

prog_percentage = int(100 * num_arrived / N_part)
sys.stdout.write("\r" + "Status\t" + "[" + "#" * prog_percentage + "." * (100 - prog_percentage) + "]\t" + f"{prog_percentage}%" + f"  ({actual_arrived} / {N_part})\n" )
		
obstacles = obstacles.reshape((obstacles.shape[0]*obstacles.shape[1], obstacles.shape[2]))

# Plot
plt.ion()
fig, ax = plt.subplots(figsize=(20,5))
ax.set_xlim((left_border, right_border))
ax.set_ylim((floor, start_y + 1 + 5 * R_part))
colors = [ ( "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) ) for i in range(N_part) ]

# Draw obstacles
ax.scatter(obstacles[:,0], obstacles[:,1], color="red", s= (R_obst * 45)**2 * np.ones(obstacles.shape[0]))
# Draw circles
curr = ax.scatter(traj[0][:,0], traj[0][:,1], c=colors, s= (R_part * 45)**2 * np.ones(particles.shape[0]))

# Redraw only the circles for the various frames
for k in range(1, len(traj)) :

	curr.remove()
	curr=ax.scatter(traj[k][:,0], traj[k][:,1], c=colors, s= (R_part * 45)**2 * np.ones(particles.shape[0]))

	fig.canvas.draw()
	fig.canvas.flush_events()

plt.ioff()
plt.close(fig)


# Calculating histogram
print("\nCalculating histogram...\n")

fig,ax = plt.subplots(figsize=(10,10))
ax.set_xlabel("x")
ax.set_ylabel("Counts")
ax.hist(particles[:,0,0], color = "red", ec="orange", lw=2 )
ax.set_title(f"Arrival position distribution (N = {N_part})", fontsize=20)
plt.show()

print("Finished!\n")


#########################################################################################################################################################

# SLOWER ALTERNATIVE FOR THE PLOT, BUT ALLOWS EASIER TUNING OF THE POINT SIZE IN TERMS OF PLOT UNITS

# # Draw obstacles
# for i in range(obstacles.shape[0]) :
# 	circ = plt.Circle( (obstacles[i][0], obstacles[i][1]), R_obst, color="red" )
# 	ax.add_artist(circ)

# actual_circles = []

# # Draw circles
# for i in range(N_part) :
# 	circ = plt.Circle( (traj[0][i][0], traj[0][i][1]), R_part, color = colors[i])
# 	actual_circles.append(circ)
# 	ax.add_artist(circ)

# # Redraw only the circles for the various frames
# for k in range(1, len(traj)) :

# 	for j in range(len(actual_circles)) :
# 		(actual_circles[j]).remove()

# 	for i in range(N_part) :
# 		circ = plt.Circle( (traj[k][i][0], traj[k][i][1]), R_part, color = colors[i])
# 		actual_circles[i] = circ
# 		ax.add_artist(circ)

# 	fig.canvas.draw()
# 	fig.canvas.flush_events()

# plt.ioff()
# plt.close(fig)