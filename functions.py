import numpy as np
from numba import njit, prange

# Functions

@njit
def velocity_verlet(particles, dt) :

    v_dt_2 = np.zeros( (particles.shape[0], particles.shape[2]) )

    v_dt_2 = particles[:, 1, :] + 0.5 * particles[:, 2, :] * dt             # Update velocities pt.1
    particles[:, 0, :] = particles[:, 0, :] + v_dt_2 * dt                   # Update position
    particles[:, 1, :] = v_dt_2 + 0.5 * particles[:, 2, :] * dt             # Update velocities pt.2
    
    return particles



@njit(parallel = True)
def check_collision(particles, obstacles, R_part, R_obst, arrived) :

    inelastic_coeff = 0.95      # Coefficient to represent the fraction of conserved modulus of the velocity in a collision

    for i in prange(particles.shape[0]) :
        if arrived[i] < 0.5 :
            for j in range(obstacles.shape[0]) :
                for k in range(obstacles.shape[1]) :

                    dr = particles[i,0,:] - obstacles[j,k,:]
                    dist = np.sqrt( dr.dot(dr) )

                    if dist < (R_part + R_obst) :
                        versor = dr / dist

                        # New velocity picked randomly in a window +/- pi/2 around the collision direction
                        theta_scatter = ( np.random.rand() - 0.5 ) * np.pi
                        new_velocity_dir = np.zeros(2)
                        new_velocity_dir[0] = np.cos(theta_scatter) * versor[0] + np.sin(theta_scatter) * versor[1]
                        new_velocity_dir[1] = - np.sin(theta_scatter) * versor[0] + np.cos(theta_scatter) * versor[1]
                        mod_v = np.sqrt( particles[i,1,:].dot(particles[i,1,:]) )
                        particles[i,1,:] = new_velocity_dir * mod_v * inelastic_coeff

                        # The new position is moved on the surface, to avoid multiple clashes due to inelasticity
                        particles[i,0,:] = particles[i,0,:] + versor * (R_part + R_obst - dist)

    return particles



@njit
def check_finish(particles, floor, arrived) :

    for i in range(particles.shape[0]) :
        if arrived[i] < 0.5 and particles[i,0,1] < floor : 
            arrived[i] = 1
            particles[i, 1, :] = np.zeros(2)
            particles[i, 2, :] = np.zeros(2)

    
    return arrived, particles