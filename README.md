# galton_board

## Physics background

Simulation of a naive Galton board. The rules are the following:

* Each particle interacts only with the obstacles, not with the other particles (hence each particle is an independent measure)
* After an inelastic collision with an obstacle, the velocity of the particle assumes a random direction in an interval [- $\pi$/2, $\pi$/2] centered around the direction connetting the center of the obstacle with the one of the particle (uniformly distributed random variable)

Since the motion of each particle is caused by many independent random collisions, the distribution of the final position of the particles along the horizontal axis should be -approximately- gaussian (Central Limit Theorem).

## Technical details

The scripts are written in python. The only additional packages required to run the simulation are `numpy`, `numba` and `matplotlib`; if you have conda installed, the command:

```bash
conda create -n galton_board numba matplotlib
```

should create an environment with all the necessary packages.

After this, you can start the simulations with the commands:

```bash
conda activate galton_board
python gaussian.py
```

## Additional information

If you have the `VMD` (Visual Molecular Dynamics) software installed, then you can visualize the result of the simulation with it via the command:

```bash
vmd data.xyz
```

I suggest using the "VDW" choice for the "Drawing Method" option in `Graphics -> Representations`.
