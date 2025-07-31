# XY Model GUI

A simple XY Model app written in Python using PIL, Tkinter, Numba, and Numpy. Arrays and math are handled with Numpy, which is then provided a significant speedup by Numba. Tkinter creates a popup window in which the simulation runs in with sliders and buttons, and PIL creates the images displayed of the simulation. Based on my other project [Ising_GUI](https://github.com/swestastic/Ising_GUI/)

![demo](images/demo.gif)

## Background

The classical XY model is a natural extension of the Ising model. The two models
 share many similarities, but the XY model has an extra degree of freedom allowing spins
 to rotate in the XY plane, whereas the Ising model only allows for spins to hold a discrete
 ±1 value. Although the difference seems small, from this change emerges the
 Berezinskii-Kosterlitz-Thousless (BKT) topical phase transition, where pairs of vorticies
 and antivorticies appear near the critical temperature. This model can be a useful tool for
 studying the BKT phase transition, which can be seen experimentally in some thin-film
 materials [1] and some trapped gas systems [2].

The classical XY model is described by the following Hamiltonian:

```math
H = -J \sum_{\langle i,j \rangle}s_i \cdot s_j - h\cdot\sum_i s_i
```

This Hamiltonian can instead be expressed in terms of angles as follows:

```math
H = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j)- h\cdot\sum_i s_i
```

Where $J$ is the interaction strength between neighboring sites, $s_i=[0,2\pi)$ is the value at site $i$, and $h$ is an external magnetic field applied along $\hat{x}$.

In two dimensions with no external magnetic field ($h=0$), the model exhibits a phase transition at $T_c \approx (0.8816)\frac{J}{k}$ where $k$ is the Boltzmann constant, which is commonly set to $k=1$. For $J>0$, the model is ferromagnetic, and for $J<0$, it is anti-ferromagnetic.

This simulation using the Metropolis-Hastings algorithm, where "updates" are proposed to random sites on the lattice. An update will change the value on a given site to a new random value on $[0,2\pi)$.
An update will either be accepted or rejected based on a Boltzmann probability, $r<e^{-\Delta E/T}$, where $r$ is a random number drawn on $(0,1)$. Decreases in energy are always accepted, and increases in energy have a chance to be accepted.

```math
  \Delta E = -J (\sum_{\text{neighbors of } i}\cos(\phi - \theta_\text{neighbor}) - \cos(\theta_{i} - \theta_\text{neighbor}))
```

## Usage

Simply run `python3 XY_GUI.py` to open a Tkinter window and run the simulation. The slider bars for $T$ and $J$ are intuitive to use, and the simulation will update automatically in correspondence with them.

## Future Work

- Improve the efficiency of the image drawing by storing a persistent version of `rgb_array` and using `flipped_sites` to only update necessary sites, rather than redrawing the entire image from scratch every time

- Add external magnetic field functionality and slider

- Add the ability to save data from simulation runs

- Possibly add a quiver mode? Where it would show arrows rather than colors. This may be more resource intensive though

## References

1. Mintu Mondal, Sanjeev Kumar, Madhavi Chand, Anand Kamlapure, Garima
 Saraswat, G. Seibold, L. Benfatto, and Pratap Raychaudhuri. Role of the vortex-core
 energy on the berezinskii-kosterlitz-thouless transition in thin films of nbn. Physical
 Review Letters, 107(21), November 2011.

2. Zoran Hadzibabic, Peter Kr¨ uger, Marc Cheneau, Baptiste Battelier, and Jean
 Dalibard. Berezinskii–kosterlitz–thouless crossover in a trapped atomic gas. Nature,
 441(7097):1118–1121, June 2006.