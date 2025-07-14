import tkinter as tk
import numpy as np
import random

from numba import jit, prange

from PIL import Image, ImageTk
import colorsys

scale = 8 # scaling factor for display
update_delay = 5 # milliseconds between updates

# parameters
L = 50 # lattice size (LxL)
T = 0.866 # temperature
J = 1 # coupling constant
Acceptance = 0 # initialize acceptance counter

# initialize spins randomly
spins = 2*np.pi*np.random.rand(L, L)

# define a sweep function
@jit(nopython=True)
def sweep(spins,T,J,Acceptance):
    # Metropolis single spin flip algorithm. We first pick a random site (x,y) and then calculate the change 
    # in energy if we were to flip it (up->down or down->up). We then draw a number to see if the move is accepted.
    # If it is, then we update the value in the lattice and update the energy, magnetization, and acceptances. 
    flipped_sites = []
    L = spins.shape[0]
    for j in prange(L**2):
        x=np.random.randint(L) #get a random position to update in the lattice
        y=np.random.randint(L)

        dE = 2*J*spins[x,y]*(spins[(x-1)%L,y]+spins[(x+1)%L,y]+spins[x,(y-1)%L]+spins[x,(y+1)%L])

        if np.random.random() < np.exp(-dE/T):# Incrementing the energy and magnetization if the move is accepted
            spins[x,y]*=-1 # update the value in the lattice
            Acceptance += 1 # increment acceptance counter
            flipped_sites.append((x,y))

    return spins,Acceptance,flipped_sites

@jit(nopython=True)
def sweep(spins,T,J,Acceptance):
    L = spins.shape[0]
    flipped_sites = []
    for i in prange(L**2):
        phi=2*np.pi*np.random.rand() #phi is our new random angle between 0,2pi
        x = np.random.randint(L)
        y = np.random.randint(L)

        dE = -J*(np.cos(phi-spins[x-1,y])+
                    np.cos(phi-spins[(x+1)%L,y])+
                    np.cos(phi-spins[x,y-1])+
                    np.cos(phi-spins[x,(y+1)%L])-(np.cos(spins[x,y]-spins[x-1,y])+
                    np.cos(spins[x,y]-spins[(x+1)%L,y])+
                    np.cos(spins[x,y]-spins[x,y-1])+
                    np.cos(spins[x,y]-spins[x,(y+1)%L])))

        # Applying the Metropolis criteria
        if np.random.random() < np.exp(-dE/T):
            spins[x,y]=phi # update the value in the lattice to the new angle
            Acceptance += 1 # increment acceptance counter
            flipped_sites.append((x,y)) # we don't need this for continuous spins

    return spins, Acceptance, flipped_sites

def spins_to_image(spins):
    L = spins.shape[0]
    rgb_array = np.zeros((L, L, 3), dtype=np.uint8)

    for i in range(L):
        for j in range(L):
            angle = spins[i, j] % (2 * np.pi)
            hue = angle / (2 * np.pi)  # Normalize to [0, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            rgb_array[i, j] = [int(255 * r), int(255 * g), int(255 * b)]

    img = Image.fromarray(rgb_array, mode='RGB')
    img = img.resize((L * scale, L * scale), resample=Image.NEAREST)
    return img

def run_simulation():
    global spins, T, J, Acceptance, label_img, label
    spins, Acceptance, flipped_sites = sweep(spins, T, J, Acceptance)

    pil_img = spins_to_image(spins)
    label_img = ImageTk.PhotoImage(pil_img)
    label.configure(image=label_img)

    root.after(5, run_simulation)

root = tk.Tk()
root.title("XY Model GUI")

img = tk.PhotoImage(width=L, height=L)
label = tk.Label(root, image=img)
label.pack()

temp_slider = tk.Scale(root, from_=0.1, to=5.0, resolution=0.01, orient=tk.HORIZONTAL, label="Temperature T")
temp_slider.set(T)
temp_slider.pack()

def update_temp(val):
    global T
    T = float(val)
temp_slider.config(command=update_temp)

coupling_slider = tk.Scale(root, from_=-2.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, label="Coupling J")
coupling_slider.set(J)
coupling_slider.pack()

def update_coupling(val):
    global J
    J = float(val)
coupling_slider.config(command=update_coupling)

root.after(5, run_simulation)
root.mainloop()