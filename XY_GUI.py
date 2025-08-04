import tkinter as tk
from tkinter import ttk
import numpy as np
import random
from numba import njit, prange, float64, int32
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import colorsys
import argparse
import time

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Ising Model Simulation GUI")
parser.add_argument("--cache", type=bool, default=False, help="Enable caching for faster simulations")
parser.add_argument("--fastmath", type=bool, default=True, help="Enable fast math optimizations")
parser.add_argument("--parallel", type=bool, default=True, help="Enable parallel processing")

# parameters
L = 64 # lattice size (LxL)
T = 0.866 # temperature
J = 1.0 # coupling constant
h = 0.0 # external magnetic field

# initializations
Acceptance = 0 # initialize acceptance counter
sweepcount = 1 # number of sweeps done
sweep_counter = 0 # counter for sweeps per second
count = 0 # counter for plot updates
theta = np.pi

scale = 512 // L # scaling factor for display

# Numba settings
FASTMATH = parser.parse_args().fastmath
PARALLEL = parser.parse_args().parallel
CACHE = parser.parse_args().cache

plot_observable = "Magnetization" # "Magnetization", "Energy", or "Acceptance"
algorithm = "Metropolis" # "Metropolis", "Metropolis Limited Change", or "Wolff"

############################# Observable Calculation Functions #############################
@njit(float64(float64[:,:], float64), parallel=PARALLEL, fastmath=FASTMATH, cache=CACHE)
def Energy(spins, J):
    TotalEnergy = 0.0
    L = spins.shape[0]
    for i in prange(L):
        for j in range(L):
            TotalEnergy += (
                np.cos(spins[i, j] - spins[(i+1)%L, j]) +  # down neighbor
                np.cos(spins[i, j] - spins[i, (j+1)%L])    # right neighbor
            )
    return -J * TotalEnergy

@njit(fastmath=FASTMATH, cache=CACHE)
def Mag(spins): #magnetization function returns X,Y components and Magnitude [Mx,My,M]
  #The magnetization is given by the sum of the X and Y spin components.
  #this function returns X and Y components in an array
  Mx=np.sum(np.cos(spins))
  My=np.sum(np.sin(spins))
  M = np.sqrt(Mx**2+My**2)
  return np.array([Mx,My,M])

############################# Monte Carlo Algorithms #############################
@njit(fastmath=FASTMATH, cache=CACHE)
def Metropolis(spins, T, J, L, E, Acceptance, sweepcount):
    sweepcount += L**2
    flip_count = 0
    flipped_sites = np.zeros((L**2, 2), dtype=np.int32)  # Preallocate for flipped sites

    for i in range(L**2):
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
            flipped_sites[flip_count] = (x,y)
            E += dE
            flip_count += 1

    return spins, E, Acceptance, flipped_sites[:flip_count], sweepcount

@njit(fastmath=FASTMATH, cache=CACHE)
def Metropolis_Limited_Change(spins, T, J, L, E, Acceptance, theta, sweepcount):
    sweepcount += L**2
    flip_count = 0
    flipped_sites = np.zeros((L**2, 2), dtype=np.int32)  # Preallocate for flipped sites
    for i in prange(L**2):
        x = np.random.randint(L)
        y = np.random.randint(L)
        saved = spins[x,y] #save the value in case we revert back to it
        phi=np.random.uniform(saved-theta,saved+theta)%(2*np.pi)
                # The difference in energy comes solely from spin at (x,y) and its neighbors.
                # Here comes Eqn 3
        dE = -J*(np.cos(phi-spins[x-1,y])+
                    np.cos(phi-spins[(x+1)%L,y])+
                    np.cos(phi-spins[x,y-1])+
                    np.cos(phi-spins[x,(y+1)%L])-(np.cos(spins[x,y]-spins[x-1,y])+
                    np.cos(spins[x,y]-spins[(x+1)%L,y])+
                    np.cos(spins[x,y]-spins[x,y-1])+
                    np.cos(spins[x,y]-spins[x,(y+1)%L])))

        # Applying the Metropolis criteria
        if np.random.random() < np.exp(-dE/T):
            spins[x,y]=phi #update the value
            Acceptance += 1 #acceptances are now counted
            flipped_sites[flip_count] = (x,y)
            E += dE
    return spins,E,Acceptance,flipped_sites[:flip_count],sweepcount

@njit(fastmath=FASTMATH, cache=CACHE)
def update_theta(AcceptanceRatio,theta):
    ##Alternative method to try: Halving and doubling based on threshhold (e.g if AR>0.5, theta=theta*2, if AR<0.5, theta=theta/2)
    if AcceptanceRatio >= 0.5:
        theta *= 2
        if theta > np.pi:
            theta = np.pi # This is the maximum range, anything larger than this will be cut off anyways
    else:
        theta /= 2
    
    if theta < 0.1:
        theta = 0.1
    return theta

@njit(fastmath=FASTMATH, cache=CACHE)
def Wolff(spins,T, J, L):
    # Pick a random site to start the cluster 
    x = np.random.randint(L)
    y  = np.random.randint(L) 
    #value = array[x,y] # Save the value of the spin at that site
    
    # Generate a random angle between 0 and 2pi
    phi = np.random.rand()*2*np.pi 
    cluster = []
    cluster.append((x,y)) # add the site to the cluster
    already_flipped=[]
    backup = spins.copy()

    for i,j in cluster: # add nearest neighbors to cluster based on probability
        north = i,(j+1)%L
        south = i,(j-1)%L
        east = (i+1)%L,j
        west = (i-1)%L,j
        neighbors = [north,south,east,west]
        
        if (i,j) not in already_flipped: #maybe create a duplicate of the matrix? and reference values from that idk 
            theta_i = spins[i,j] 
            spins[i,j] = (np.pi-theta_i+2*phi)%(2*np.pi) #update the value, but the probability is based on the old value
            already_flipped.append((i,j))
        else:
            theta_i = backup[i,j]
            
        for k in neighbors:
            theta_j = spins[k]
            theta_prime = (np.pi-theta_j+2*phi)%(2*np.pi) #This is the new angle that will be flipped to
            rand = np.random.rand()
            prob = 1-np.exp(min(0,-2*J/T*np.cos(phi-theta_i)*np.cos(phi-theta_j)))
            #print(prob)
            if rand < prob and k not in cluster:
                cluster.append(k)
                if k not in already_flipped:
                    spins[k] = theta_prime #update the value
                    already_flipped.append(k)

    ClusterSize = len(cluster)

    return spins,cluster

@njit(fastmath=FASTMATH, cache=CACHE)
def overrelaxation(spins, L):
    for i in range(L):
        for j in range(L):
            # Sum of neighbor spin vectors (local field)
            hx = np.cos(spins[(i+1)%L, j]) + np.cos(spins[i-1, j]) + \
                 np.cos(spins[i, (j+1)%L]) + np.cos(spins[i, j-1])
            hy = np.sin(spins[(i+1)%L, j]) + np.sin(spins[i-1, j]) + \
                 np.sin(spins[i, (j+1)%L]) + np.sin(spins[i, j-1])
            
            theta_local = np.arctan2(hy, hx)
            # Reflect spin across local field direction
            spins[i, j] = (2 * theta_local - spins[i, j]) % (2 * np.pi)
    return spins

############################# Image Generation #############################

def init_rgb_array(spins, L):
    rgb_array = np.zeros((L, L, 3), dtype=np.uint8)
    for i in range(L):
        for j in range(L):
            angle = spins[i, j] % (2 * np.pi)
            hue = angle / (2 * np.pi)  # Normalize to [0, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            rgb_array[i, j] = [int(255 * r), int(255 * g), int(255 * b)]
    return rgb_array

def update_spins_image(spins, flipped_sites, rgb_array, scale):
    for x, y in flipped_sites:
        angle = spins[x, y] % (2 * np.pi)
        hue = angle / (2 * np.pi)  # Normalize to [0, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        rgb_array[x, y] = [int(255 * r), int(255 * g), int(255 * b)]

    # Scale using repeat
    scaled_array = np.repeat(np.repeat(rgb_array, scale, axis=0), scale, axis=1)

    return Image.fromarray(scaled_array, 'RGB')

def reset_for_parameter_change():
    global Acceptance, sweepcount, E, Mx, My, M, sweep_counter, timer
    Acceptance = 0
    sweepcount = 1
    E = Energy(spins,J)
    Mx, My, M = Mag(spins)
    timer = time.time()
    sweep_counter = 0

def update_temp(val):
    global T
    T = float(val)
    temp_entry.delete(0, tk.END)
    temp_entry.insert(0, f"{T:.2f}")
    reset_for_parameter_change()

def update_coupling(val):
    global J
    J = float(val)
    coupling_entry_main.delete(0, tk.END)
    coupling_entry_main.insert(0, f"{J:.2f}")
    reset_for_parameter_change()

def update_temp_entry(val):
    try:
        T_val = float(val)
        if 0.1 <= T_val <= 5.0:
            temp_slider.set(T_val)
    except ValueError:
        pass
    reset_for_parameter_change()

def update_coupling_entry(val):
    try:
        J_val = float(val)
        if -2.0 <= J_val <= 2.0:
            coupling_slider.set(J_val)
    except ValueError:
        pass
    reset_for_parameter_change()

def update_plot_choice(event):
    global data_buffer, line, plot_observable, sweepcount
    plot_observable = observable_dropdown.get()
    if plot_observable == "Energy":
        ax.set_ylabel("Energy / (L^2 J)")
        ax.set_ylim(-2, 2)
    elif plot_observable == "Magnetization":
        ax.set_ylabel("Magnetization (M/$L^2$)")
        ax.set_ylim(-1, 1)
    elif plot_observable == "Acceptance":
        ax.set_ylabel("Acceptance")
        ax.set_ylim(0, 1)
    data_buffer.clear()
    line.set_ydata([0]*100)
    ax.set_title(f"Live {plot_observable} Vs. Time")
    canvas.draw()

def update_observable_labels():
    energy_label.config(text=f"Energy: {E / (L**2):.3f}")
    magnetization_label.config(text=f"Magnetization: {M / (L**2):.3f}")
    acceptance_label.config(text=f"Acceptance: {Acceptance/sweepcount:.3f}")
    timer_label.config(text=f"Sweeps/second: {sweep_counter / (time.time() - timer - 0.1):.3f}")
    root.after(50, update_observable_labels)

def update_algorithm_choice(event):
    global algorithm, flipped_sites
    algorithm = algorithm_dropdown.get()
    if algorithm == "Overrelaxation":
        # For Overrelaxation, update all sites, so flipped_sites is all indices
        flipped_sites = [(i, j) for i in range(L) for j in range(L)]

def update_size_choice(event):
    global L, scale, spins, rgb_array, label_img, label, algorithm, flipped_sites
    L = int(size_dropdown.get())
    scale = 512 // L
    spins = np.random.rand(L, L) * 2 * np.pi
    rgb_array = init_rgb_array(spins, L)
    pil_img = update_spins_image(spins, [], rgb_array, scale)
    label_img = ImageTk.PhotoImage(pil_img)
    if algorithm == "Overrelaxation":
        # For Overrelaxation, update all sites, so flipped_sites is all indices
        flipped_sites = [(i, j) for i in range(L) for j in range(L)]
    reset_for_parameter_change()

def update_plot(E, M, L, data_buffer):
    global root, line
    if plot_observable == "Energy":
        data_buffer.append(E / L**2)
    elif plot_observable == "Magnetization":
        data_buffer.append(M / L**2)
    elif plot_observable == "Acceptance":
        data_buffer.append(Acceptance / sweepcount)
    line.set_ydata(list(data_buffer) + [0] * (100 - len(data_buffer)))
    root.after_idle(canvas.draw)

def run_simulation():
    global spins, T, J, Acceptance, label_img, label, count, E, M, sweepcount, algorithm, theta, flipped_sites, sweep_counter
    if algorithm == "Metropolis":
        spins, E, Acceptance, flipped_sites, sweepcount = Metropolis(spins, T, J, L, E, Acceptance, sweepcount)
        Mx, My, M = Mag(spins)
    elif algorithm == "Metropolis Limited Change":
        spins, E, Acceptance, flipped_sites, sweepcount = Metropolis_Limited_Change(spins, T, J, L, E, Acceptance, theta, sweepcount)
        AcceptanceRatio = Acceptance / sweepcount
        theta = update_theta(AcceptanceRatio, theta)
        Mx, My, M = Mag(spins)
    elif algorithm == "Wolff":
        spins, flipped_sites = Wolff(spins, T, J, L)
        E = Energy(spins, J)
        Mx, My, M = Mag(spins)
    elif algorithm == "Overrelaxation":
        spins = overrelaxation(spins, L)
        E = Energy(spins, J)
        Mx, My, M = Mag(spins)
    sweep_counter += 1
    # update the image
    pil_img = update_spins_image(spins, flipped_sites, rgb_array, scale)
    label_img = ImageTk.PhotoImage(pil_img)
    label.configure(image=label_img)

    # update the plot after 3 run_simulation calls (~15ms)
    count = (count + 1) % 3
    if count == 0:
        update_plot(E, M, L, data_buffer)

    root.after(5, run_simulation)

# initialize spins randomly
spins = np.random.rand(L, L) * 2 * np.pi
E = Energy(spins,J)
Mx,My,M = Mag(spins)

# initialize the RGB image array
rgb_array = init_rgb_array(spins, L)

# initialize the timer
timer = time.time() # timer for sweeps per second

## Set up the GUI
# Create the main window
root = tk.Tk()
root.title("XY Model GUI")

# Create the image frame and set it to the left side of the window
image_frame = ttk.Frame(root)
image_frame.pack(side=tk.LEFT)

img = tk.PhotoImage(width=L, height=L)
label = ttk.Label(image_frame, image=img)
label.pack()

# Create the slider frame and set it to the right side of the window
slider_frame = ttk.Frame(root)
slider_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

plot_frame = ttk.Frame(slider_frame)
plot_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

# Create the matplotlib figure and axis for plotting
plt.style.use('fast')
fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Data buffer for live plot (e.g. tracking magnetization or acceptance)
data_buffer = deque(maxlen=100)
x_vals = list(range(100))
line, = ax.plot(x_vals, [0]*100)
ax.set_ylim(-1, 1)
ax.set_title(f"Live {plot_observable} Vs. Time")
ax.set_xlabel("Time")
ax.set_ylabel(f"{plot_observable}")
fig.tight_layout()

# Create the sliders and add them to the slider frame
temp_label = ttk.Label(slider_frame, text="Temperature (T):")
temp_label.grid(row=1, column=0, padx=5, pady=5)
temp_slider = ttk.Scale(slider_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, value=T, length=250)
temp_slider.grid(row=1, column=1, padx=5, pady=5)
temp_slider.config(command=update_temp)
temp_entry = ttk.Entry(slider_frame, width=5)
temp_entry.insert(0, str(T))  # set initial value
temp_entry.bind("<Return>", lambda event: update_temp_entry(temp_entry.get()))
temp_entry.grid(row=1, column=2, padx=5, pady=5)


coupling_label = ttk.Label(slider_frame, text="Coupling (J):")
coupling_label.grid(row=2, column=0, padx=5, pady=5)
coupling_slider = ttk.Scale(slider_frame, from_=-2.0, to=2.0, orient=tk.HORIZONTAL, value=J, length=250)
coupling_slider.grid(row=2, column=1, padx=5, pady=5)
coupling_slider.config(command=update_coupling)
coupling_entry_main = ttk.Entry(slider_frame, width=5)
coupling_entry_main.insert(0, str(J))  # set initial value
coupling_entry_main.bind("<Return>", lambda event: update_coupling_entry(coupling_entry_main.get()))
coupling_entry_main.grid(row=2, column=2, padx=5, pady=5)

# magneticfield_label = ttk.Label(slider_frame, text="Magnetic Field (h):")
# magneticfield_label.grid(row=3, column=0, padx=5, pady=5)
# magneticfield_slider = ttk.Scale(slider_frame, from_=-2.0, to=2.0, orient=tk.HORIZONTAL, value=h, length=250)
# magneticfield_slider.grid(row=3, column=1, padx=5, pady=5)
# magneticfield_slider.config(command=update_magneticfield)
# magneticfield_entry_main = ttk.Entry(slider_frame, width=5)
# magneticfield_entry_main.insert(0, str(h))  # set initial value
# magneticfield_entry_main.bind("<Return>", lambda event: update_magneticfield_entry(magneticfield_entry_main.get()))
# magneticfield_entry_main.grid(row=3, column=2, padx=5, pady=5)

observable_label = ttk.Label(slider_frame, text="Observable to Plot:")
observable_label.grid(row=4, column=0, padx=5, pady=5)
observable_dropdown = ttk.Combobox(slider_frame, values=["Magnetization", "Energy", "Acceptance"], state="readonly")
observable_dropdown.current(0)
observable_dropdown.grid(row=4, column=1, padx=5, pady=5)
observable_dropdown.bind("<<ComboboxSelected>>", update_plot_choice)

size_label = ttk.Label(slider_frame, text="Size (L):")
size_label.grid(row=6, column=0, padx=5, pady=5)
size_dropdown = ttk.Combobox(slider_frame, values=[4, 8, 16, 32, 64, 128, 256], state="readonly")
size_dropdown.current(4)
size_dropdown.grid(row=6, column=1, padx=5, pady=5)
size_dropdown.bind("<<ComboboxSelected>>", update_size_choice)

acceptance_label = ttk.Label(slider_frame, text=f"Acceptance: {Acceptance/sweepcount:>7.3f}")
acceptance_label.grid(row=4, column=2, padx=5, pady=5)
acceptance_label.config(width=25)
energy_label = ttk.Label(slider_frame, text=f"Energy: {E / (L**2):>7.3f}")
energy_label.grid(row=5, column=2, padx=5, pady=5)
energy_label.config(width=25)
magnetization_label = ttk.Label(slider_frame, text=f"Magnetization: {M / (L**2):>7.3f}")
magnetization_label.grid(row=6, column=2, padx=5, pady=5)
magnetization_label.config(width=25)
timer_label = ttk.Label(slider_frame, text=f"Sweeps/second: {sweep_counter / (time.time() - timer - 0.1):>7.3f}")
timer_label.grid(row=7, column=2, padx=5, pady=5)
timer_label.config(width=25)


algorithm_label = ttk.Label(slider_frame, text="Algorithm:")
algorithm_label.grid(row=5, column=0, padx=5, pady=5)
algorithm_dropdown = ttk.Combobox(slider_frame, values=["Metropolis", "Metropolis Limited Change", "Wolff", "Overrelaxation"], state="readonly")
algorithm_dropdown.current(0)
algorithm_dropdown.grid(row=5, column=1, padx=5, pady=5)
algorithm_dropdown.bind("<<ComboboxSelected>>", update_algorithm_choice)

# precompile numba functions
if not CACHE:
    Wolff(spins, T, J, L)
    Metropolis_Limited_Change(spins, T, J, L, E, Acceptance, theta, sweepcount)
    overrelaxation(spins, L)

# Get a fresh timer so the initial sweeps/sec calculation isn't skewed
timer = time.time()
# run the window and simulation
root.after(50, update_observable_labels)
root.after(5, run_simulation)
root.mainloop()