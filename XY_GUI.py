import tkinter as tk
from tkinter import ttk
import numpy as np
import random
from numba import jit, prange
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import colorsys

scale = 8 # scaling factor for display
update_delay = 5 # milliseconds between updates

# parameters
L = 50 # lattice size (LxL)
T = 0.866 # temperature
J = 1 # coupling constant
Acceptance = 0 # initialize acceptance counter
plot_observable = "Magnetization" # what to plot in the live graph
algorithm = "Metropolis" # which algorithm to use
sweepcount = 1 # number of sweeps done
count = 0 # counter for plot updates

# initialize spins randomly
spins = 2*np.pi*np.random.rand(L, L)

@jit(nopython=True)
def Energy(spins,J): # Calculates the energy of a given lattice of spins  
  # Assumes a square lattice
  TotalEnergy=0
  L=spins.shape[0] 
  for i in prange(L):
    for j in prange(L):
      TotalEnergy+= np.cos(spins[i,j]-spins[i-1,j]) + np.cos(spins[i,j]-spins[(i+1)%L,j])  +np.cos(spins[i,j]-spins[i,j-1]) + np.cos(spins[i,j]-spins[i,(j+1)%L])
  TotalEnergy*=-J/2
  return TotalEnergy

@jit(nopython=True)
def Mag(spins): #magnetization function returns X,Y components and Magnitude [Mx,My,M]
  #The magnetization is given by the sum of the X and Y spin components.
  #this function returns X and Y components in an array
  Mx=np.sum(np.cos(spins))
  My=np.sum(np.sin(spins))
  M = np.sqrt(Mx**2+My**2)
  return np.array([Mx,My,M])

@jit(nopython=True)
def sweep(spins,T,J,Acceptance, sweepcount):
    L = spins.shape[0]
    sweepcount += L**2
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

    return spins, Acceptance, flipped_sites, sweepcount

@jit(nopython=True)
def Wolff(spins,J,T):
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
    # print(cluster)
    # print(already_flipped)


def spins_to_image_init(spins):
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
    return rgb_array

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

def reset_for_parameter_change():
    global Acceptance, sweepcount, E, Mx, My, M
    Acceptance = 0
    sweepcount = 1
    E = Energy(spins,J)
    Mx, My, M = Mag(spins)

def update_temp(val):
    global T
    T = float(val)
    temp_entry.delete(0, tk.END)
    temp_entry.insert(0, f"{T:.2f}")
    reset_for_parameter_change()

def update_coupling(val):
    global J
    J = float(val)
    coupling_entry.delete(0, tk.END)
    coupling_entry.insert(0, f"{J:.2f}")
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
    energy_label.config(text=f"Energy / (L^2 J): {E / (L**2):.3f}")
    magnetization_label.config(text=f"Magnetization (M/L^2): {M / (L**2):.3f}")
    acceptance_label.config(text=f"Acceptance: {Acceptance/sweepcount:.3f}")
    root.after(50, update_observable_labels)

def update_algorithm_choice(event):
    global algorithm
    algorithm = algorithm_dropdown.get()
    # reset_for_parameter_change()

def run_simulation():
    global spins, T, J, Acceptance, label_img, label, count, E, M, sweepcount, algorithm
    if algorithm == "Metropolis":
        spins, Acceptance, flipped_sites, sweepcount = sweep(spins, T, J, Acceptance, sweepcount)
    elif algorithm == "Wolff":
        spins, cluster = Wolff(spins, J, T)
        E = Energy(spins, J)
        Mx, My, M = Mag(spins)

    pil_img = spins_to_image(spins)
    label_img = ImageTk.PhotoImage(pil_img)
    label.configure(image=label_img)

    # update the plot after 2 run_simulation calls (~10ms)
    count = (count + 1) % 2
    if count == 0:
        if plot_observable == "Energy":
            data_buffer.append(E / L**2)
        elif plot_observable == "Magnetization":
            data_buffer.append(M / L**2)
        elif plot_observable == "Acceptance":
            data_buffer.append(Acceptance / sweepcount)
        line.set_ydata(list(data_buffer) + [0] * (100 - len(data_buffer)))
        canvas.draw()

    root.after(5, run_simulation)

# initialize spins randomly
spins = np.random.rand(L, L) * 2 * np.pi
E = Energy(spins,J)
Mx,My,M = Mag(spins)

# initialize the RGB image array
rgb_array = spins_to_image_init(spins)

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
plot_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

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
temp_label.grid(row=0, column=0, padx=5, pady=5)
temp_slider = ttk.Scale(slider_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, value=T)
temp_slider.grid(row=0, column=1, padx=5, pady=5)
temp_slider.config(command=update_temp)
temp_entry = ttk.Entry(slider_frame, width=5)
temp_entry.insert(0, str(T))  # set initial value
temp_entry.bind("<Return>", lambda event: update_temp_entry(temp_entry.get()))
temp_entry.grid(row=0, column=2, padx=5, pady=5)


coupling_label = ttk.Label(slider_frame, text="Coupling (J):")
coupling_label.grid(row=1, column=0, padx=5, pady=5)
coupling_slider = ttk.Scale(slider_frame, from_=-2.0, to=2.0, orient=tk.HORIZONTAL, value=J)
coupling_slider.grid(row=1, column=1, padx=5, pady=5)
coupling_slider.config(command=update_coupling)
coupling_entry = ttk.Entry(slider_frame, width=5)
coupling_entry.insert(0, str(J))  # set initial value
coupling_entry.bind("<Return>", lambda event: update_coupling_entry(coupling_entry.get()))
coupling_entry.grid(row=1, column=2, padx=5, pady=5)

observable_label = ttk.Label(slider_frame, text="Observable to Plot:")
observable_label.grid(row=3, column=0, padx=5, pady=5)

observable_dropdown = ttk.Combobox(slider_frame, values=["Magnetization", "Energy", "Acceptance"], state="readonly")
observable_dropdown.current(0)
observable_dropdown.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

observable_dropdown.bind("<<ComboboxSelected>>", update_plot_choice)

acceptance_label = ttk.Label(slider_frame, text=f"Acceptance: {Acceptance/sweepcount:.3f}")
acceptance_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

energy_label = ttk.Label(slider_frame, text=f"Energy / (L^2 J): {E / (L**2):.3f}")
energy_label.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
magnetization_label = ttk.Label(slider_frame, text=f"Magnetization (M/L^2): {M / (L**2):.3f}")
magnetization_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5)


algorithm_label = ttk.Label(slider_frame, text="Algorithm:")
algorithm_label.grid(row=7, column=0, padx=5, pady=5)
algorithm_dropdown = ttk.Combobox(slider_frame, values=["Metropolis", "Wolff"], state="readonly")
algorithm_dropdown.current(0)
algorithm_dropdown.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

algorithm_dropdown.bind("<<ComboboxSelected>>", update_algorithm_choice)


# run the window and simulation
root.after(50, update_observable_labels)
root.after(5, run_simulation)
root.mainloop()