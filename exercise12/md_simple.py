"""
Simple Molecular Dynamics code for course FYS-4096 Computational Physics

Problem 2:
- Make the code to work and solve H2 using the Morse potential.
- Modify and comment especially at parts where it reads "# ADD"
- Follow instructions on ex12.pdf

Problems 3:
- Add observables: temperature, distance, and heat capacity.
- Follow instructions on ex12.pdf


Problem 2:

Default settings:
    - Update/integration algorithm: Velocity Verlet
    - Boundary conditions: None
    - Dimensions: 3d

Problem 3: 
    What values do you get with the default settings?
        -Internuclear distance, 1.407036871517909
        -Temperature 92.55776366505762 with std of 0.6566211196793035
        -Heat capacity 4.049809042671255e-22 <- Still bad scaling

"""



from numpy import *
from matplotlib.pyplot import *
import matplotlib.patches as mpatches

class Atom:
    def __init__(self,index,mass,dt,dims):
        self.index = index
        self.mass = mass
        self.dt = dt
        self.dims = dims
        self.LJ_epsilon = None
        self.LJ_sigma = None
        self.R = None
        self.v = None
        self.force = None

    def set_LJ_parameters(self,LJ_epsilon,LJ_sigma):
        self.LJ_epsilon = LJ_epsilon
        self.LJ_sigma = LJ_sigma

    def set_position(self,coordinates):
        self.R = coordinates
        
    def set_velocity(self,velocity):
        self.v = velocity

    def set_force(self,force):
        self.force = force

class Observables:
    # Observables class, for storing data
    def __init__(self):
        self.E_kin = []
        self.E_pot = []
        self.E_tot = []
        self.E_totsq = []
        self.distance = []
        self.Temperature = []

def calculate_energetics(atoms):
    """ Calculates the kinetic and potential energy of the system """
    N = len(atoms)
    V = 0.0
    E_kin = 0.0

    # calculation of kinetic and potential energy
    # Kinetic part
    for i in range(N):
        vel = atoms[i].v
        # Sum the components of kinetic energy (vx2+vy2+vz2)
        E_kin += 0.5*atoms[i].mass*sum(vel**2)

    # Potential part
    for i in range(0,N-1):
        for j in range(i+1,N):
            V += pair_potential(atoms[i],atoms[i+1])


    return E_kin, V

def calculate_force(atoms): 
    # Returns the net force on each atom in the system as a numpy array, where each 
    # element is an array representing the components of the force (in this case, 3d)
    N = len(atoms)
    ij_map = zeros((N,N),dtype=int)
    Fs = []
    ind = 0

    # Calculate pair force for every adjacent atom pair
    for i in range(0,N-1):
        for j in range(i+1,N):
            Fs.append(pair_force(atoms[i],atoms[j]))

            # Map the index (used in accessing Fs later)
            ij_map[i,j] = ind
            ij_map[j,i] = ind
            ind += 1
    F = []

    # For each atom, calculate the net force
    for i in range(N):
        f = zeros(shape=shape(atoms[i].R)) # F_i
        for j in range(N):
            ind = ij_map[i,j] # Index for the atom pair

            # F_mi = -F_im
            if i<j:
                f += Fs[ind]
            elif i>j:
                f -= Fs[ind]
        F.append(f)
    F = array(F)
    return F

def pair_force(atom1,atom2):
    """ Calculates the pair force via Morse model """
    return Morse_force(atom1,atom2)

def pair_potential(atom1,atom2):
    """ Calculates the pair potential via Morse model """
    return Morse_potential(atom1,atom2)

def Morse_potential(atom1,atom2):
    """ Morse potential, lecture slides of week 11 """
    # H2 parameters given here
    De = 0.1745
    re = 1.40
    a = 1.0282
    r = atom1.R-atom2.R # vector from atom2 to atom1
    dr = sqrt(sum(r**2)) # length of the vector (=distance r in equation)

    u_M = De*(1-exp(-a*(dr-re)))**2-De
    
    return u_M

def Morse_force(atom1,atom2):
    """ Morse force, obtained from the derivative of the morse potential """

    # H2 parameters
    De = 0.1745
    re = 1.40
    a = 1.0282
    r = atom1.R-atom2.R
    dr = sqrt(sum(r**2))

    # F = -du/dr
    F = -2*a*De*exp(-a*(dr-re))*(1-exp(-a*(dr-re))) 
    # Direction of force: unit vector of r
    r_normalized = r/dr
    return F*r_normalized

def lennard_jones_potential(atom1,atom2):
    # Lorentz-Berthelot mixing for epsilon and sigma
    epsilon = sqrt(atom1.LJ_epsilon*atom2.LJ_epsilon)
    sigma = (atom1.LJ_sigma+atom2.LJ_sigma)/2
    # If interested one could add the calculation of LJ potential here
    #
    #
    return 

def lennard_jones_force(atom1,atom2):
    # Lorentz-Berthelot mixing for epsilon and sigma
    epsilon = sqrt(atom1.LJ_epsilon*atom2.LJ_epsilon)
    sigma = (atom1.LJ_sigma+atom2.LJ_sigma)/2
    # If interested one could add the calculation of LJ force here
    #
    #
    return 

def velocity_verlet_update(atoms):
    """ Velocity Verlet method for updating the velocity and distance of atoms """
    dt = atoms[0].dt
    dt2 = dt**2

    # Update the position of each atom with the formula
    for i in range(len(atoms)):
        atoms[i].R += dt*atoms[i].v+dt2/2/atoms[i].mass*atoms[i].force # Position update
    # Calculate the new forces (due to changing the positions)
    Fnew = calculate_force(atoms)
    # Update the velocity of each atom with the formula
    for i in range(len(atoms)):
        atoms[i].v += dt/2/atoms[i].mass*(Fnew[i]+atoms[i].force)# Velocity update
        atoms[i].force = Fnew[i] # Force update
    return atoms
    
def initialize_positions(atoms):
    # diatomic case
    atoms[0].set_position(array([-0.8,0.0,0.0]))
    atoms[1].set_position(array([0.7,0.0,0.0]))

def initialize_velocities(atoms):
    # diatomic case 
    dims = atoms[0].dims
    kB=3.16e-6 # in hartree/Kelvin
    for i in range(len(atoms)):
        v_max = sqrt(3.0/atoms[i].mass*kB*10.0)
        atoms[i].set_velocity(array([1.0,0.0,0.0])*v_max)
    atoms[1].v = -1.0*atoms[0].v

def initialize_force(atoms):
    F=calculate_force(atoms)
    for i in range(len(atoms)):
        atoms[i].set_force(F[i])

def Temperature(atoms,E_k):
    # Boltzmann constant in Hartree/Kelvin
    kB = 3.16e-6
    # Equipartition theorem: d*N/2*kb*T=<K>
    T = 2*E_k/kB/atoms[0].dims/len(atoms)
    return T

def calculate_observables(atoms,observables):
    """ Calculate the observables in Observables class """
    E_k, E_p = calculate_energetics(atoms)
    observables.E_kin.append(E_k)
    observables.E_pot.append(E_p)
    E_tot = E_k+E_p

    # Total energy and the squared total energy, used for heat capacity calculation
    observables.E_tot.append(E_tot)
    observables.E_totsq.append(E_tot**2)
    # Temperature
    T = Temperature(atoms,E_k)
    observables.Temperature.append(T)
    # Distance, 2 atoms
    dist = sqrt(sum(atoms[1].R - atoms[0].R)**2)
    observables.distance.append(dist)
    
    return observables

def main():
    N_atoms = 2
    dims = 3
    dt = 0.1
    mass = 1860.0

    # Initialize atoms
    atoms = []    
    for i in range(N_atoms):
        atoms.append(Atom(i,mass,dt,dims))
        # LJ not used, but the next line shows a way to set LJ parameters 
        atoms[i].set_LJ_parameters(0.1745,1.25) 

    # Initialize observables
    observables = Observables()

    # Initialize positions, velocities, and forces
    initialize_positions(atoms)
    initialize_velocities(atoms)
    initialize_force(atoms)

    for i in range(100000):
        atoms = velocity_verlet_update(atoms)

        # Calculate observables every 10*dt = 1 time unit
        if ( i % 10 == 0):
            observables = calculate_observables(atoms,observables)            

    # Print energies
    E_kin = array(observables.E_kin)
    E_pot = array(observables.E_pot)
    E_tot = array(observables.E_tot)
    E_totsq = array(observables.E_totsq)
    T = array(observables.Temperature)
    r = array(observables.distance)
    Cv = (mean(E_totsq)-mean(E_tot)**2)/mean(T)**2 # Still wrong
    print('E_kin',mean(E_kin))
    print('E_pot',mean(E_pot))
    print('E_tot',mean(E_tot))
    print('Internuclear distance,',mean(r))
    print('Temperature',mean(T), '+/-',std(T)/sqrt(len(T)))
    print('Heat capacity',Cv)

    # Plot total energy vs. time from the simulation observables
    figure()
    Emean =  r'$\langle E \rangle$ = {:.3f}'.format(mean(E_tot))
    text = mpatches.Patch(color='white',label=Emean)
    plot(E_tot)
    xlabel('time')
    ylabel(r'$E_{tot}$')
    # Whole plot is too long
    xlim(0,5000)
    legend(handles=[text],loc=1)

    figure()
    temperature =  r'$ \langle T \rangle$ = {0:.3f} +/- {1:.3f}'.format(mean(T), std(T)/sqrt(len(T)))
    text = mpatches.Patch(color='white',label=temperature)
    plot(T)
    xlabel('time')
    ylabel('Temperature')
    # Whole plot is too long
    xlim(0,5000)
    legend(handles=[text],loc=1)

    figure()
    ri =  r'$\langle r_i \rangle$ = {:.3f}'.format(mean(r))
    text = mpatches.Patch(color='white',label=ri)
    plot(r)
    xlabel('time')
    ylabel('Internuclear distance')
    # Whole plot is too long
    xlim(0,5000)
    legend(handles=[text],loc=1)
    show()

if __name__=="__main__":
    main()
        
