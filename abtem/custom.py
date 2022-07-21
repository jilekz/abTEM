import numpy as np
import scipy.constants as c
from abtem import CTF
from abtem.utils import energy2sigma
from abtem.potentials import PotentialArray


def get_potential(phase_shift,extent,energy,gpts,num = 1,slice_thickness = 20): 
    # phase_shift - in rad
    # extent - in A
    # energy - in eV
    # num - number of slices of thickness `slice_thickness`
    
    sampling = extent/gpts #A #potencial will be sampled with this value 
    array=np.zeros(gpts,dtype=np.float64)

    ctf_tmp = CTF(energy = energy); wavelength = ctf_tmp.wavelength * 1e-10 # in meters
    electron_energy=energy*c.e #J

    #POTENTIAL
#    interaction_parameter = 2*np.pi/wavelength/electron_energy*((c.m_e*c.c**2+c.e*electron_energy)/(2*c.m_e*c.c**2+c.e*electron_energy))#kirkland (5.6) in rad/joul/meter
#    proj_pot_val = phase_shift/interaction_parameter #Jm
#    proj_pot_val_eva = proj_pot_val/(1e-10)/c.e #PotentialArray's array should be in eV*A not eV how I previously thought.. each slice contains already projected potential and total projected potencial is just sum of them

    interaction_parameter = energy2sigma(energy)
    proj_pot_val_eva = phase_shift/interaction_parameter

    y,x = np.indices(array.shape)
    array[y<gpts[0]//2] = proj_pot_val_eva # eV*A it is projected potential in a given slice

    array=array/num
    print(sampling)
    potential = PotentialArray(array=np.array([array]*num),slice_thicknesses=np.array([slice_thickness]*num),extent=extent,sampling=sampling)
    
#    proj_pot_max=np.max(potential.array)*num #eV*A
#    print(proj_pot_max) 
#    print(proj_pot_max*interaction_parameter*1e-10*c.e) # phase shift sanity check
    
    return(potential)

def get_gaussian_spread(alpha,energy,reduced_brightness,I): # alpha in mrad #energy in eV # reduced brightness in si# I in A
    alpha = alpha*1e-3
    phi_star=energy*(1-c.elementary_charge*energy/2/c.m_e/c.c**2)
    d_50 = 2/np.pi*np.sqrt(I/reduced_brightness/phi_star)*1/alpha
    sigma = d_50/np.sqrt(np.log(256))
    return(sigma/1e-10) # returning in angstrom
