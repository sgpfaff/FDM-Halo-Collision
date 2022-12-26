#!/usr/bin/env python
# coding: utf-8

# In[71]:


from gravhopper import Simulation, IC
from astropy import units as u, constants as const
from galpy import potential, df
from galpy.potential import MWPotential2014 as mwp14
from galpy.potential import LogarithmicHaloPotential
from pynbody.analysis.profile import Profile
import matplotlib.pyplot as plt

# Set up useful constants
G = 4.3009172706E-3 *u.pc*(u.km**2)/(u.M_sun*(u.s**2))
step = 1000
mm = 8e9

def M_sol(r, init_values):
    m_boson, r_c, r_eps, ratio, r_s, c_dens = init_values
    a = .301*r/r_c
    v = 3465*((1+(a**2))**7)*np.arctan(a)
    return ((4.2E6 * u.M_sun)/(((m_boson/(10**-23 *u.eV))**2)*(r_c/(1000*u.pc))*((1+(a**2))*7)))*(3465*(a**13) + 23100*(a**11) + 65373*(a**9) + 101376*(a**7) + 92323*(a**5) + 48580*(a**3) - 3465*a + v.value)

def M_nfw(r, init_values):
    m_boson, r_c, r_eps, ratio, r_s, c_dens = init_values
    s_dens = c_dens*(r_eps/r_s)*((1+(r_eps/r_s))**2)/((1+(0.091*(ratio**2)))**8)
    return  4*np.pi*s_dens*(r_s**3)*(np.log(1+(r/r_s))-((r/r_s)/(1+(r/r_s))))

def M_dm(r, init_values):
    m_boson, r_c, r_eps, ratio, r_s, c_dens = init_values
    if r <= r_eps:
        M_dm = M_sol(r, init_values)
    else:
        M_dm = M_sol(r_eps, init_values) - M_nfw(r_eps, init_values) + M_nfw(r, init_values)
    return M_dm
        
def rot_vel(r, M):
    return np.sqrt((G*M)/(r))


def make_FDM_halo(init_values, center, vel, c):
    m_boson, r_c, r_eps, ratio, r_s, c_dens = init_values
    
    for r in r_list:
        if r.value <= step:
            amt_of_DM_pts = M_dm(step*u.pc, init_values)/(mm*u.M_sun)
            new_r_list = np.arange(step/amt_of_DM_pts, step, step/amt_of_DM_pts)*u.pc

            for i, radius in enumerate(new_r_list): # Constant Density Sphere
                v_circ = rot_vel(radius, M_dm(radius, init_values))
                theta = 48*np.pi*i*u.rad/amt_of_DM_pts
                phi = np.pi*i*u.rad/amt_of_DM_pts

                x = radius * np.cos(theta) * np.sin(phi) + center[0]
                y = radius * np.sin(theta) *np.sin(phi) + center[1]
                z = radius * np.cos(phi) + center[2]

                v_x = np.cos(phi) * np.sin(theta) * v_circ.value + vel[0].value
                v_y = -1 * np.cos(phi) * np.cos(theta) * v_circ.value + vel[1].value
                v_z = 0 + vel[2].value

                new_particle = {'pos':[x, y, z]*u.pc, 'vel':[v_x, v_y, v_z]*u.km/u.s, 'mass':[mm]*u.Msun}
                sim.add_IC(new_particle)
        
            
        else: # Concentric Shells of Mass
            amt_of_DM_pts = (M_dm(r, init_values) - M_dm(r-step*u.pc, init_values))/(mm*u.M_sun)
            radius = r.value + (step/2)
            v_circ = rot_vel(r, M_dm(r, init_values))
            i = 0
            while i <= int(amt_of_DM_pts):
                theta = 48*np.pi*i*u.rad/amt_of_DM_pts
                phi = np.pi*i*u.rad/amt_of_DM_pts
                x = radius * np.cos(theta) * np.sin(phi) * u.pc + center[0]
                y = radius * np.sin(theta) *np.sin(phi) * u.pc + center[1]
                z = radius * np.cos(phi) * u.pc + center[2]

                v_x = np.cos(phi) * np.sin(theta) * v_circ.value + vel[0].value
                v_y = -1 * np.cos(phi) * np.cos(theta) * v_circ.value + vel[1].value
                v_z = 0 + vel[2].value

                new_particle = {'pos':[x, y, z]*u.kpc, 'vel':[v_x, v_y, v_z]*u.km/u.s, 'mass':[mm]*u.Msun}
                sim.add_IC(new_particle)

                i += 1

#Simulation
sim = Simulation(dt=0.1*u.Myr, eps=0.5*u.kpc)            

init_values_1 = [0.554E-23 *u.eV, 250*u.pc, 100*u.pc, 0.67, 10000*u.pc] #[m_boson, r_c, r_eps, ratio, r_s]
m_boson, r_c, r_eps, ratio, r_s = init_values_1
c_dens = 1.9*((m_boson/(10**-23 * u.eV))**-2)*((r_c/(u.pc * 1000))**-4) *u.M_sun/(u.pc**3)
init_values_1.append(c_dens)
init_values_2 = init_values_1.copy()

r_list = [rad*u.pc for rad in np.arange(step,10000+step,step)]

print(f'Mass at {r_s}: {M_dm(r_s, init_values_1)}')

make_FDM_halo(init_values_1, [0*u.pc, 0*u.pc, 0*u.pc], [0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s], "blue")
sim.run(400)
sim.movie_particles('DF.mp4', unit=u.kpc, fps=30, xlim=[-25,25], ylim=[-25,25])

