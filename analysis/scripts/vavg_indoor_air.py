from sympy import * 
from spb import *

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, "../scripts")
import wall_cond as w


class PhysicalConstants:
    rho= 1.225 #kg/m^3 
    cp= 1005 # J/(kg-K) 
    V= 2800 #m^3 
    T0= 293.15 # K 


class VavgIndoorAir:
    def __init__(self, pc:PhysicalConstants, n_mins=60, calcs_per_min=4,):
        # physical constant for the room 
        self.pc = pc 

        # TODO initial val for dt should be input to twall 
        self.times = np.linspace(start=0, stop=n_mins, num=n_mins*calcs_per_min, endpoint=False)
        self.dt = self.times[1] - self.times[0]

        # initialize temperature array 
        self.temps = np.zeros(len(self.times))
        self.temps[0] = self.pc.T0

        # current information for the whole class
        self.index = 0
        
        # init wall conduction calculation 
        self.phys_wall = w.FabricPhysicalConstants()
        self.twall = w.TransientWallConduction(self.phys_wall, self.times, self.pc.T0) 
        self.surface_temps = np.zeros(len(self.times))
        self.wall_heat_flux = np.zeros(len(self.times))
        self.all_wall_temps = np.zeros((len(self.times), self.twall.M))

        pass
   

    def fabric_at_t(self):
        # find the index of the current time...
        wall_temps_at_t = self.twall.calc_Tx_at_t(self.index, self.temps[self.index]) 

        # update knowledge of wall temps across time -> result is N*M matrix 
        self.all_wall_temps[self.index] = wall_temps_at_t 

        # update knowlegede about surface temps over time 
        wall_surface_temp = wall_temps_at_t[self.twall.M - 1]
        self.surface_temps[self.index] = wall_surface_temp

        # update knowledge about heat transfer over time 
        self.E_fabric = -1*self.phys_wall.h_int*self.phys_wall.A*(self.temps[self.index] - wall_surface_temp)
        self.wall_heat_flux[self.index] = self.E_fabric

        
        # TODO determine if this should be negative 
        
        return self.E_fabric
    
    def interior_at_t(self):
        # interior heat sources 
        return 0
    
    def ventilation_at_t(self):
        return 0
    
    def energy_at_t(self):
        return self.fabric_at_t() + self.interior_at_t() + self.ventilation_at_t()
    
    def step_indoor_calc(self):
        pass
        
    
    def calc_transient_indoor_air(self):
        # constant exponent 
        const_exp = self.dt/(self.pc.rho*self.pc.cp*self.pc.V) 

        # iterating in time: t_{i} => t_{i=N}
        for i, _ in enumerate(self.temps):  
            if i < len(self.temps) - 1: 
                self.index = i
                self.temps[i+1] = self.temps[i] + const_exp*self.energy_at_t()
        # TODO In similar way to transient wall conduction make seperate function to calculate a single step, then a big function to calculate all the steps

        return self.times, self.temps
    

