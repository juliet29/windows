import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from icecream import ic


class FabricPhysicalConstants:
    # wind enegineering problem 6 defaults 
    rho= 2300 # kg/m^3 
    cp= 750 # J/(kg-K) 
    k=0.8 # W/(m-K)
    h_int = 4 # W/(m-K)
    h_ext = 4 # W/(m-K)
    T0 = 293.15 # K 
    Tinf_ext = 295.65 # K 
    Tinf_int = 295.65 # K 
    A = 1 # m2 # TODO find realistic 

    alpha = k/(cp*rho) # thermal diffusivity

class TransientWallConduction:
    def __init__(self, pc:FabricPhysicalConstants, times):
        self.pc = pc
        # x vals init in example => th = 0.10 m, dt = 0.010 m 
        thickness = 0.10 
        self.dx = 0.010
        self.x_vals = np.arange(start=0, stop=thickness, step=self.dx)
        self.M = len(self.x_vals)

        # # time vals init => needs to be passed in now 
        # since havent checked tau => ensure that dt = 15s or less
        self.times = times
        self.dt = self.times[1] - self.times[0]

        # initialize N*M matrix - time * x nodes 
        self.Ttx = np.zeros((self.N, self.M))
        self.Ttx[0,:] = self.pc.T0
        self.Ttx[0,0] = self.pc.Tinf_ext 
        self.Ttx[0, self.M-1] = self.pc.Tinf_int

        # time constant
        # # TODO check tau  
        self.tau = self.pc.alpha * self.dt / (self.dx**2)
        

    def calc_boundary_nodes(self, i):
        # TODO replace i with a class value instead of passing it in always
        beta = lambda h: h*self.dx/self.pc.k
        eq = lambda Tself, Tnb, Tinf, h: (1 -2*self.tau - 2*self.tau*beta(h))*Tself + 2*self.tau*Tnb + 2*self.tau*beta(h)*Tinf
        
        # T0 at exterior
        T0 = eq(self.Ttx[i,0], self.Ttx[i, 1], self.pc.Tinf_ext, self.pc.h_ext )
        # TM on interior 
        TM = eq(self.Ttx[i,self.M-1], self.Ttx[i, self.M-2], self.pc.Tinf_int, self.pc.h_int)

        return T0, TM

    def calc_interior_nodes(self, i):
        Tint =  np.zeros(self.M)
        row = self.Ttx[i,:] 
        for m in range(self.M):
            m = m + 1 # avoid first and last nodes 
            if m < len(self.x_vals) - 1:
                Tint[m] = self.tau*(row[m - 1] + row[m + 1]) + (1 - 2*self.tau)*row[m]
        
        assert Tint[0] == 0 and Tint[self.M-1] == 0 

        return Tint 
    
    def calc_Tx_at_t(self, i):
        if i < self.N - 1:
            self.Ttx[i+1, :] = self.calc_interior_nodes(i)
            self.Ttx[i+1, 0], self.Ttx[i+1, self.M-1] = self.calc_boundary_nodes(i)

            return self.Ttx[i+1, :]

    def calc_all(self):
        for i in range(self.N):
            self.calc_Tx_at_t(i)

        return self.Ttx