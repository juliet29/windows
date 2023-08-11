import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "../scripts")
import helpers as h

from icecream import ic

C_TO_KELVIN = 273.15


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
    def __init__(self, pc:FabricPhysicalConstants, times, T0_int):
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
        self.N = len(times)

        # establish adjacent temps  
        self.ext_temps = self.get_ext_temps()
        self.int_temps = np.zeros(self.N)
        self.int_temps[0] = T0_int

        # initialize N*M matrix - time * x nodes 
        self.Ttx = np.zeros((self.N, self.M))
        self.Ttx = self.define_init_temps()

        # time constant
        # # TODO check tau  
        self.tau = self.pc.alpha * self.dt / (self.dx**2)


    def get_ext_temps(self):
        b00, b01 = h.import_desired_data("B", "15T")
        # ensure windows are always closed in this dataset 
        assert(b01["Window Open"].unique() == [0]) 

        # only using one day for now 
        mask = (b01['DateTime'] <= pd.Timedelta(1, "d") + b01["DateTime"].iloc[0]) 
        b01_day = b01.loc[mask].reset_index(drop=True)

        # #TODO fix interpolation issue 
        # resample (15 min data, to be 15s) 
        sample_time = "15s"
        # assert pd.Timedelta(sample_time).seconds == self.dt #TODO
        b01_dt = b01_day.set_index(b01_day["DateTime"].values)
        b01_dt = b01_dt.resample(sample_time).ffill()
        # return temps that are appropriate for solution 
        return b01_dt["Ambient Temp"][0: self.N] + C_TO_KELVIN



    def define_init_temps(self):
        # interpolate between outdoor and indoor temps
        T0_ext  = self.ext_temps[0]
        T0_int = self.int_temps[0]
        m = (T0_int - T0_ext)/(self.x_vals[-1] - self.x_vals[0])
        self.T_inits = m*(self.x_vals) + T0_ext

        # set init values in the N*M matrix 
        self.Ttx[0,:] = self.T_inits

        return self.Ttx
    
        

    def calc_interior_nodes(self, i):
        Tint =  np.zeros(self.M)
        row = self.Ttx[i,:] 
        for m in range(self.M):
            m = m + 1 # avoid first and last nodes 
            if m < len(self.x_vals) - 1:
                Tint[m] = self.tau*(row[m - 1] + row[m + 1]) + (1 - 2*self.tau)*row[m]

        # only change interior nodes 
        assert Tint[0] == 0 and Tint[self.M-1] == 0 

        return Tint 


    def calc_boundary_nodes(self, i):
        # TODO replace i with a class attribute instead of passing it in always
        beta = lambda h: h*self.dx/self.pc.k
        eq = lambda Tself, Tnb, Tinf, h: (1 -2*self.tau - 2*self.tau*beta(h))*Tself + 2*self.tau*Tnb + 2*self.tau*beta(h)*Tinf
        
        # T0 at exterior
        T0 = eq(self.Ttx[i,0], self.Ttx[i, 1], self.ext_temps[i], self.pc.h_ext)
        # TM on interior 
        TM = eq(self.Ttx[i,self.M-1], self.Ttx[i, self.M-2], self.int_temps[i], self.pc.h_int)

        return T0, TM
    
    def calc_Tx_at_t(self, i, Tavg_int):
        # update record of indoor temps 
        self.int_temps[i] = Tavg_int

        # calculate 
        if i < self.N - 1:
            self.Ttx[i+1, :] = self.calc_interior_nodes(i)
            self.Ttx[i+1, 0], self.Ttx[i+1, self.M-1] = self.calc_boundary_nodes(i)

            return self.Ttx[i+1, :]

    def calc_all(self):
        for i in range(self.N):
            self.calc_Tx_at_t(i)

        return self.Ttx