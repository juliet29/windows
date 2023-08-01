from fem_geom import *
import sympy as smp
from fem_calc import *

import fem_helpers as fh
from icecream import ic


class FEM_Term():
    def __init__(self, condition, line_data: fh.LineData, cell_quantities: fh.CellQuantities):
        self.condition = condition 
        self.line_data = line_data
        self.cell_quantities = cell_quantities
        self.T_infinity, self.h, self.k, self.edot, self.delta_x, self.delta_y = smp.symbols("T_ininity, h, k, edot, delta_x, delta_y")
        pass


    def create_convection_term(self):
        term =  self.h * (self.T_infinity - self.cell_quantities.Tself)  * self.line_data.rel_len

        if self.line_data.dirxy == "x": 
            term = term * self.delta_x
        else:
            term = term * self.delta_y

        return term

    def create_adiabatic_term(self):
        return smp.symbols("Q_adiabatic")

    def create_conduction_term(self):
        dxdy = self.delta_x/self.delta_y if self.line_data.dirxy == "x" else self.delta_y/self.delta_x


        term = self.k * self.line_data.rel_len * dxdy * (self.line_data.Tneighbour - self.cell_quantities.Tself)

        return term 
        



    def create_term(self):
        term_creation = {
            "CONVECTION": self.create_convection_term,
            "ADIABATIC": self.create_adiabatic_term,
            "CONDUCTION": self.create_conduction_term
        }

        term = term_creation[self.condition]()

        return term
