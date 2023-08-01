from fem import *
import sympy as smp
from fem_calc import *

import fem_helpers as fh


class FEM_Term(FEM_Calc):
    def __init__(self, condition, line_data: fh.LineData, cell_quantities: fh.CellQuantities):
        self.condition = self.condition 
        self.line_data = line_data
        self.cell_quantities = cell_quantities
        pass


    def create_convection_term(self):
        term =  self.h * (self.T_infinity - self.cell_quantities.Tself)  * self.line_data.rel_len

        if self.line_data.dirxy == "x": 
            term = term * self.delta_x
        else:
            term = term * self.delta_y

        return term

    def create_adiabatic_term(self):
        return 1 

    def create_conduction_term(self):
        dxdy = delta_x/delta_y if self.line_data.dirxy == "x" else delta_y/delta_x

        term = self.k * self.line_data.rel_len * dxdy * (self.line_data.Tneighbour - self.cell_quantities.Tself)
        



    def create_term(self):
        term_creation = {
            "CONVECTION": self.create_convection_term,
            "ADIABATIC": self.create_adiabatic_term,
            "CONVECTION": self.create_convection_term
        }

        term = term_creation[self.condition]()

        return term
