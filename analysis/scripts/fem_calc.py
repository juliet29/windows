from fem_geom import *
from fem_term import * 
import sympy as smp
from icecream import ic

import fem_helpers as fh


class FEM_Calc(FEM_Geom):
    def __init__(self):
        super().__init__() # initialize the parent class 
        self.create_setup()

        self.cells_temp = {ix: k for ix, k in enumerate(smp.symbols(f"T0:{len(self.cells.values())}"))}

        self.T_infinity, self.h, self.k, self.edot, self.delta_x, self.delta_y = smp.symbols("T_ininity, h, k, edot, delta_x, delta_y") # TODO can remove some of these 
        
        # self.terms = []

    def create_cell_eq(self, cell_num=0):
        # specific cell and its vertices 
        cell = self.cells[cell_num]
        lines = fh.get_polygon_boundary_lines(cell)
        # ic(lines)

        cell_quantities = fh.CellQuantities()
        cell_quantities.Tself = self.cells_temp[cell_num]

        terms = []
        cell_data = {}
        for line_num, L in lines.items():
            line_data = fh.LineData()
            
            # get info about this specific line 
            line_data.dirxy = fh.get_flow_direction(L)
            line_data.rel_len = fh.get_rel_length(L, self.dx) if line_data.dirxy == "x" else fh.get_rel_length(L, self.dy)

            # update info about the cell's geometry 
            cell_data[line_data.dirxy] = line_data.rel_len


            # only cycle over boundary conditions if cell is missing neighbours
            if len(self.cells_nb[cell_num]) < len(lines.items()):
                for bc_name, bc_cell in self.bc_data.items():

                    # # ic(line_num, line_data.dirxy, bc_name, L.relate(bc_cell["poly"]))
                    if L.relate_pattern(bc_cell["poly"], fh.DE9IMPattern.BC_LINE_CELL_ADJ.value):
                        # ic("we in ")
                        term = FEM_Term(bc_cell["condition"], line_data, cell_quantities).create_term()
                        terms.append(term)

            # all cells have interior neighbours 
            for ixn in self.cells_nb[cell_num]: 
                if L.relate_pattern(self.cells[ixn], fh.DE9IMPattern.LINE_CELL_ADJ.value):
                    line_data.Tneighbour = self.cells_temp[ixn]
                    # ic(line_data.Tneighbour, self.cells_temp[ixn] )

                    term = FEM_Term(fh.BoundaryCondition.CONDUCTION.name, line_data, cell_quantities).create_term()
                    terms.append(term)

        # heat generation term - at the cell level, not the line level
        term = self.edot * cell_data["x"] * self.delta_x * cell_data["y"] * self.delta_y  # TODO should move to fem_term and combinge cell_quantities to just be cell_data 
        terms.append(term)

        terms = terms 
                        
        # should have 5 terms total 
        assert len(terms) == 5

        self.eqn = smp.Eq(0, sum(terms))

        return self.eqn