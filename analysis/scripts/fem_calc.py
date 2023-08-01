from fem import *
from fem_term import * 
import sympy as smp

import fem_helpers as fh


class FEM_Calc(FEM_Geom):
    def __init__(self):
        self.cells_temp = {ix: k for ix, k in enumerate(smp.symbols(f"T0:{len(self.cells.values())}"))}

        self.T_infinity, self.h, self.k, self.edot, self.delta_x, self.delta_y = smp.symbols("T_ininity, h, k, edot, delta_x, delta_y")

    def create_cell_eq(self, cell_num=0):
        # specific cell and its vertices 
        cell = self.cells[cell_num]
        lines = fh.get_polygon_boundary_lines(cell)

        cell_quantities = fh.CellQuantities()
        cell_quantities.Tself = self.cells_temp[cell_num]

        terms = []
        cell_data = {}
        for line_num, L in lines.items():
            line_data = fh.LineData()
            
            # get info about this specific line 
            line_data.dirxy = get_flow_direction(L)
            line_data.rel_len = get_rel_length(L, self.dx) if dirxy == "x" else get_rel_length(L, self.gen_dy)

            # update info about the cell's geometry 
            cell_data[dirxy] = rel_len


            # only cycle over boundary conditions if cell is missing neighbours
            if self.cells_nb[cell_num] < len(lines.items):
                for bc_name, bc_cell in self.bc_data.items():
                    if L.relate_pattern(bc_cell["poly"], fh.DE9IMPattern.LINE_CELL_ADJ):
                        term = FEM_Term(bc_cell["condition"], line_data, cell_quantities).create_BC_term()
                        terms.append(term)

            # all cells have interior neighbours 
            for ixn in self.cells_nb[cell_num]: 
                if L.relate_pattern(self.cells[ixn], fh.DE9IMPattern.LINE_CELL_ADJ):
                    line_data.Tneighbour = self.cells_temp[ixn]
                    term = FEM_Term(bc_cell["condition"], line_data, cell_quantities).create_BC_term()
                    terms.append(term)

        # heat generation term - at the cell level, not the line level
        term = edot * cell_data["x"] * delta_x * cell_data["y"] * delta_y 
        terms.append(term)
                        
        # should have 5 terms total 
        assert len(terms) == 5

        eqn = smp.Eq(0, sum(terms))

        return eqn