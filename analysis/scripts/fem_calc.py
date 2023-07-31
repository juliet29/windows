from fem import *
import sympy as smp

import fem_helpers as fh


class FEM_Calc(FEM_Geom):
    def __init__(self):
        self.cells_temp = {ix: k for ix, k in enumerate(smp.symbols(f"T0:{len(self.cells.values())}"))}

        self.T_infinity, self.h, self.k, self.edot, self.delta_x, self.delta_y = smp.symbols("T_ininity, h, k, edot, delta_x, delta_y")



    # TODO this is now the general procedure => just need to split it up for different types of BCs
    def create_cell_eq(self, cell_num=0):
        # specific cell and its vertices 
        cell = self.cells[cell_num]
        lines = fh.get_polygon_boundary_lines(cell)
        Tself = self.cells_temp[cell_num]

        terms = []
        cell_data = {}
        for line_num,L in lines.items():
            
            # get info about this specific line 
            dirxy = get_flow_direction(L)
            rel_len = get_rel_length(L, self.dx) if dirxy == "x" else get_rel_length(L, self.gen_dy)
            cell_data[dirxy] = rel_len

            # convection 
            if L.relate_pattern(amb_poly_diff, amb_touch_pattern):
                term =  h * (T_infinity - Tself)  * rel_len
                if dirxy == "x": 
                    term = term * delta_x
                else:
                    term = term * delta_y
                
                terms.append(term)
            
            # conduction 
            else:
                dxdy = delta_x/delta_y if dirxy == "x" else delta_y/delta_x
                # have to cycle though all neigbours..
                for ixn in self.cells_nb[cell_num]: 
                    if L.relate_pattern(self.cells[ixn], line_nb_touch_pattern):
                    # ic(line_num, ixn) # this should only occur once per line number # TODO enforce a check here 
                        Tnb = self.cells_temp[ixn]

                        term = k * rel_len * dxdy * (Tnb - Tself)

                        terms.append(term)
        
        # heat generation term 
        term = edot * cell_data["x"] * delta_x * cell_data["y"] * delta_y 
        terms.append(term)
                        
        # should have 5 terms total 
        assert len(terms) == 5

        eqn = smp.Eq(0, sum(terms))

        return eqn