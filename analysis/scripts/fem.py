

from shapely import *
import shapely.plotting as splt
import shapely.ops as sopt
import shapely as shp

import matplotlib.pyplot as plt
import numpy as np

from icecream import ic

import fem_helpers as fh

# TODO rename file to match class

class FEM_Geom:
    def __init__(self):
        self.room_lx = 20 # m - length in x
        self.room_ly = 10 # m 
        self.num_points = 5

        # assigned later 
        self.room_pts = None # TODO make an empty 2d array
        self.room_poly = None # TODO empty polygon 
        self.cells_untrimmed = []
        self.cells = {} # list of cell polygons 
        self.cells_nb = {} # dict of lists containing index of neighbours in self.cells
        self.dx = 0
        self.dy = 0
        self.bc_data = {}


    def create_room(self):
        # self.room_lx = 20 # m - length in x
        # self.room_ly= 10 # m 
        self.room_poly = box(0.0, 0.0, self.room_lx, self.room_ly)

    def create_cell_pts(self):
        # use numpy to make a mesh of points that are evenly distributed 
        room_xy = self.room_poly.exterior.coords.xy

        x = np.linspace(np.min(room_xy[0]), np.max(room_xy[0]), self.num_points)
        y = np.linspace(np.min(room_xy[1]), np.max(room_xy[1]), self.num_points)
        xv, yv = np.meshgrid(x,y)

        self.room_pts = [(x,y) for x,y in zip(xv.flatten(), yv.flatten())]

    def create_cells(self):
        # create cells around points in mesh of points 
        self.dx = self.room_lx/(self.num_points - 1)
        self.dy = self.room_ly/(self.num_points - 1)
        self.cells_untrimmed = [fh.box_from_centroid(pt, dx, dy) for pt in self.room_pts]

        # all cells should have the same area before they are trimmed
        assert len(np.unique(np.array([cell.area for cell in cells_untrimmed]))) == 1

        # edit cell geometries based on overlap with room geom 
        self.cells = {}
        for ix, cell in enumerate(self.cells_untrimmed):
            if not cell.within(room_poly):
                self.cells[ix] = cell.intersection(self.room_poly)
            else:
                self.cells[ix] = cell

    def find_neigbours(self):
        tree = STRtree(list(self.cells.values()))

        self.cells_nb = {} # nb - neighbours

        for k,v in self.cells.items(): 
            self.cells_nb[k] = []
            near_cells_ix = tree.query(v).tolist() 

            for ix in near_cells_ix:
                if v.relate_pattern(self.cells[ix], fh.DE9IMPattern.CELL_ADJ):
                    self.cells_nb[k].append(ix)

    
    def create_boundary_cells(self):
        for pos in fh.Position:
            p = pos.name
            self.bc_data[p] = {}
            self.bc_data[p]["poly"] = generate_BC_geom(p, room_poly=room_poly, dx=dx, dy=dy)

            # fixed rn, need to be able to make arbitrary..
            if p == Position.LEFT.name:
                self.bc_data[p]["condition"] = fh.BoundaryCondition.CONVECTION.name
            else:
                self.bc_data[p]["condition"] = fh.BoundaryCondition.ADIABATIC.name 




    


    

