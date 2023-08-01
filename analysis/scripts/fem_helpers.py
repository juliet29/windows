from shapely import *
import sympy as smp
from enum import Enum

class DE9IMPattern(Enum):
    # used shapely relate functions to ID these, and plug into obj.relate_pattern()
    # adjacency means next to each other, not diagonal from each other 
    CELL_ADJ = "FF2F11212"
    LINE_CELL_ADJ = "F1FF0F212"
    BC_LINE_CELL_ADJ = "F1FF0FFF2"

class Position(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4

class BoundaryCondition(Enum):
    CONVECTION = 1
    ADIABATIC = 2
    FIXED_TEMP = 3
    CONDUCTION = 4

class LineData: # more general than quantities => has geometrical and quantity info 
    dirxy = "x"
    rel_len = 1
    Tneighbour = smp.symbols("T_neighbour")

class CellQuantities:
    Tself = smp.symbols("T_self")






def box_from_centroid(centroid, dx, dy):
    # centroid should be a tuple
    # create a box based on a centroid with coordinates going in ccw direction (starting from top-left => v1, v2, v3, v4 <= top right)
    cx, cy = centroid

    v1x = v2x = cx - dx/2
    v4x = v3x = cx + dx/2

    v1y = v4y = cy - dy/2
    v2y = v3y = cy + dy/2

    coords = (
        (v1x, v1y),
        (v2x, v2y),
        (v3x, v3y),
        (v4x, v4y)
    )

    # return coords

    return Polygon(coords)

def get_flow_direction(line):
    # TODO assert that this is a shaperly line 
    t = line.xy
    dy = t[1][1] - t[1][0] # y dir if rise != 0
    dx = t[0][1] - t[0][0] # x 

    if dy==0:
        dir="x"
    else:
        dir="y"

    return dir

def get_rel_length(line, typical_length):
    # TODO assert that this is a shaperly line 
    return line.length/typical_length

def get_polygon_boundary_lines(polygon):
    # TODO somehow check that this is valid and assert intro value 
    vs = list(polygon.exterior.coords)

    # create lines based on vertices 
    lines = {}
    for ix, _ in enumerate(vs):
        if ix < len(vs) - 1:
            lines[ix] = LineString((vs[ix], vs[ix+1]))

    return lines
