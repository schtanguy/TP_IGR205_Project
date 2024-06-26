import numpy as np
from scipy import interpolate
#import ipdb

class Point:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    # Check if the point overlaps another (sum of radius >= distance of centers)
    def overlap(self, point):
        distance = np.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)
        return self.radius + point.radius >= distance



class Cell:
    # store points within the cell
    def __init__(self):
        self.nb_points = 0
        self.list_points = []

    def get_point_list(self):
        return self.list_points

    def append(self, point):
        self.list_points.append(point)
        self.nb_points += 1



class Grid:   
    def __init__(self, N, density_map, radius=0.003):
        self.nb_cell = N
        # Nested list: NxN matrix of cell objects
        self.list_cell = [[Cell() for _ in range(self.nb_cell)] for _ in range(self.nb_cell)]
        # Cells virtual width
        self.radius = np.mean(density_map)*radius
        self.width = self.radius/np.min(density_map)
        self.scaling = N*self.width
        # self.width = 1/N
        # Interpolate the given density over a [0,1]x[0,1] domain
        x = np.arange(0, self.nb_cell, 1)
        y = np.arange(0, self.nb_cell, 1)
        self.density_map = density_map
        # ipdb.set_trace()
        self.density = interpolate.interp2d(x, y, density_map, kind='linear')

    def return_coordinate(self, point):
        return int(point.x/self.width), int(point.y/self.width)

    # Return list of points from neighboring cells of the current point
    def get_point_list_neighbour(self, pt):
        i,j = self.return_coordinate(pt)
        cell_num = self.nb_cell-1
        neighbour_cell = []
        neighbour_point = []
        neighbour_cell.append(self.list_cell[i][j])

        # add neibouring cells
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni <= cell_num and 0 <= nj <= cell_num:
                neighbour_cell.append(self.list_cell[ni][nj])

        for cell in neighbour_cell:
            for point in cell.list_points:
                neighbour_point.append(point)
        return neighbour_point
    
    # Draw a new point and add it to the current grid if it's valid
    def new_point(self):
        is_not_valid = False
        x,y = np.random.uniform(0,1), np.random.uniform(0,1)
        # create a new point in a cell with radius radius/density of the cell
        x_grid = x*self.nb_cell
        y_grid = y*self.nb_cell
        p = Point(x, y, self.radius / self.density(x_grid,y_grid))
        neighbour_point = self.get_point_list_neighbour(p)
        for point in neighbour_point:
            # check if the new point overlaps with one of the list
            if p.overlap(point) :
                is_not_valid = True
                break
        if not is_not_valid:
            i,j = self.return_coordinate(p)
            cell = self.list_cell[i][j]
            cell.append(p)
        return is_not_valid, p
