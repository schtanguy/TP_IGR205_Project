import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Some utils function I (Tanguy) coded when lost on the project, thought they could be useful at some point at least

class Interpolated2DVectorField:
    """Continuous 2D vector field interpolated from 2D matrix.

    Interpolated via Bilinear interpolation.
    
    Uses a continuous coordinate system (x, y) with:
    - Horizontal x axis, increasing to the right (≡ j in (i, j) arrays indexing)
    - Vertical y axis, increasing to the top (≡ -i in (i, j) arrays indexing)

    Parameters
    ----------
    magnitude : np.ndarray, shape (n_rows, n_cols)
        2D array of discrete vector field magnitudes
    direction : np.ndarray, shape (n_rows, n_cols)
        2D array of discrete vector field directions
    xdomain : 2-tuple, optional
        Domain of the x axis of the vector field
        Default value: (0, 1)
    ydomain : 2-tuple, optional
        Domain of the y axis of the vector field
        Default value: (0, 1)
    """

    def __init__(self, magnitude, direction, xdomain=(0,1), ydomain=(0,1), angle_unit='radian'):
        if not (isinstance(magnitude, np.ndarray) and isinstance(direction, np.ndarray)):
            raise ValueError("Expected two numpy arrays for magnitude and direction")
        if not magnitude.ndim == direction.ndim == 2:
            raise ValueError("Input magnitude and direction arrays should be of dimension 2")
        if magnitude.shape != direction.shape:
            raise ValueError("Input magnitude and direction arrays should be of same shape")
        if angle_unit not in ['radian', 'degree']:
            raise ValueError("Angle unit can either be radian or degree")
            
        self.magnitude = np.array(magnitude, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.xmin, self.xmax = float(xdomain[0]), float(xdomain[1])
        self.ymin, self.ymax = float(ydomain[0]), float(ydomain[1])
        self.imax = magnitude.shape[0] - 1
        self.jmax = magnitude.shape[1] - 1
        self.angle_unit = angle_unit

    # Helper functions: (i,j) indexing in arrays (v down, h right) mapped to common CCS (x,y) (h right, v up)
    def _j_to_x(self, j):
        return float(j) / self.jmax * (self.xmax - self.xmin) + self.xmin
    def _i_to_y(self, i):
        return (1 - float(i) / self.imax) * (self.ymax - self.ymin) + self.ymin
    def _x_to_j(self, x):
        return (x - self.xmin) / (self.xmax - self.xmin) * self.jmax
    def _y_to_i(self, y):
        return (1 - (y - self.ymin) / (self.ymax - self.ymin)) * self.imax
        
    def get(self, x, y):
        """Get interpolated magnitude and direction at coordinates x, y.
        
        Parameters
        ----------
        x : float, x ∈ xdomain
        y : float, y ∈ ydomain

        Returns
        ----------
        2-tuple (magnitude, direction) of floats, corresponding to the interpolated vector at (x, y) in the vector field
            """
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            raise ValueError("Input coordinates out of domain: [{}, {}]×[{}, {}]".format(self.xmin, self.xmax, self.ymin, self.ymax))
        i = self._y_to_i(float(y))
        j = self._x_to_j(float(x))
        if abs(i - round(i)) < 1e-5 and abs(j - round(j)) < 1e-5:
            # (i, j) falls on a node of the regular grid
            return (self.magnitude[round(i), round(j)], self.direction[round(i), round(j)])
        elif abs(i - round(i)) < 1e-5 and abs(j - round(j)) >= 1e-5:
            # (i, j) falls on an edge (via i) of the regular grid
            alpha = j - int(j)
            magnitude = alpha * self.magnitude[round(i), int(j) + 1] + (1 - alpha) * self.magnitude[round(i), int(j)]
            direction = alpha * self.direction[round(i), int(j) + 1] + (1 - alpha) * self.direction[round(i), int(j)]
            return (magnitude, direction)
        elif abs(i - round(i)) >= 1e-5 and abs(j - round(j)) < 1e-5:
            # (i, j) falls on an edge (via j) of the regular grid
            alpha = i - int(i)
            magnitude = alpha * self.magnitude[int(i) + 1, round(j)] + (1 - alpha) * self.magnitude[int(i), round(j)]
            direction = alpha * self.direction[int(i) + 1, round(j)] + (1 - alpha) * self.direction[int(i), round(j)]
            return (magnitude, direction)
        else:
            # (i, j) falls in a cell of the regular grid
            alpha = i - int(i)
            beta = j - int(j)
            i_m, i_M = int(i), int(i) + 1
            j_m, j_M = int(j), int(j) + 1
            mag_i_M = beta * self.magnitude[i_M, j_M] + (1 - beta) * self.magnitude[i_M, j_m]
            mag_i_m = beta * self.magnitude[i_m, j_M] + (1 - beta) * self.magnitude[i_m, j_m]
            dir_i_M = beta * self.direction[i_M, j_M] + (1 - beta) * self.direction[i_M, j_m]
            dir_i_m = beta * self.direction[i_m, j_M] + (1 - beta) * self.direction[i_m, j_m]
            magnitude = alpha * mag_i_M + (1 - alpha) * mag_i_m
            direction = alpha * dir_i_M + (1 - alpha) * dir_i_m
            return (magnitude, direction)

    def toSVG(self, padding=0.1, length_multiplier=1):
        # Forget this, it's giga slow and BAD whenever the matrices are slightly large
        min_x = self.xmin - (self.xmax - self.xmin) * padding
        min_y = self.ymin - (self.ymax - self.ymin) * padding
        width = (self.xmax - self.xmin) * (1 + padding * 2)
        height = (self.ymax - self.ymin) * (1 + padding * 2)

        stroke_width = min(width, height) * 0.002
        marker_size = stroke_width * 10000
        
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}" version="1.1"> \n""".format(min_x, min_y, width, height)
        svg += """<defs> \n <marker id="arrow" markerWidth="{}" markerHeight="{}" refX="3" refY="2" orient="auto"> \n
                  <path d="M 0 0 L 0 4 L 6 2 z"/> \n </marker> \n </defs> \n""".format(6, 6)

        for i in range(self.imax + 1):
            for j in range(self.jmax + 1):
                if (i * (self.jmax + 1) + j) % 10000 == 0:
                    print("Step:",i * (self.jmax + 1) + j + 1, "/", (self.imax+1)*(self.jmax+1))
                start_x = self._j_to_x(j)
                start_y = self._i_to_y(i)
                angle = self.direction[i, j] if self.angle_unit == 'radian' else self.direction[i, j] * np.pi / 180
                end_x = start_x + np.cos(angle) * self.magnitude[i, j] * length_multiplier
                end_y = start_y + np.sin(angle) * self.magnitude[i, j] * length_multiplier
                svg += """<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="{}"
                          marker-end="url(#arrow)"/> \n""".format(start_x, start_y, end_x, end_y, stroke_width)

        svg += """</svg> \n"""
        print("Done")
        return svg

    def show(self, angle_unit='radian'):
        """Display the magnitudes and directions as heatmaps."""
        # Redo some details

        if angle_unit not in ['radian', 'degree']:
            raise ValueError("angle_unit must be either 'radian' or 'degree'")

        # Prepare the directions for display
        if angle_unit == 'radian':
            directions = np.mod(self.direction, np.pi)
        else:
            directions = np.mod(np.degrees(self.direction), 180)
        cmap = plt.cm.hsv # Circular colormap
        
        # Plot the magnitudes
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Magnitudes")
        plt.imshow(self.magnitude, extent=(self.xmin, self.xmax, self.ymin, self.ymax))
        plt.colorbar(label="Magnitude")

        # Plot the directions
        plt.subplot(1, 2, 2)
        plt.title("Directions ({})".format(angle_unit))
        plt.imshow(directions, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=cmap)
        plt.colorbar(label="Direction ({})".format(angle_unit))

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------------------------------------------------


class Generate:
    """Helper class to generate random line segments with various settings (static methods only)"""

    @staticmethod
    def CenterLengthAngleToEndPoints(n, positions, angles, lengths):
        segments = np.zeros((n, 2, 2))
        segments[:, 0, 0] = positions[:, 0] - np.cos(angles) * lengths / 2
        segments[:, 0, 1] = positions[:, 1] - np.sin(angles) * lengths / 2
        segments[:, 1, 0] = positions[:, 0] + np.cos(angles) * lengths / 2
        segments[:, 1, 1] = positions[:, 1] + np.sin(angles) * lengths / 2
        return segments

    @staticmethod
    def CLAtoEP(n, positions, angles, lengths):
        return Generate.CenterLengthAngleToEndPoints(n, positions, angles, lengths)

    @staticmethod
    def FixedLengthUniformAngles(n, length):
        positions = np.random.rand(n, 2)
        angles = np.random.rand(n) * np.pi
        lengths = np.array([length]).repeat(n)
        return Generate.CLAtoEP(n, positions, angles, lengths)

    @staticmethod
    def FixedLengthGaussianAngles(n, length, angle_mean, angle_std):
        positions = np.random.rand(n, 2)
        angles = np.random.normal(loc=angle_mean, scale=angle_std, size=n) * np.pi
        lengths = np.array([length]).repeat(n)
        return Generate.CLAtoEP(n, positions, angles, lengths)

    @staticmethod
    def show(segments, linewidths=0.5):
        lineCollection = LineCollection(segments, linewidths=linewidths, colors='black', linestyle='solid')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_collection(lineCollection)
        plt.show()