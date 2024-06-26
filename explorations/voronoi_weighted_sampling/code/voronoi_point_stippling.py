import tqdm
import voronoi
import os.path
import scipy.ndimage
import numpy as np
import imageio
import cv2 as cv
from scipy import interpolate
#import ipdb
import argparse
import matplotlib.pyplot as plt

def normalize(data):
    Vmin, Vmax = data.min(), data.max()
    if Vmax - Vmin > 1e-5:
        data = (data-Vmin)/(Vmax-Vmin)
    else:
        data = np.zeros_like(data)
    return data


def initialization(n, data):
    points = []
    while len(points) < n:
        X = np.random.uniform(0, data.shape[1], 10*n)
        Y = np.random.uniform(0, data.shape[0], 10*n)
        P = np.random.uniform(0, 1, 10*n)
        idx = 0
        while idx < len(X) and len(points) < n:
            x, y = X[idx], Y[idx]
            x_, y_ = int(np.floor(x)), int(np.floor(y))
            if P[idx] < data[y_, x_]:
                points.append([x, y])
            idx += 1
    return np.array(points)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Weighted Vororonoi Stippler")
    parser.add_argument('filename', metavar='image filename', type=str,
                        help='density_map image filename ')
    parser.add_argument('--n_iter', metavar='n', type=int,
                        default=30,
                        help='Maximum number of iterations')
    parser.add_argument('--n_point', metavar='n', type=int,
                        default=40000,
                        help='Number of points')
    parser.add_argument('--pointsize', metavar='(min,max)', type=float,
                        nargs=2, default=(1.0, 1.0),
                        help='Point mix/max size')
    parser.add_argument('--figsize', metavar='w,h', type=int,
                        default=6,
                        help='Figure size')
    parser.add_argument('--threshold', metavar='n', type=int,
                        default=255,
                        help='Grey level threshold')
    parser.add_argument('--display', action='store_true',
                        default=False,
                        help='Display final result')
    args = parser.parse_args()

    filename = args.filename
    density_map = imageio.imread(filename, mode='L')

    # we set 500x500 pixels
    scaling = (args.n_point * 500) / (density_map.shape[0]*density_map.shape[1])
    scaling = int(round(np.sqrt(scaling)))
    density_map = scipy.ndimage.zoom(density_map, scaling, order=0)

    # Any color > threshold will be white
    density_map = np.minimum(density_map, args.threshold)

    density_map = 1.0 - normalize(density_map)
    density_map = density_map[::-1, :]
    density_P = density_map.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    #dirname = os.path.dirname(filename)
    basename = (os.path.basename(filename).split('.'))[0]
    #output_file = os.path.join(dirname, basename + "-stipple.png")
    output_file = basename + "-stipple.png"

    # Initialization
    points = initialization(args.n_point, density_map)
    print("Number of points:", args.n_point)
    print("Number of iterations:", args.n_iter)
    print("density_map file: %s (resized to %dx%d)" % (
          filename, density_map.shape[1], density_map.shape[0]))
    
    print("Output file: %s " % output_file)

        
    xmin, xmax = 0, density_map.shape[1]
    ymin, ymax = 0, density_map.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    ratio = (xmax-xmin)/(ymax-ymin)

    for i in tqdm.trange(args.n_iter):
        regions, points = voronoi.centroids(points, density_map, density_P, density_Q)
            
    # Display final result
    fig = plt.figure(figsize=(args.figsize, args.figsize/ratio),
                        facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim([xmin, xmax])
    ax.set_xticks([])
    ax.set_ylim([ymin, ymax])
    ax.set_yticks([])
    scatter = ax.scatter(points[:, 0], points[:, 1], s=1, 
                            facecolor="k", edgecolor="None")
    Pi = points.astype(int)
    X = np.maximum(np.minimum(Pi[:, 0], density_map.shape[1]-1), 0)
    Y = np.maximum(np.minimum(Pi[:, 1], density_map.shape[0]-1), 0)
    sizes = (args.pointsize[0] +
                (args.pointsize[1]-args.pointsize[0])*density_map[Y, X])
    scatter.set_offsets(points)
    scatter.set_sizes(sizes)

    # Save stipple points and tippled image
    plt.savefig(output_file)

    if args.display:
        plt.show()

