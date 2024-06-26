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
    parser.add_argument('--length', type=float,
                    default=0.005,
                    help='length of the line segment')
    parser.add_argument('--linewidth', type=float,
                    default=0.3,
                    help='width of the line segment')
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

    #dirname = os.path.dirname(filename)+"/../result"
    basename = (os.path.basename(filename).split('.'))[0]
    #output_file = os.path.join(dirname, basename + f"-line{args.n_point}.png")
    output_file = basename + f"-line{args.n_point}.png"


    # Initialization
    points = initialization(args.n_point, density_map)
    print("Number of points:", args.n_point)
    print("Number of iterations:", args.n_iter)
    print("density_map file: %s (resized to %dx%data)" % (
          filename, density_map.shape[1], density_map.shape[0]))
    
    print("Output file: %s " % output_file)

        
    xmin, xmax = 0, density_map.shape[1]
    ymin, ymax = 0, density_map.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    ratio = (xmax-xmin)/(ymax-ymin)

    for i in tqdm.trange(args.n_iter):
        regions, points = voronoi.centroids(points, density_map, density_P, density_Q)

    depth = cv.CV_16S
    grad_x = cv.Sobel(src=density_map, ddepth=depth, dx=1, dy=0, ksize=5)
    grad_y = cv.Sobel(src=density_map, ddepth=depth, dx=0, dy=1, ksize=5)

    # interpolate the gradient
    N = 500
    grad_x = cv.resize(grad_x, (N, N)).clip(0.1, 100)
    grad_y = cv.resize(grad_y, (N, N)).clip(0.1, 100)
    # maximum gradient value
    grad_ = np.sqrt(grad_x**2 + grad_y**2)
    grad_normalize_x = grad_x/grad_
    grad_normalize_y = grad_y/grad_
    
    x = np.arange(0, N, 1)
    y = np.arange(0, N, 1)
    grad_x_interpolate = interpolate.interp2d(x, y, grad_normalize_x, kind='linear')
    grad_y_interpolate = interpolate.interp2d(x, y, grad_normalize_y, kind='linear')
    length = args.length
    
    x_coord_point = points[:, 0]/np.max(points[:, 0])
    y_coord_point = points[:, 1]/np.max(points[:, 1])

    plt.figure(figsize=(8,8))
    for i in tqdm.tqdm(range(len(x_coord_point))):
        x_0 = x_coord_point[i]
        y_0 = y_coord_point[i]
        grad_x_0 = grad_x_interpolate(x_0*N, y_0*N)[0]
        grad_y_0 = grad_y_interpolate(x_0*N, y_0*N)[0]
        # x_1 = x_0 - length*grad_x_0
        # y_1 = y_0 - length*grad_y_0
        # x_2 = x_0 + length*grad_x_0
        # y_2 = y_0 + length*grad_y_0

        x_1 = x_0 + length*grad_y_0
        y_1 = y_0 - length*grad_x_0
        x_2 = x_0 - length*grad_y_0
        y_2 = y_0 + length*grad_x_0

        plt.plot([x_1, x_2], [y_1, y_2], 'k-', linewidth=args.linewidth)
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    plt.axis("off")
    plt.title("Line segment of %s points" %args.n_point)
    plt.savefig(output_file, dpi=300)

