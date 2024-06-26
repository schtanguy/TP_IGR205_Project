import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from poisson_disk_sampling import Grid
import argparse

if __name__ == '__main__':
    description = "poisson disk Stippler"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('filename', metavar='image filename', type=str,
                        help='Density image filename ')
    parser.add_argument('--display', action='store_true',
                    default=False,
                    help='Display final result')
    parser.add_argument('--output_name', type=str,
                    default="ouput",
                    help='output file name (without extension)')
    parser.add_argument('--npoints', type=int,
                    default=40000,
                    help='number of points')
        
    parser.add_argument('--pointsize', type=float,
                    default=0.4,
                    help='size of the points')
    parser.add_argument('--grid_size', type=int,
                    default=800,
                    help='size of the grid')
    parser.add_argument('--radius', type=float,
                    default=0.002,
                    help='radius constraint')
        
    args = parser.parse_args()
    filename = args.filename
    image = Image.open(filename).convert("L")

    N = args.grid_size # Size of the output image
    print("Resize the image to size %ix%i" %(N,N))
    image = image.resize((N, N))
    image = np.array(image, dtype=float)[-1::-1,:]
    density_map = np.max(image) - image

    # Apply minimum threshold to avoid zero density areas
    density_map = np.clip(density_map, 10, None)

    start = time.time()

    # Initialize vectors of points coordinates (x,y)
    x_coord_point = []
    y_coord_point = []

    # Constant
    max_nb_cfails = 5000
    max_nb_points = args.npoints
    radius = args.radius

    grid = Grid(args.grid_size, density_map, radius)

    num_failure = 0
    points_found  = 0
    while points_found < max_nb_points and num_failure < max_nb_cfails:
        is_not_valid, p = grid.new_point()
        
        # If the point is non-valid, count a failure and skip
        if is_not_valid:
            num_failure += 1
            
        # Else, the point is valid and is stored. We reset the num_failure
        else:
            num_failure = 0
            points_found += 1
            x_coord_point.append(p.x)
            y_coord_point.append(p.y)
            
            percent = int(max_nb_points / 10)
            if points_found % percent == 0:
                intermediate_time = time.time() - start
                formated_time = time.strftime("%H:%M:%S", time.gmtime(intermediate_time))
                progress = 100*points_found/max_nb_points
                print("Time: %s | Percentage: %2.2f %%" %(formated_time, progress))

    print(f"Finally we use {points_found} points.")

    plt.figure(figsize=(8,8))
    plt.scatter(x_coord_point, y_coord_point, s=args.pointsize, c="k", marker=".")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.axis("off")
    plt.title("Stippling of %s points" %max_nb_points)
    plt.savefig(args.output_name+".png", dpi=300)
    if args.display:
        plt.show()