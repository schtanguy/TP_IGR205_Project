#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <vector>
#include <iostream>
#include <random>

float square(float i){
    return i*i ;
}

void draw_line(unsigned char* & img, int width, int height, int x0, int y0, int x1, int y1, int channels) {
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    while (true) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            for (int c=0 ; c<channels ; c++){
                img[(y0 * width + x0) * channels + c] = 0;  // Draw black pixel
            }
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = err * 2;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

class Point {
public:
    int x;
    int y;

    // Constructor
    Point(int xCoord, int yCoord) : x(xCoord), y(yCoord) {}

    // Method to print the point
    void print() const {
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }

    // Example of added functionality: distance to another point
    double distanceTo(const Point& other) const {
        int dx = x - other.x;
        int dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

int main(int argc, char* argv[]) {
    
    if(argc != 7){
        std::cout << "You must set 6 arguments : input file, output file, block size, number of steps, min segment length, max segment length" << std::endl;
        return 1;
    }

    double min_dist = static_cast<double>(std::stoi(argv[5]));
    double max_dist = static_cast<double>(std::stoi(argv[6]));

    std::cout << "Min segment length : " << min_dist << std::endl;
    std::cout << "Max segment length : " << max_dist << std::endl;

    // Load an image
    int width, height, channels;
    unsigned char *image_data = stbi_load(argv[1], &width, &height, &channels, 0);

    // Check if image loading was successful
    if (!image_data) {
        std::cerr << "Error: Failed to load image\n";
        return 1;
    }

    // Print image dimensions and number of channels
    std::cout << "Image width: " << width << ", height: " << height << ", channels: " << channels << std::endl;

    // Initialize a new array that will be the new image
    unsigned char *modified_image_data = new unsigned char[width * height * channels];

    // Define block size
    int block_size = std::stoi(argv[3]);
    std::cout << "Block size : " << block_size << std::endl;
    int block_area = block_size * block_size;

    // Setting the random engine
    std::random_device rd;
    // Use the Mersenne Twister 19937 engine
    std::mt19937 gen(rd());
    // Uniform distribution = noise while putting points
    std::uniform_int_distribution<> dis(-(block_size/2),(block_size/2));
    // Uniform distribution to determine if a point is placed
    std::uniform_int_distribution<> random_255(0,255);

    // Completely white image, where we'll add dots
    // We compute the darkness_treshold at the same time
    float global_average_intensity = 0.0 ;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = ((y) * width + (x)) * channels;

                        for (int c = 0; c < channels; ++c) {
                            // Put a white pixel
                            modified_image_data[pixel_index + c] = 255;
                            // Compute the average brightness of the image
                            global_average_intensity += static_cast<float>(image_data[pixel_index + c])/(width*height*channels);
                        }

        }
    }

    // Define threshold for darkness
    //int darkness_threshold = std::stoi(argv[4]);
    std::cout << "Global average intensity : " << global_average_intensity << std::endl;    
    

    int nb_steps = std::stoi(argv[4]);

    std::cout << "Go for " << nb_steps << " steps !" << std::endl;

    // Declare a vector to store the points
    std::vector<Point> points;
    // Iterate through the image in block_size x block_size blocks
    for (int count=0; count<nb_steps; count++){
        for (int y = 0; y < height; y += block_size) {
            for (int x = 0; x < width; x += block_size) {
                // Compute average pixel intensity in the block
                int sum = 0;
                int pixel_count = 0;
                for (int j = 0; j < block_size && y + j < height; ++j) {
                    for (int i = 0; i < block_size && x + i < width; ++i) {
                        int pixel_index = ((y + j) * width + (x + i)) * channels;
                        for (int c = 0; c < channels; ++c) {
                            sum += image_data[pixel_index + c];
                        }
                        pixel_count++;
                    }
                }
                int average_intensity = sum / (pixel_count * channels);

                // Check if we put a point (according to a semi-random process : more dark = more chance)
                int random = random_255(gen);

                bool is_dark = random > (255.0 / static_cast<float>(global_average_intensity))*average_intensity;

                double mid = static_cast<float>(block_size - 1)/2.0 ;

                double radius = mid*0.6;

                if(is_dark){
                    int random_x = dis(gen);
                    int random_y = dis(gen);

                    points.push_back(Point(x+block_size/2+random_x,y+block_size/2+random_y));
                    
                }
            }
        }
    }
    while(points.size()>=1){
        Point p1 = points[points.size()-1];
        Point p2(0,0);
        bool exist = false;
        int min_index = 0;
        
        for (int index=0; index<points.size()-1; index++){
            Point p = points[index];
            if (p1.distanceTo(p) < max_dist && p1.distanceTo(p) > min_dist){
                exist = true;
                min_index = index;
                p2 = p;
                break;
            }
        }
        if(exist){
            draw_line(modified_image_data, width, height, p1.x, p1.y, p2.x, p2.y, 3);
            points.pop_back();
            points.erase(points.begin() + min_index);
        } else {
            points.pop_back();
        }
       
    }

    // Write the processed image to a new file as JPEG
    stbi_write_jpg(argv[2], width, height, channels, modified_image_data, 100); // Quality: 100

    // Free the image data
    stbi_image_free(image_data);

    return 0;
}
