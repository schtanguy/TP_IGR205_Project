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

int main(int argc, char* argv[]) {
    
    if(argc != 5){
        std::cout << "You must set 4 arguments : input file, output file, block size, number of steps" << std::endl;
        return 1;
    }

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

                bool is_dark = random > (255.0 / static_cast<float>(global_average_intensity)) * average_intensity;

                double mid = static_cast<float>(block_size - 1)/2.0 ;

                double radius = mid*0.6;

                if(is_dark){
                    int random_x = dis(gen);
                    int random_y = dis(gen);

                    
                    for (int j = 0; j < block_size && y + j + random_y < height; ++j) {
                        for (int i = 0; i < block_size && x + i + random_x < width; ++i) {

                            int pixel_index = ((y + j + random_y) * width + (x + i + random_x)) * channels;

                            if (square(i-mid) + square(j-mid) < square(radius)){
                                for (int c = 0; c < channels; ++c) {
                                modified_image_data[pixel_index + c] = 0;
                            }
                            } 

                            
                        }
                    }
                    
                }
            }
        }
    }

    // Write the processed image to a new file as JPEG
    stbi_write_jpg(argv[2], width, height, channels, modified_image_data, 100); // Quality: 100

    // Free the image data
    stbi_image_free(image_data);

    return 0;
}
