#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "Fractal.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    int width = 2000;
    int height = 2000;
    int maxIterations = 500;

    Fractal fractal(width, height, maxIterations);

    std::cout << "Generating fractal..." << std::endl;
    fractal.generate();

    const auto& image = fractal.getImage();

    if (stbi_write_png("mandelbrot.png", width, height, 3, image.data(), width * 3)) {
        std::cout << "Fractal image saved as mandelbrot.png" << std::endl;
    }
    else {
        std::cout << "Failed to write image." << std::endl;
    }

    return 0;
}
