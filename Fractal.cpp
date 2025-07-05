#include "Fractal.hpp"
#include <complex>

Fractal::Fractal(int width, int height, int maxIterations)
    : width(width), height(height), maxIterations(maxIterations), image(width* height * 3, 0)
{
}

void Fractal::generate() {
    // Define the complex plane range
    double xMin = -2.0, xMax = 1.0;
    double yMin = -1.5, yMax = 1.5;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Map pixel to complex plane
            double x0 = xMin + (xMax - xMin) * i / width;
            double y0 = yMin + (yMax - yMin) * j / height;

            int iterations = computeIterations(x0, y0);

            unsigned char r, g, b;
            mapColor(iterations, maxIterations, r, g, b);

            int idx = (j * width + i) * 3;
            image[idx + 0] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
    }
}

int Fractal::computeIterations(double x0, double y0) {
    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < maxIterations) {
        double xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        ++iter;
    }
    return iter;
}

void Fractal::mapColor(int iterations, int maxIterations, unsigned char& r, unsigned char& g, unsigned char& b) {
    if (iterations == maxIterations) {
        r = g = b = 0; // black for points inside the set
    }
    else {
        // Simple coloring: map iterations to a gradient
        double t = (double)iterations / maxIterations;
        r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
        g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
        b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    }
}

const std::vector<unsigned char>& Fractal::getImage() const {
    return image;
}
