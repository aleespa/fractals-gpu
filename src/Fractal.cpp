#include "../include/Fractal.hpp"
#include <complex>

Fractal::Fractal(int width, int height, int maxIterations)
    : width(width), height(height), maxIterations(maxIterations), image(width* height * 3, 0) 
{
	xLocation = -0.5;
	yLocation = 0.0;
	zoomFactor = 3;
}
Fractal::Fractal(int width, int height, int maxIterations, double xLocation, double yLocation, double zoomFactor)
	: width(width), height(height), maxIterations(maxIterations), xLocation(xLocation), yLocation(yLocation), zoomFactor(zoomFactor), image(width* height * 3, 0)
{
}

void Fractal::generate() {
    // Define the complex plane range
    double xMin = xLocation - zoomFactor / 2, xMax = xLocation + zoomFactor / 2;
    double yMin = yLocation - zoomFactor / 2, yMax = yLocation + zoomFactor / 2;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Map pixel to complex plane
            double x0 = xMin + (xMax - xMin) * i / width;
            double y0 = yMin + (yMax - yMin) * j / height;

            double iterations = computeIterations(x0, y0);

            unsigned char r, g, b;
            mapColor(iterations, maxIterations, r, g, b);

            int idx = (j * width + i) * 3;
            image[idx + 0] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
    }
}

double Fractal::computeIterations(double x0, double y0) {
    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < maxIterations) {
        double xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        ++iter;
    }

    if (iter < maxIterations) {
        double mag = sqrt(x * x + y * y);
        double nu = iter + 1 - log(log(mag)) / log(2.0);
        return nu;
    }

    return (double)iter;
}

void Fractal::mapColor(double iterations, int maxIterations, unsigned char& r, unsigned char& g, unsigned char& b) {
    if (iterations >= maxIterations) {
        r = g = b = 0; // black
    }
    else {
        double t = iterations / maxIterations;
        r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
        g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
        b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    }
}

const std::vector<unsigned char>& Fractal::getImage() const {
    return image;
}
