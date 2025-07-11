#pragma once

#include <vector>

class Fractal {
public:
    Fractal(int width, int height, int maxIterations);
    Fractal(int width, int height, int maxIterations, double xLocation, double yLocation, double zoomFactor);
    void generate();
    const std::vector<unsigned char>& getImage() const;

private:
    int width;
    int height;
	double xLocation;
	double yLocation;
	double zoomFactor;
    int maxIterations;
    std::vector<unsigned char> image; // RGB pixels

    double computeIterations(double x0, double y0);
    void mapColor(double iterations, int maxIterations, unsigned char& r, unsigned char& g, unsigned char& b);
};
