#pragma once

#include <vector>

class Fractal {
public:
    Fractal(int width, int height, int maxIterations);
    void generate();
    const std::vector<unsigned char>& getImage() const;

private:
    int width;
    int height;
    int maxIterations;
    std::vector<unsigned char> image; // RGB pixels

    int computeIterations(double x0, double y0);
    void mapColor(int iterations, int maxIterations, unsigned char& r, unsigned char& g, unsigned char& b);
};
