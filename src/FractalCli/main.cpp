#include "fractal_core.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  int type = 0; // 0 for mandelbrot, 1 for julia
  int width = 3840;
  int height = 2160;
  int maxIterations = 400;
  double xCenter = -1.4002, yCenter = 0.0;
  double zoom = 0.10;
  double cx = -0.4, cy = 0.6;
  std::string outputFilename = "mandelbrot.png";

  if (argc >= 2)
    type = std::atoi(argv[1]);
  if (argc >= 3)
    width = std::atoi(argv[2]);
  if (argc >= 4)
    height = std::atoi(argv[3]);
  if (argc >= 5)
    maxIterations = std::atoi(argv[4]);
  if (argc >= 6)
    xCenter = std::atof(argv[5]);
  if (argc >= 7)
    yCenter = std::atof(argv[6]);
  if (argc >= 8)
    zoom = std::atof(argv[7]);
  if (argc >= 9)
    cx = std::atof(argv[8]);
  if (argc >= 10)
    cy = std::atof(argv[9]);
  if (argc >= 11)
    outputFilename = argv[10];

  // Remaining args: optional "--ss N" (supersampling: N x N samples per
  // pixel) and palette stops as pairs of (t_float, hexcolor)
  int ss = 1;
  std::vector<std::string> palArgs;
  for (int a = 11; a < argc; ++a) {
    std::string arg = argv[a];
    if (arg == "--ss" && a + 1 < argc) {
      ss = std::atoi(argv[++a]);
    } else {
      palArgs.push_back(arg);
    }
  }
  if (ss < 1)
    ss = 1;
  if (ss > 8)
    ss = 8;

  std::vector<float> palT;
  std::vector<float> palRGB;

  for (size_t a = 0; a + 1 < palArgs.size(); a += 2) {
    float t = std::stof(palArgs[a]);
    unsigned int hex = (unsigned int)std::stoul(palArgs[a + 1], nullptr, 16);
    palT.push_back(t);
    palRGB.push_back(((hex >> 16) & 0xFF) / 255.0f);
    palRGB.push_back(((hex >> 8) & 0xFF) / 255.0f);
    palRGB.push_back((hex & 0xFF) / 255.0f);
  }

  // Fallback default palette if none supplied
  if (palT.empty()) {
    float defT[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float defRGB[] = {
        0.0f,  0.0f,  0.0f,  // #000000
        0.13f, 0.19f, 0.78f, // #2130C8
        0.55f, 0.0f,  0.13f, // #8C0021
        1.0f,  0.6f,  0.0f,  // #FF9900
        1.0f,  1.0f,  1.0f   // #FFFFFF
    };
    for (int i = 0; i < 5; ++i) {
      palT.push_back(defT[i]);
      palRGB.push_back(defRGB[i * 3 + 0]);
      palRGB.push_back(defRGB[i * 3 + 1]);
      palRGB.push_back(defRGB[i * 3 + 2]);
    }
  }

  int paletteSize = (int)palT.size();

  // Print all parameters for debugging
  std::cout << "=== Render Parameters ===" << std::endl;
  std::cout << "Type:          " << (type == 0 ? "Mandelbrot" : "Julia")
            << std::endl;
  std::cout << "Resolution:    " << width << " x " << height << std::endl;
  std::cout << "Max Iters:     " << maxIterations << std::endl;
  std::cout << "Center:        (" << xCenter << ", " << yCenter << ")"
            << std::endl;
  std::cout << "Zoom:          " << zoom << std::endl;
  if (type == 1)
    std::cout << "Julia C:       (" << cx << ", " << cy << ")" << std::endl;
  std::cout << "Output:        " << outputFilename << std::endl;
  std::cout << "Supersampling: " << ss << "x" << ss << std::endl;

  std::cout << "Palette stops: " << paletteSize << std::endl;
  for (int i = 0; i < paletteSize; ++i) {
    int r = (int)std::round(palRGB[i * 3 + 0] * 255);
    int g = (int)std::round(palRGB[i * 3 + 1] * 255);
    int b = (int)std::round(palRGB[i * 3 + 2] * 255);
    std::cout << "  [" << i << "] t=" << palT[i] << "  #" << std::hex
              << std::uppercase << std::setfill('0') << std::setw(2) << r
              << std::setw(2) << g << std::setw(2) << b << std::dec
              << std::endl;
  }
  std::cout << "=========================" << std::endl;

  FcParams p = {};
  p.type = type;
  p.width = width;
  p.height = height;
  p.maxIterations = maxIterations;
  p.xCenter = xCenter;
  p.yCenter = yCenter;
  p.zoom = zoom;
  p.cx = cx;
  p.cy = cy;
  p.ss = ss;
  p.paletteSize = paletteSize;
  p.palT = palT.data();
  p.palRGB = palRGB.data();

  std::cout << "Generating fractal on GPU..." << std::endl;
  std::vector<uint8_t> image;
  if (width >= 1 && height >= 1 && width <= 16384 && height <= 16384)
    image.resize(static_cast<size_t>(width) * height * 3);
  else
    image.resize(1); // fc_render reports the invalid size below

  if (fc_render(&p, image.data(), FC_FORMAT_RGB8, nullptr) != FC_OK) {
    std::cerr << "CUDA error: " << fc_last_error() << std::endl;
    return 1;
  }

  if (fc_write_png(outputFilename.c_str(), width, height, image.data()) ==
      FC_OK) {
    std::cout << "Fractal saved in " << outputFilename << std::endl;
  } else {
    std::cerr << "Failed to save image!" << std::endl;
    return 1;
  }
  return 0;
}
