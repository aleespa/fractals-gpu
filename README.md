# CppFractals

A simple C++ application that generates beautiful fractal images and saves them as PNG files.

---

## Features

- Generates fractal images in the complex plane.
- Allows customizing image resolution, location and iteration limits.
- Outputs PNG images (using `stb_image_write.h`).

---

## Requirements

- **Visual Studio 2019 or later** (or another C++17-compatible compiler).
- C++17 standard or higher.
- No external dependencies except [`stb_image_write.h`](https://github.com/nothings/stb).

---

## Building

### Using Visual Studio

1. Clone or download this repository.

2. Open `CppFractals.sln` in Visual Studio.

3. Make sure your configuration is set (e.g., Debug/Release and x64 or Win32).

4. Build the solution (`Build → Build Solution` or press **Ctrl + Shift + B**).

5. Run the application (`Debug → Start Without Debugging` or press **Ctrl + F5**).

---

## Usage

You can adjust parameters such as image width, height, location, zoom factor and maximum iterations.

```bash
CppFractals.exe <width> <height> <maxIterations> <xLocation> <yLocation> <zoomFactor> <outputFilename>
```

---

## Example Output

![Example Mandelbrot](./mandelbrot.png)

---

