# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CUDA-accelerated Mandelbrot/Julia renderer (`CppFractals.exe`) plus a Node/Express web explorer that shells out to the exe for every frame. Windows-only toolchain (MSVC + CUDA 12.9 build customization).

## Commands

- **Build the renderer:** open `CppFractals.sln` in Visual Studio, select **Release | x64**, build. The web server hard-codes `x64/Release/CppFractals.exe`, so Debug builds are not picked up by the UI. From a VS Developer prompt: `msbuild CppFractals.sln /p:Configuration=Release /p:Platform=x64`.
- **Run the web UI:** `npm install` then `node server.js` → http://localhost:3000.
- **Render from CLI:**
  ```
  x64\Release\CppFractals.exe <type> <width> <height> <maxIter> <xCenter> <yCenter> <zoom> <cx> <cy> <output.png> [--ss N] [t1 hex1 t2 hex2 ...]
  ```
  `type`: 0 = Mandelbrot, 1 = Julia (`cx, cy` is the Julia constant, ignored for Mandelbrot). Palette stops are `t` in [0,1] followed by a hex color without `#`; if omitted a built-in 5-stop palette is used. `--ss N` averages N×N samples per pixel (anti-aliasing; the UI offers 1/2/4). Note: the README's CLI example uses an older argument order and is out of date — `src/main.cu` is authoritative.
- There are no tests or linters.

## Architecture

Three layers that share one positional argument contract — changing parameters means updating all three:

1. **`src/main.cu`** — parses argv positionally, uploads palette (`palT` positions + flat `palRGB` floats) to the GPU, computes the view rect (`zoom` is the half-width in the complex plane; half-height = `zoom / aspect`), launches 16×16-block kernels in 32-row bands (each band is a separate launch so long renders stay under the Windows ~2 s GPU watchdog/TDR — keep this when changing kernels), writes PNG via `stb_image_write` (implementation is defined in `main.cu`).
2. **`src/Fractal.cu`** — `mandelbrotKernel` / `juliaKernel` are thin wrappers over a shared `renderPixel` (supersampling) → `escapeColor` (double precision, smooth iteration coloring) → `paletteColor` (linear interpolation between stops). Kernels are declared `extern` in `main.cu`; adding a fractal type means adding a kernel here, a declaration and dispatch branch in `main.cu`.
3. **`server.js`** — `GET /api/fractal?type&w&h&iter&x&y&z&cx&cy&pal` maps query params to the exe's argv, writes to `tmp/`, streams the PNG back and deletes it. `GET /api/export` takes the same params plus `ss` and `va` (on-screen aspect ratio; used to widen `zoom` so a different export shape still contains the framed view), allows one export at a time, and responds as a file download. `pal` is a flat comma list `t,hex,t,hex,...` that is split straight into argv.
4. **`public/app.js`** — client state + palette editor (draggable gradient stops). Pan/zoom is applied instantly as a CSS transform on the current image, then a debounced `fetchFractal()` (with `AbortController` cancelling stale requests) re-renders at the new view. The Export panel calls `/api/export` and saves the response through a blob download. The "copy command" button puts a 5000×5000 CLI command for the current view and palette on the clipboard. Running that command from the repo root produces the `output_<timestamp>.png` files that show up there; `.gitignore` doesn't ignore them, so don't commit them by accident.

Stale/unused: `include/Fractal.hpp` (old CPU `Fractal` class, no implementation) and `src/stb_impl.cpp` (not in the vcxproj; would duplicate the stb implementation if added).
