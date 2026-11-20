# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CUDA-accelerated Mandelbrot/Julia renderer with a WinUI 3 desktop explorer and a CLI. Windows-only toolchain (MSVC, .NET 8, CUDA 12.9 build customization).

## Prerequisites

VS 2022 (not 2026) with workloads "Desktop development with C++", ".NET desktop development", "WinUI application development"; CUDA 12.9; NVIDIA driver.

## Commands

- **Build everything** (VS 2022 Developer prompt): `msbuild CppFractals.sln /restore /m /p:Configuration=Release /p:Platform=x64`
- **Run the app:** `src\FractalExplorer\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64\FractalExplorer.exe`. That folder (FractalExplorer.exe, FractalCore.dll, CppFractals.exe) is unpackaged, self-contained and can be copied anywhere; it needs only an NVIDIA driver.
- **CLI:**
  ```
  x64\Release\CppFractals.exe <type> <width> <height> <maxIter> <xCenter> <yCenter> <zoom> <cx> <cy> <output.png> [--ss N] [t1 hex1 t2 hex2 ...]
  ```
  `type`: 0 = Mandelbrot, 1 = Julia (`cx, cy` is the Julia constant, ignored for Mandelbrot). `zoom` is the half-width in the complex plane (half-height = `zoom / aspect`). Palette stops are `t` in [0,1] plus a hex color without `#`; omitted means the built-in 5-stop palette. `--ss N` averages N x N samples per pixel.
- **Tests:** `dotnet test tests\FractalExplorer.Tests\FractalExplorer.Tests.csproj -c Release -p:Platform=x64` (xUnit; run the full msbuild first: it copies the prebuilt native DLL/exe from x64\Release, and dotnet build/test cannot build the C++/CUDA projects).
- **Pixel-identity check:** `pwsh -File tools\compare-baseline.ps1 -Baseline <saved baseline CppFractals.exe> -Candidate x64\Release\CppFractals.exe` (5 cases, all must PASS).
- No linters.

## Architecture

- **`src/FractalCore`** (CUDA DLL): flat C API in `fractal_core.h` (`fc_render`, `fc_write_png`, `fc_last_error`). `Fractal.cu` has the kernels (`renderPixel` supersampling -> `escapeColor`, double precision, smooth coloring -> `paletteColor`, linear interpolation between stops); `render.cu` uploads the palette, computes the view rect and launches; `png.cpp` writes PNG via stb. Static cudart and static CRT, so the DLL depends only on KERNEL32.
- **`src/FractalCli`**: `CppFractals.exe`, thin wrapper over the DLL with the argv contract above.
- **`src/FractalExplorer.Core`** (C#): Palette, ViewState, PreviewTransform, ExportMath, CliCommand, and the `NativeRenderer` P/Invoke.
- **`src/FractalExplorer`**: WinUI 3 app (pan/zoom applied instantly as a preview transform, then re-rendered; gradient editor; export; copy-CLI-command).
- **`tests/FractalExplorer.Tests`**: xUnit tests for Core and the native renderer.

## Invariants

- Keep the 32-row band launches (one launch per band) so long renders stay under the Windows ~2 s GPU watchdog (TDR).
- Keep kernel math pixel-identical; verify with `tools\compare-baseline.ps1`.
- CUDA codegen is `compute_75,sm_75`.
- C# `FcParams` must mirror `fractal_core.h` (size 80); a test checks this. Changing parameters means updating the header, `render.cu`, `NativeRenderer.cs`, the CLI and `CliCommand`.
- C# -> vcxproj ProjectReferences carry the `MSBuildRuntimeType=='Full'` condition.
- Adding a fractal type: a kernel in `Fractal.cu`, a dispatch branch in `render.cu`, and `FractalType` in Core.
- `output_*.png` files in the repo root come from the copied CLI command; `.gitignore` does not ignore them, so do not commit them.

