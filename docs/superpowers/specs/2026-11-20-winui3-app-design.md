# WinUI 3 Fractal Explorer: Design

## Intent

The fractal explorer is a Node/Express web app that runs `CppFractals.exe` once per frame, writes a PNG to `tmp/` and streams it back to a browser. The owner doesn't think a web app fits here. They want a **standalone Windows desktop app** that you double-click and that runs on the local NVIDIA GPU, with **no server, no hosting, no browser and no network**.

The new app is a **C# WinUI 3** front end over the existing **CUDA kernels packaged as a native DLL**. It has **the same features as the web version**. Once it works, **the web version is deleted**.

## Goals

1. A WinUI 3 desktop app with full feature parity with `public/app.js` + `server.js` (list below).
2. CUDA rendering lives in a native DLL called in-process from C# (P/Invoke). No child processes and no temp files for interactive frames.
3. The command-line renderer keeps working with its **current argument contract** (`<type> <w> <h> <iter> <x> <y> <zoom> <cx> <cy> <out.png> [--ss N] [t hex ...]`), now built on the same DLL. The app's "Copy CLI command" feature depends on it.
4. The web version is removed: `server.js`, `public/`, `package.json`, `package-lock.json`, `node_modules/`, `tmp/`.

## Non-goals

- New fractal types, new features, or a UI redesign beyond translating to native controls.
- An installer, MSIX packaging, Store distribution or code signing.
- Non-NVIDIA / CPU fallback rendering.
- Deeper-zoom precision work (perturbation, double-double). Precision stays double as today.

## Anti-goals (failure even if it "works")

- Needs any network access or a localhost server to run.
- Needs an installer, a certificate, or developer mode to launch.
- Renders different pixels than today's CLI for the same parameters (kernel math must be unchanged).
- The UI freezes during a render, or long renders trip the Windows GPU watchdog (TDR).

## Feature parity checklist (from the web version)

- **Fractal type:** Mandelbrot / Julia. Julia shows Cx and Cy sliders (-2..2, step 0.001, default -0.4 / 0.6).
- **Max iterations** slider: 50–3000, step 10, default 400.
- **Pan:** drag moves the current image instantly; releasing re-renders.
- **Zoom:** mouse wheel zooms toward the cursor. The image scales instantly, then a debounced (~150 ms) re-render runs. Same math as `app.js` wheel handler.
- Stale renders are cancelled or superseded when a newer view is requested (web: `AbortController`).
- **Palette editor:** a gradient bar with stop handles. Click the empty bar to add a stop at that position, colored with the sampled gradient color. Drag a stop to move it (clamped to 0.001–0.999). Click a stop (<5 px movement) to open a color picker that live-updates. Right-click a stop to delete it (never the pinned stops at t=0/1, never below 2 stops). The end stops at 0 and 1 are pinned but their color can be changed. Every palette change re-renders.
- **Stats readout:** X, Y (6 decimals), Zoom shown as `1/zoom` with "x".
- **Reset View:** restores defaults (Mandelbrot: x=-1.4002, y=0, zoom=0.10; Julia: x=0, zoom=1.5; iter 400; cx/cy defaults) and the default 5-stop palette (`000000@0, 2130C8@.25, 8C0021@.5, FF9900@.75, FFFFFF@1`).
- **Export:** size presets 4K (3840×2160, default) and 8K (7680×4320), or custom W×H (16–16384). Smoothing Off / 4× / 16× (ss = 1/2/4, default 4×). Optional iteration override (blank means same as view). The **framed-view rule** from `server.js` applies: if the export aspect is wider than the on-screen view, `zoom *= exportAspect / viewAspect`. Only one export runs at a time (the button is disabled while rendering). Status text shows progress, the saved filename and elapsed seconds, or the error. **Save goes through a native Save As dialog**, with the default filename `<mandelbrot|julia>_<W>x<H>_<yyyyMMdd_HHmmss>.png`.
- **Copy CLI command:** copies a 5000×5000 command for the current view and palette with output `output_<timestamp>.png`, same format as `app.js`. The exe path is the CLI shipped alongside the app (absolute path, quoted). *(My call: the old relative `.\x64\Release\CppFractals.exe` path won't exist after the restructure.)*
- Window resize re-renders (debounced ~500 ms).

## Decided by the owner

- UI: **WinUI 3 in C#**. Backend: **CUDA as a native DLL**.
- Same features as the web version.
- Delete the web version.
- Standalone, offline, Windows-only.
- Work on branch `winui3-app`. Web deletion goes in its own commit.
- VS 2022 + CUDA 12.9 toolchain.

## My calls (owner can override)

- **Unpackaged, self-contained** WinUI 3 app (`WindowsPackageType=None`, `WindowsAppSDKSelfContained=true`, .NET self-contained). The output folder can be copied and its `.exe` run directly. An NVIDIA driver is still required.
- **Layout mirrors the web version:** the fractal fills the window, with a controls panel overlaid on the left. Mica/acrylic backdrop and native WinUI controls (ComboBox, Slider, NumberBox, ColorPicker in a Flyout, Buttons).
- **Render at physical pixels** (DPI-aware), so the view is crisp on high-DPI screens.
- **Project layout:**
  - `src/FractalCore/`: native DLL (vcxproj, CUDA build customization). Contains `Fractal.cu` kernels (math unchanged) and the band-launch/palette-upload logic moved out of `main.cu`. Exports a flat C API.
  - `src/FractalCli/`: console exe (argv parsing + PNG write via `stb_image_write`), linking FractalCore. Produces the CLI, ideally still named `CppFractals.exe` so existing commands and habits keep working.
  - `src/FractalExplorer/`: C# WinUI 3 app. Copies the DLL and the CLI exe into its output.
  - All three projects are in `CppFractals.sln`, so one `msbuild` Release|x64 builds everything.
- **C API shape** (the builder may refine it): an `extern "C"` render call that takes a params struct (type, w, h, iter, x, y, zoom, cx, cy, ss, palette arrays + count) and a caller-owned output buffer, returns an error code, and supports `fc_last_error()` text. It also takes a **cancel flag checked between 32-row bands**, so superseded interactive renders stop early and the 32-row band launches (TDR guard) are kept. A separate PNG-save export in the DLL is optional. C# can encode the PNG with `BitmapEncoder` instead; either is fine as long as the pixels match.
- Render calls run off the UI thread and are serialized (one GPU job at a time; an export may queue behind or cancel the interactive render).
- Docs: update `CLAUDE.md` and `README.md` to the new build/run instructions and project layout. Also remove the stale `include/Fractal.hpp` and `src/stb_impl.cpp`.

## Constraints

- Machine: Windows 11, VS 2022 Community (17.14) with CUDA 12.9 build customizations, plus VS 2026 Community, CUDA 12.9 and 13.2 toolkits, Windows SDK 10.0.26100.
- **No .NET SDK or WinUI workload is installed yet.** Before building, the owner must add the VS **".NET desktop development"** and **"WinUI application development"** workloads (or the .NET 8 SDK + Windows App SDK). The builder should treat this as a prerequisite step to tell the owner about, not something to install silently.
- Target x64 only. Target the .NET LTS version available (net8.0-windows10.0.19041.0 or later).
- Keep kernel math in `Fractal.cu` identical. Keep the 32-row band launches.
- **Git (owner decided):** the previous work in progress is committed as `59eca05` on branch `winui3-app`. All restructure work happens on that branch. That commit is the **pixel-identity baseline**: build its `CppFractals.exe` before restructuring and keep a copy outside the repo for comparison. Delete the web version **in its own commit**, so it stays recoverable from history.
- **Toolchain (owner decided):** build with **VS 2022 (17.14, v143)** and **CUDA 12.9**, keeping the current `compute_75,sm_75` code generation (don't change arch flags or toolkit, so pixels stay identical). The owner adds the ".NET desktop development" and "WinUI application development" workloads to VS 2022.
- `output_*.png` files in the repo root are the owner's renders. Leave them alone and don't commit them.

## Left to the builder

- Interactive display mechanism: `WriteableBitmap` with a CPU copy, or CUDA–D3D11 interop into a SwapChainPanel. Pick the simpler option that keeps pan/zoom responsive. WriteableBitmap is expected to suffice.
- How the instant pan/zoom preview is drawn (a transform on the current image, as the web version does with CSS).
- Kernel output format (RGB vs BGRA) and any conversion step.
- MVVM vs code-behind, control styling details, and exact debounce timings within the ranges above.

## Done means

- `msbuild CppFractals.sln /p:Configuration=Release /p:Platform=x64` builds the DLL, the CLI and the app.
- The app launches by double-click with networking disabled, and every checklist item above works.
- The CLI renders a PNG byte-identical (or pixel-identical) to the pre-change exe for the same arguments.
- The web files are gone and the docs describe the new setup.
