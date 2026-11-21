# Fractal Explorer UI Redesign: Design

## Intent

v0.1.0 ported the web explorer as-is: a single translucent overlay panel holding every control, a "Copy CLI Command" button, and an "Export PNG" form. The owner wants the app to feel like a native Windows app with a layout that can **grow**. Planned future features are more fractal types, more customization options, predefined colormaps, choosing where images are saved, and application settings. The app is **UI-only**. The CLI is no longer part of the product.

## Decided by the owner

1. **No CLI in the app.** The "Copy CLI command" feature is removed, and the UI↔backend integration no longer has to stay compatible with a command-line tool.
2. **Coordinates in the bottom-right corner** of the fractal view, as a small translucent overlay (Google Earth style).
3. **A Save button** replaces "Export PNG".
4. **Layout A** (chosen from three mockups): a WinUI `NavigationView` rail of icons on the left, a collapsible side panel whose content follows the selected rail item, the fractal canvas filling the rest, and the built-in Settings entry at the bottom of the rail.

## My calls (the owner said "go ahead" on these)

- **Rail items:**
  - **Fractal**: type, max iterations, Julia Cx/Cy (Julia only), Reset view.
  - **Colours**: the existing gradient editor, plus a reserved "Presets" section header. Presets are not built now.
  - **Settings**: footer item, NavigationView's built-in settings entry.

  Selecting the currently open item toggles the pane closed or open. A hamburger button collapses the pane. The pane uses `PaneDisplayMode="Left"` with a compact rail when collapsed.
- **Save:** a `SplitButton` at the top-right of the canvas, accent style, labelled "Save".
  - **Main click** saves immediately to the configured save folder (default `%USERPROFILE%\Pictures\Fractal Explorer`, created if missing). It uses the default save size and smoothing from Settings and the view's iterations. The file name is `DefaultFileName(...)`. If that name exists, it appends ` (2)`, ` (3)` and so on.
  - **The ▾ part** opens **"Save as…"**, a `ContentDialog` with:
    - Size: 4K / 8K / Custom W×H (16–16384)
    - Smoothing: Off / 4× / 16×
    - Iterations: blank means same as view
    - Location: FileSavePicker

    It keeps the framed-view zoom rule (`AdjustZoomForAspect`).
  - **While saving**, a progress strip shows "Saving 3840 × 2160… 42%" with a **Cancel** button, and Save is disabled. When it finishes, an `InfoBar` (Success) shows "Saved <name>" with an **Open folder** button. Errors appear in the InfoBar (Error).
- **Settings page** (shown in the content area, replacing the canvas while open):
  - Save folder (path + "Change…" folder picker + "Open")
  - Default save size (4K/8K/Custom W×H)
  - Default smoothing
  - App theme (Use system / Light / Dark)
  - An "About" line with the version, 0.2.0

  Settings persist as JSON at `%LOCALAPPDATA%\FractalExplorer\settings.json`. A missing or corrupt file means defaults are used, and the app never crashes on it.
- **Coordinates overlay** (bottom-right): `x <F7>  y <F7>  zoom <ZoomText>`, monospace, translucent background, non-interactive, so it never blocks pan or zoom.
- **Backend robustness (now that the CLI no longer constrains the API):**
  - **Progress + cancel for saves.** Add `fc_render_progress(const FcParams*, uint8_t* out, int32_t format, const volatile int32_t* cancelFlag, FcProgressFn cb, void* user)`. The callback is called after each band with (rowsDone, totalRows). `fc_render` stays as a wrapper with no callback. Kernel math, bands and launch geometry are unchanged.
  - **Managed render scheduling.** A single `RenderQueue` serializes GPU jobs in C# (`SemaphoreSlim`). A superseded interactive render that is still waiting is dropped before it touches the GPU. The pixel buffer is rented only after the GPU slot is acquired. This fixes v0.1.0's pile-up of blocked threads and buffers behind a long export.
- **CLI:** the app no longer bundles or references `CppFractals.exe`, and `CliCommand` and its tests are removed from Core. The `FractalCli` project stays in the solution as a **developer tool**, for the pixel-identity baseline script and the `Render_MatchesCli_OddSize` test.

## Non-goals

- Predefined colormaps, new fractal types, tabs/multiple documents, undo/redo. The layout leaves room for them, but none are built now.
- Changing kernel math or rendered pixels.
- Packaging/installer changes. The app stays unpackaged and self-contained.

## Anti-goals

- Any regression of v0.1.0 behaviour: pan/zoom feel, stale-render guards, palette editor, physical-pixel rendering, error InfoBar, crash-safety of async handlers.
- The coordinates overlay or Save button intercepting pan/zoom input.
- Settings I/O failures crashing the app.

## Constraints

- Same toolchain and invariants as `CLAUDE.md`: VS 2022 MSBuild, CUDA 12.9, `compute_75,sm_75`, 32-row bands, `FcParams` size 80, the `MSBuildRuntimeType=='Full'` condition on C#→vcxproj references, and pixel identity verified with `tools\compare-baseline.ps1`.
- WinUI controls only from the Windows App SDK (1.8), with no extra NuGet UI toolkits.
- Version bumps to **0.2.0** (app About line, csproj `Version`).

## Left to the builder

- Exact XAML styling, icons (Segoe Fluent Icons glyphs), spacing.
- Whether each pane section is a `Page` in a `Frame` or a `UserControl` swapped in code.
- How settings are bound (code-behind is fine).
