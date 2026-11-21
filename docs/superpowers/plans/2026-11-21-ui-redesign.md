# Fractal Explorer UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the WinUI shell as a NavigationView app with Fractal and Colours panes, a Settings page, a bottom-right coordinates overlay, and a Save split button with progress and cancel. Remove the CLI from the app, and add progress reporting plus managed render scheduling in the backend.

**Architecture:** Testable logic goes into `FractalExplorer.Core`:
- `RenderQueue` serializes GPU jobs.
- `AppSettings`/`SettingsStore` hold the persisted settings.
- `SavePlanner` resolves file paths and requests.
- `NativeRenderer` gets a progress overload.

`FractalCore.dll` gains `fc_render_progress`. The app becomes a thin shell:
- `MainWindow` hosts the NavigationView, canvas, overlays and Save.
- Each pane is a `UserControl`.
- Settings is a `UserControl` shown in place of the canvas.

**Tech Stack:** MSVC v143 + CUDA 12.9 (FractalCore), .NET 8, Windows App SDK 1.8 (WinUI 3, unpackaged, self-contained), xUnit, System.Text.Json.

**Spec:** `docs/superpowers/specs/2026-11-21-ui-redesign-design.md`

## Global Constraints

- Branch `ui-redesign`. VS 2022 MSBuild only (never VS 2026), with the full build command `msbuild CppFractals.sln /restore /m /p:Configuration=Release /p:Platform=x64`.
- Tests: `dotnet test tests\FractalExplorer.Tests\FractalExplorer.Tests.csproj -c Release -p:Platform=x64`, run after the full build.
- Kernel math, 32-row bands and launch geometry are unchanged. `pwsh -File tools\compare-baseline.ps1 -Baseline $env:LOCALAPPDATA\CppFractals-baseline\CppFractals.exe -Candidate x64\Release\CppFractals.exe` must print 5× PASS after any native change.
- `FcParams` stays at 80 bytes and its layout does not change. C#→vcxproj ProjectReferences carry `Condition="'$(MSBuildRuntimeType)' == 'Full'"` and `ReferenceOutputAssembly="false"`.
- No extra NuGet UI toolkits. Only Windows App SDK controls.
- Settings file: `%LOCALAPPDATA%\FractalExplorer\settings.json`. Default save folder: `%USERPROFILE%\Pictures\Fractal Explorer`.
- App version **0.2.0** (csproj `<Version>0.2.0</Version>`, shown in Settings → About).
- Text↔number conversions use `CultureInfo.InvariantCulture`, except user-facing zoom text (`ViewState.ZoomText(CurrentCulture)`).
- Never touch or commit the owner's `output_*.png`. Stage files explicitly.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01WsobN8MYWP8HEe1AuiyE8B`.

## Review Focus

1. **Corrupt, partial or unwritable settings.json** (hand-edited, disk full, read-only profile): the app starts with defaults and never crashes, and a failed save of settings is silent apart from a debug log. Test: `SettingsStoreTests.Load_CorruptJson_ReturnsDefaults`, `Load_UnknownAndMissingFields_FillsDefaults`, and `Save_UnwritableDirectory_DoesNotThrow` (Task 3).
2. **Quick-save folder missing or deleted, or the file name already exists**: the folder is recreated and the name gets ` (2)`, ` (3)` and so on, so nothing is overwritten. Test: `SavePlannerTests.UniquePath_AppendsCounter` and `QuickSavePath_CreatesFolder` (Task 3).
3. **Cancel pressed during a long save**: the render stops within one band, no file is left behind, and Save is re-enabled. Test: `NativeRendererTests.RenderWithProgress_CancelFromCallback_ReturnsFalse` (Task 1). The app wiring is covered by a manual check in Task 6.
4. **A burst of interactive render requests while a save holds the GPU**: only the newest interactive request runs after the save, and older waiting ones never touch the GPU. Test: `RenderQueueTests.SupersededWaitingJob_IsCancelledWithoutRunning` (Task 2).
5. **Settings save folder chosen on a path that later disappears or isn't writable**: quick save falls back to creating the folder; if that fails it shows an error InfoBar suggesting "Save as…" and never crashes. Test: `SavePlannerTests.QuickSavePath_UncreatableFolder_Throws` (Task 3, a typed `IOException` that the app catches). Manual check in Task 6.

---

## File Structure

```
src/FractalCore/fractal_core.h, render.cu      + fc_render_progress (Task 1)
src/FractalExplorer.Core/
  NativeRenderer.cs        + Render(..., Action<double>? progress, ct) overload (Task 1)
  RenderQueue.cs           new (Task 2)
  AppSettings.cs           new: settings record + SettingsStore (Task 3)
  SavePlanner.cs           new: quick-save path, unique names, SaveSpec (Task 3)
  CliCommand.cs            deleted; Palette.ToCliArgs deleted (Task 4)
src/FractalExplorer/
  MainWindow.xaml(.cs)     rewritten shell (Task 4), Save wiring (Task 6)
  Panes/FractalPane.xaml(.cs)   new (Task 4)
  Panes/ColoursPane.xaml(.cs)   new (Task 4), hosts Controls/GradientEditor
  Panes/SettingsView.xaml(.cs)  new (Task 5)
  Dialogs/SaveAsDialog.xaml(.cs) new (Task 6)
tests/FractalExplorer.Tests/  RenderQueueTests.cs, SettingsStoreTests.cs, SavePlannerTests.cs new; CliCommandTests.cs deleted
CLAUDE.md, README.md          updated (Task 7)
```

---

### Task 1: Native progress callback + managed overload

**Files:** Modify `src/FractalCore/fractal_core.h`, `src/FractalCore/render.cu`, `src/FractalExplorer.Core/NativeRenderer.cs`. Test: `tests/FractalExplorer.Tests/NativeRendererTests.cs`.

**Interfaces:**
- Produces (C, appended to `fractal_core.h`):
  ```c
  typedef void (*FcProgressFn)(int32_t rowsDone, int32_t totalRows, void* user);
  /* Same as fc_render; calls cb (if non-NULL) on the calling thread after each completed band. */
  FC_API int32_t fc_render_progress(const FcParams* p, uint8_t* out, int32_t format,
                                    const volatile int32_t* cancelFlag, FcProgressFn cb, void* user);
  ```
  `fc_render` becomes `return fc_render_progress(p, out, format, cancelFlag, NULL, NULL);`.
- Produces (C#): `public static bool Render(RenderRequest req, PixelFormat format, Span<byte> output, Action<double>? progress, CancellationToken ct)`. Progress is a fraction from 0 to 1 and reaches exactly 1.0 when the render completes. The existing 4-arg `Render` forwards with `progress: null`. Use an `[UnmanagedCallersOnly]` static callback, passing a `GCHandle` to the `Action<double>` as `user`. Exceptions thrown by the progress delegate must not cross into native code: catch them and set the cancel flag.

- [ ] **Step 1: Write failing tests:**
  - `RenderWithProgress_ReportsMonotonicToOne`: render 64×100 (4 bands). Collect the values; they are non-decreasing, the last is `1.0`, and `Count == 4`.
  - `RenderWithProgress_CancelFromCallback_ReturnsFalse`: 64×256. The progress delegate cancels a `CancellationTokenSource` on the first call, so the result is `false` and fewer than 8 callbacks arrive.
  - `RenderWithProgress_ThrowingCallback_CancelsWithoutCrash`: the delegate throws `InvalidOperationException`, so the result is `false` and no exception escapes.
- [ ] **Step 2:** Full build, then run the tests → compile FAIL (no overload).
- [ ] **Step 3:** Implement the native and managed code per Interfaces.
- [ ] **Step 4:** Full build. All tests PASS. Baseline script 5× PASS.
- [ ] **Step 5:** Commit `feat(core): render progress callback for long renders`.

---

### Task 2: RenderQueue (managed GPU scheduling)

**Files:** Create `src/FractalExplorer.Core/RenderQueue.cs`, `tests/FractalExplorer.Tests/RenderQueueTests.cs`.

**Interfaces:**
- Produces:
  ```csharp
  public enum RenderLane { Interactive, Save }
  public sealed class RenderQueue
  {
      // Runs job on a thread-pool thread once the single GPU slot is free. Jobs run one at a time, FIFO.
      // Interactive lane: queuing a new Interactive job cancels any Interactive job that is still WAITING
      // (its Task completes as canceled, job never invoked). A RUNNING job is only cancelled via its own ct.
      // Save-lane jobs are never superseded. If ct is cancelled while waiting, the job is never invoked.
      public Task<T> RunAsync<T>(RenderLane lane, Func<CancellationToken, T> job, CancellationToken ct);
  }
  ```

- [ ] **Step 1: Write failing tests** (use `TaskCompletionSource` gates, no sleeps over 2 s):
  - `JobsRunOneAtATime`: two jobs. The first blocks on a gate. The second hasn't started until the gate opens, and the max observed concurrency is 1.
  - `SupersededWaitingJob_IsCancelledWithoutRunning`: a Save job is blocked. Queue Interactive A, then B. A's task is canceled (`TaskCanceledException`) and its delegate is never invoked. B runs after the Save job.
  - `SaveJobs_AreNotSuperseded`: a blocked Interactive job, then Save S1 and Save S2. Both saves run.
  - `CancelledWhileWaiting_NeverRuns`: the job's own ct is cancelled while it waits, so the task is canceled and the delegate is not invoked.
  - `JobException_PropagatesAndQueueContinues`: a job throws `FractalRenderException("x")`, and awaiting rethrows it. The next job still runs.
- [ ] **Step 2:** Run the tests → FAIL (type missing).
- [ ] **Step 3:** Implement with a `SemaphoreSlim(1,1)` plus a per-lane "latest waiting interactive" CTS.
- [ ] **Step 4:** Tests PASS.
- [ ] **Step 5:** Commit `feat(core): RenderQueue serializes GPU jobs and drops superseded renders`.

---

### Task 3: Settings and save planning

**Files:** Create `src/FractalExplorer.Core/AppSettings.cs`, `src/FractalExplorer.Core/SavePlanner.cs`, `tests/FractalExplorer.Tests/SettingsStoreTests.cs`, `tests/FractalExplorer.Tests/SavePlannerTests.cs`.

**Interfaces:**
- Produces:
  ```csharp
  public enum AppTheme { System, Light, Dark }
  public sealed record AppSettings
  {
      public string SaveFolder { get; init; }              // default: Path.Combine(Environment.GetFolderPath(MyPictures), "Fractal Explorer")
      public int SaveWidth { get; init; } = 3840;
      public int SaveHeight { get; init; } = 2160;
      public int SaveSmoothingIndex { get; init; } = 1;     // 0 Off, 1 4×, 2 16× (ExportMath.SupersampleForIndex)
      public AppTheme Theme { get; init; } = AppTheme.System;
      public static AppSettings Default { get; }
      public AppSettings Normalized();                      // invalid size → 3840×2160, index outside 0..2 → 1, blank folder → default
  }
  public sealed class SettingsStore
  {
      public SettingsStore(string filePath);               // app uses %LOCALAPPDATA%\FractalExplorer\settings.json
      public static string DefaultPath { get; }
      public AppSettings Load();                            // never throws; missing/corrupt → Default; result Normalized()
      public void Save(AppSettings s);                      // never throws; creates directory; writes temp file then File.Move(overwrite)
  }
  public sealed record SaveSpec(int Width, int Height, int Supersample, int MaxIterations, string Path);
  public static class SavePlanner
  {
      // "name.png" → "name (2).png", "name (3).png", … first non-existing.
      public static string UniquePath(string folder, string fileName);
      // Creates folder (Directory.CreateDirectory); throws IOException/UnauthorizedAccessException if it can't.
      public static string QuickSavePath(AppSettings s, FractalType type, DateTime local);
      public static RenderRequest BuildRequest(ViewState view, (float[] T, float[] Rgb) palette, SaveSpec spec, double viewAspect);
      // BuildRequest: Zoom = ExportMath.AdjustZoomForAspect(view.Zoom, w, h, viewAspect <= 0 ? w/(double)h : viewAspect)
  }
  ```
  JSON: `System.Text.Json`, camelCase, `JsonStringEnumConverter` for `Theme`.

- [ ] **Step 1: Write failing tests** (each uses its own temp dir and cleans up in `finally`):
  - `Load_MissingFile_ReturnsDefaults`
  - `SaveThenLoad_RoundTrips`: with folder `C:\x`, 1000×500, index 2 and Dark, the loaded record equals the saved one.
  - `Load_CorruptJson_ReturnsDefaults`: the file contains `{not json`.
  - `Load_UnknownAndMissingFields_FillsDefaults`: the file contains `{"saveWidth":1234,"bogus":1}`, giving SaveWidth 1234, SaveHeight 2160 and Theme System.
  - `Load_InvalidValues_Normalized`: `{"saveWidth":5,"saveSmoothingIndex":9}` gives 3840×2160 and index 1.
  - `Save_UnwritableDirectory_DoesNotThrow`: the path is under a file used as a directory (`<tmp>\file.txt\settings.json`), so no exception.
  - `UniquePath_AppendsCounter`: with `a.png` and `a (2).png` existing, the result is `a (3).png`. In an empty folder the result is `a.png`.
  - `QuickSavePath_CreatesFolder`: the folder doesn't exist, and afterwards it exists and the file name matches `ExportMath.DefaultFileName(...)`.
  - `QuickSavePath_UncreatableFolder_Throws`: SaveFolder is under an existing file, so it throws an `IOException`.
  - `BuildRequest_WidensZoomForWiderExport`: view zoom 0.1, viewAspect 1.0, spec 2000×1000 gives `Zoom == 0.2`, and `Supersample`/`MaxIterations` are copied from the spec.
- [ ] **Step 2:** Run the tests → FAIL.
- [ ] **Step 3:** Implement per Interfaces.
- [ ] **Step 4:** Tests PASS.
- [ ] **Step 5:** Commit `feat(core): persisted app settings and save planning`.

---

### Task 4: Remove CLI from the app; new NavigationView shell

**Files:**
- Delete: `src/FractalExplorer.Core/CliCommand.cs`, `tests/FractalExplorer.Tests/CliCommandTests.cs`, and `Palette.ToCliArgs` with its test `ToCliArgs_FormatsSixDecimalsUppercaseHex`.
- Modify: `src/FractalExplorer/FractalExplorer.csproj`:
  - remove the `CppFractals.exe` copy item;
  - point the conditioned ProjectReference at `..\FractalCore\FractalCore.vcxproj` instead of FractalCli;
  - add `<Version>0.2.0</Version>`.

  Then rewrite `MainWindow.xaml(.cs)` and create `Panes/FractalPane.xaml(.cs)` and `Panes/ColoursPane.xaml(.cs)`.
- The test project keeps its FractalCli reference and the `CppFractals.exe` copy, because the dev-tool tests still use them.

**Interfaces:**
- Consumes: `RenderQueue` (Task 2), `ViewState`, `PreviewTransform`, `Palette`, `NativeRenderer`, `Controls/GradientEditor` (existing).
- Produces:
  - `MainWindow` with:
    - `internal ViewState View`, `internal Palette Palette`, `public void RequestRender(int debounceMs)`;
    - `internal RenderQueue Renderer`;
    - `internal void ShowError(string message, string title)` and `internal void ShowSuccess(string message, string? openFolderPath)`;
    - a named `Grid CanvasHost` (the fractal area) and a `Grid SaveHost` (top-right of the canvas, where Task 6 puts the SplitButton).
  - `FractalPane(MainWindow owner)`, `ColoursPane(MainWindow owner)`.
  - `public void SyncFromState()` on both panes (called after Reset).

Behaviour:
- **Shell:** `NavigationView` with `PaneDisplayMode="Left"`, `IsSettingsVisible="True"` and `IsBackButtonVisible="Collapsed"`. Menu items are **Fractal** (glyph E8B9 or similar) and **Colours** (glyph E790). The pane width is about 300. The selected item's pane content shows in `NavigationView.PaneFooter`/content as the builder prefers, as long as the result is a rail + side panel + canvas.
  - Re-selecting the open item toggles `IsPaneOpen`.
  - The Settings item shows a placeholder TextBlock "Settings" in the content area for now. Task 5 replaces it.
- **FractalPane:**
  - Type ComboBox (Mandelbrot Set / Julia Set)
  - Max iterations Slider 50–3000 step 10, with a "Max iterations · N" label
  - Julia Cx/Cy Sliders −2..2 step 0.001, labels to 3 decimals, visible only for Julia
  - a "Reset view" button with the same semantics as v0.1.0
- **ColoursPane:** "Gradient" header + `GradientEditor`, a caption "Click the bar to add a stop · drag to move · click a stop to recolour · right-click to delete", and a "Presets" header with the caption "Coming soon" (reserved space, spec non-goal).
- **Canvas:** the same pan/zoom/preview/render pipeline as v0.1.0 (keep the `_renderSeq`/`_viewVersion` guards and physical-pixel sizing), but interactive renders go through `Renderer.RunAsync(RenderLane.Interactive, ...)`. The frame buffer is rented **inside** the job, not before queuing. A `TaskCanceledException` from a superseded job is ignored silently.
- **Coordinates overlay:** bottom-right of `CanvasHost`, `IsHitTestVisible="False"`, monospace, translucent dark background with CornerRadius 6 and padding 10,6. The text is `x {X:F7}   y {Y:F7}   zoom {ZoomText(CurrentCulture)}`, with invariant formatting for X/Y. It updates on every pan and zoom.
- Keep the error `InfoBar`. Add a success `InfoBar` (Severity Success) with an optional "Open folder" `HyperlinkButton` that runs `explorer.exe /select,"<file>"`.
- No Export panel, no Copy CLI button. The old export code is removed; Task 6 adds Save.

- [ ] **Step 1:** Delete the CLI-related code and tests. Full build + tests → PASS (fewer tests).
- [ ] **Step 2:** Implement the shell, panes, overlay and queue integration.
- [ ] **Step 3:** Full build + tests → PASS. Check the app output folder no longer contains `CppFractals.exe` (`Test-Path` is False).
- [ ] **Step 4:** Smoke launch: the app stays alive ~10 s and closes cleanly. Optionally take a screenshot to `.superpowers/` (git-ignored) and look at it to confirm the rail, the pane and the overlay at bottom-right.
- [ ] **Step 5:** Commit `feat(app): NavigationView shell, panes, bottom-right coordinates; drop CLI from app`.

---

### Task 5: Settings page

**Files:** Create `src/FractalExplorer/Panes/SettingsView.xaml(.cs)`. Modify `MainWindow.xaml.cs` and `App.xaml.cs`.

**Interfaces:**
- Consumes: `AppSettings`, `SettingsStore`, `AppTheme` (Task 3), MainWindow (Task 4).
- Produces:
  - `MainWindow.Settings` (`AppSettings`, current), loaded at startup with `new SettingsStore(SettingsStore.DefaultPath).Load()`;
  - `MainWindow.UpdateSettings(Func<AppSettings, AppSettings> change)`, which applies the change, saves and re-applies the theme.

Behaviour:
- Selecting the NavigationView Settings item shows `SettingsView` in place of the canvas. Selecting Fractal or Colours shows the canvas again, keeping the view state, with no re-render needed unless the size changed.
- **Sections (headers + cards):**
  - **Saving:**
    - Save folder: the path text, a "Change…" button (`FolderPicker` initialised with the window HWND) and "Open folder" (`explorer.exe "<folder>"`, creating it first if missing).
    - Default size: 4K / 8K / Custom with W×H NumberBoxes, `ValidationMode="Disabled"`, applied only if `ExportMath.IsValidSize`. Otherwise show an inline red caption "Width and height must be between 16 and 16384." and don't save.
    - Default smoothing: Off / 4× / 16×.
  - **Appearance:** Theme RadioButtons for Use system setting / Light / Dark. Apply by setting `((FrameworkElement)Content).RequestedTheme` (System → `ElementTheme.Default`).
  - **About:** "Fractal Explorer 0.2.0", read from the assembly's informational/version attribute, with "GPU-rendered Mandelbrot and Julia sets." under it.
- The theme from settings is applied at startup.

- [ ] **Step 1:** Implement.
- [ ] **Step 2:** Full build + tests PASS.
- [ ] **Step 3:** Smoke launch. Delete `%LOCALAPPDATA%\FractalExplorer\settings.json` first, then confirm the app starts with defaults. Write `{bad` into it and confirm the app still starts.
- [ ] **Step 4:** Commit `feat(app): settings page with save folder, defaults and theme`.

---

### Task 6: Save split button, Save as dialog, progress and cancel

**Files:** Create `src/FractalExplorer/Dialogs/SaveAsDialog.xaml(.cs)`. Modify `MainWindow.xaml(.cs)`.

**Interfaces:**
- Consumes: `SavePlanner`, `SaveSpec`, `RenderQueue` (`RenderLane.Save`), `NativeRenderer.Render(..., progress, ct)` + `WritePng`, `MainWindow.Settings`, `ShowError`/`ShowSuccess`.
- Produces: `SaveAsDialog : ContentDialog` with `public SaveSpec? Result`, set when the user confirms with a valid size and a chosen path.

Behaviour:
- **Button:** a `SplitButton` in `SaveHost` (top-right of the canvas), accent style (`Style="{StaticResource AccentButtonStyle}"` content + a Save glyph E74E), with the content "Save". Its flyout has one item, "Save as…". Ctrl+S runs quick save and Ctrl+Shift+S runs Save as (`KeyboardAccelerator`).
- **Quick save:**
  1. `SavePlanner.QuickSavePath(Settings, View.Type, DateTime.Now)`. An `IOException`/`UnauthorizedAccessException` goes to `ShowError("Couldn't use the save folder: <msg>. Try Save as… or change the folder in Settings.", "Save failed")`.
  2. The spec is built from Settings (`SaveWidth`/`SaveHeight`, `SupersampleForIndex(SaveSmoothingIndex)`) and `View.MaxIterations`.
- **Save as dialog:**
  - Size: 4K/8K/Custom, pre-selected from Settings, with Custom NumberBoxes `ValidationMode="Disabled"`.
  - Smoothing: Off/4×/16×, pre-selected from Settings.
  - Iterations NumberBox: placeholder "Same as view (N)"; blank or ≤0 means the view value.
  - Primary button "Save…": validates with `IsValidSize`; if invalid, shows the inline red error and keeps the dialog open (cancel the Closing event). It then opens a `FileSavePicker` (HWND-initialised, PNG, suggested name = `DefaultFileName` without `.png`, start location Pictures). If the picker is cancelled, the dialog stays open.
- **Running a save (both paths):**
  1. Snapshot `View` + `Palette.ToNative()` + the view aspect on the UI thread, and build the request with `SavePlanner.BuildRequest`.
  2. Show a progress strip at the top of the canvas: a `ProgressBar` with value 0–100, the text "Saving W × H… NN%" (invariant), and a "Cancel" button. Disable the SplitButton.
  3. Run `Renderer.RunAsync(RenderLane.Save, ct => { allocate W*H*3 (OOM → specific message); Render(req, Rgb8, buf, p => DispatcherQueue.TryEnqueue(update UI), ct) ; if completed WritePng(path) }, cts.Token)`.
  4. Outcomes:
     - **Success:** `ShowSuccess($"Saved {Path.GetFileName(path)} ({secs:F1}s)", path)`.
     - **Cancel:** InfoBar Informational "Save cancelled". No file is left: delete it if it exists and was created by this save.
     - **Error:** ShowError with the message. Delete an empty or partial file as v0.1.0 does.
     - **Always:** hide the strip and re-enable the button in `finally`.
  5. Catch everything (async void).

- [ ] **Step 1:** Implement.
- [ ] **Step 2:** Full build + tests PASS.
- [ ] **Step 3:** Smoke launch: the app stays up ~10 s. If feasible, take a screenshot showing the Save button top-right.
- [ ] **Step 4:** Commit `feat(app): Save split button with Save as dialog, progress and cancel`.

---

### Task 7: Docs and version

**Files:** Modify `CLAUDE.md`, `README.md`.

- [ ] **Step 1:** Update the docs:
  - Describe the app layout (rail + panes, Settings, Save/Save as, the bottom-right coordinates).
  - Settings file location and the default save folder.
  - The CLI is now a developer tool only (baseline script + tests), not shipped in the app folder.
  - Architecture: add `RenderQueue`, `SettingsStore`, `SavePlanner`, `fc_render_progress`. Remove `CliCommand` and the copy-command text.
  - Keep all invariants.
- [ ] **Step 2:** Full build + tests + baseline 5× PASS.
- [ ] **Step 3:** Commit `docs: describe redesigned UI, settings and dev-only CLI`.
