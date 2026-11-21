using System.Buffers;
using System.Globalization;
using System.Runtime.InteropServices.WindowsRuntime;
using FractalExplorer.Core;
using FractalExplorer.Dialogs;
using FractalExplorer.Panes;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;

namespace FractalExplorer;

public sealed partial class MainWindow : Window
{
    internal ViewState View { get; } = ViewState.CreateDefault();
    internal Palette Palette { get; } = Palette.CreateDefault();
    internal RenderQueue Renderer { get; } = new();

    private readonly SettingsStore _settingsStore = new(SettingsStore.DefaultPath);

    /// <summary>Current app settings; always normalized.</summary>
    internal AppSettings Settings { get; private set; }

    private readonly PreviewTransform _preview = new();
    private readonly DispatcherQueueTimer _renderTimer;
    private readonly FractalPane _fractalPane;
    private readonly ColoursPane _coloursPane;
    private readonly SettingsView _settingsView;
    private readonly RectangleGeometry _canvasClip = new();

    private CancellationTokenSource? _renderCts;
    private WriteableBitmap? _bitmap;
    private int _viewVersion;   // bumped on every view/parameter change; stale results are discarded
    private int _renderSeq;     // id of the most recently started render
    private bool _ready;        // false while XAML initialises

    private bool _panning;
    private Windows.Foundation.Point _lastPointer;

    private const string RenderErrorTitle = "Render failed";

    private CancellationTokenSource? _saveCts;
    private bool _saving;            // a save is running (Save disabled, no second save)
    private bool _saveDialogOpen;    // only one ContentDialog may be open at a time
    private bool _saveWriting;       // the PNG is being written: progress updates ignored, Cancel disabled

    private string? _successFilePath;
    private object? _currentNavItem;   // container of the selected rail item, updated on SelectionChanged

    // NavigationView template parts used to size the side panel (null if the template changes).
    private Grid? _navItemsGrid;
    private FrameworkElement? _navMenuItems;
    private FrameworkElement? _navFooterItems;

    public MainWindow()
    {
        Settings = _settingsStore.Load().Normalized();
        InitializeComponent();
        SystemBackdrop = new MicaBackdrop();
        AppWindow.Resize(new Windows.Graphics.SizeInt32(1400, 900));

        _renderTimer = DispatcherQueue.CreateTimer();
        _renderTimer.IsRepeating = false;
        _renderTimer.Tick += (_, _) => StartRender();

        _fractalPane = new FractalPane(this);
        _coloursPane = new ColoursPane(this);
        _settingsView = new SettingsView(this);
        SettingsHost.Children.Add(_settingsView);
        ApplyTheme();

        CanvasHost.Clip = _canvasClip;
        NavView.SelectedItem = FractalNavItem;
        _currentNavItem = FractalNavItem;
        PanelContent.Content = _fractalPane;
        UpdateCoordinates();
        _ready = true;

        CanvasHost.Loaded += (_, _) => RequestRender(0);
    }

    // ── Shell ────────────────────────────────────────────────────

    private void NavView_ItemInvoked(NavigationView sender, NavigationViewItemInvokedEventArgs args)
    {
        // NavigationView updates SelectedItem before raising ItemInvoked (and SelectionChanged after it),
        // so compare against the item we last saw selected rather than sender.SelectedItem.
        bool reselected = ReferenceEquals(args.InvokedItemContainer, _currentNavItem);
        if (args.IsSettingsInvoked) return;
        if (reselected) sender.IsPaneOpen = !sender.IsPaneOpen;
        else sender.IsPaneOpen = true;
    }

    private void NavView_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        _currentNavItem = args.SelectedItemContainer;
        if (args.IsSettingsSelected)
        {
            PanelContent.Content = null;
            NavView.IsPaneOpen = false;   // the side panel has nothing to show for Settings
            SidePanel.Visibility = Visibility.Collapsed;
            _settingsView.SyncFromState();   // settings may have changed elsewhere (e.g. Save as folder)
            SettingsHost.Visibility = Visibility.Visible;
            return;
        }
        SettingsHost.Visibility = Visibility.Collapsed;
        if (ReferenceEquals(args.SelectedItem, ColoursNavItem)) PanelContent.Content = _coloursPane;
        else if (ReferenceEquals(args.SelectedItem, FractalNavItem)) PanelContent.Content = _fractalPane;
        SidePanel.Visibility = NavView.IsPaneOpen ? Visibility.Visible : Visibility.Collapsed;
    }

    // The hamburger can still open the pane on the Settings page; the side panel stays hidden there
    // (the rail labels still show), so it never appears as an empty panel.
    private void NavView_PaneOpening(NavigationView sender, object args) =>
        SidePanel.Visibility = ReferenceEquals(NavView.SelectedItem, NavView.SettingsItem)
            ? Visibility.Collapsed : Visibility.Visible;

    private void NavView_PaneClosing(NavigationView sender, NavigationViewPaneClosingEventArgs args) =>
        SidePanel.Visibility = Visibility.Collapsed;

    private void NavView_Loaded(object sender, RoutedEventArgs e)
    {
        SidePanel.Visibility = NavView.IsPaneOpen ? Visibility.Visible : Visibility.Collapsed;

        // The pane footer row is Auto-sized below a star-sized menu row, so on its own the panel would
        // sit at the bottom of the pane. Size it to fill the space between the menu items and Settings.
        _navItemsGrid = FindDescendant<Grid>(NavView, "ItemsContainerGrid");
        _navMenuItems = FindDescendant<FrameworkElement>(NavView, "MenuItemsHost");
        _navFooterItems = FindDescendant<FrameworkElement>(NavView, "FooterItemsScrollViewer");
        if (_navItemsGrid is null || _navMenuItems is null || _navFooterItems is null) return;
        _navItemsGrid.SizeChanged += (_, _) => FitSidePanel();
        _navMenuItems.SizeChanged += (_, _) => FitSidePanel();
        _navFooterItems.SizeChanged += (_, _) => FitSidePanel();
        FitSidePanel();
    }

    private void FitSidePanel()
    {
        if (_navItemsGrid is null || _navMenuItems is null || _navFooterItems is null) return;
        const double footerMargin = 4, slack = 8;
        // DesiredSize, not ActualHeight: the menu repeater is stretched to its ScrollViewer's viewport.
        double h = _navItemsGrid.ActualHeight - _navMenuItems.DesiredSize.Height - _navFooterItems.ActualHeight
                   - footerMargin - slack;
        SidePanel.Height = Math.Max(0, h);
    }

    private static T? FindDescendant<T>(DependencyObject root, string name) where T : FrameworkElement
    {
        int n = VisualTreeHelper.GetChildrenCount(root);
        for (int i = 0; i < n; i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is T fe && fe.Name == name) return fe;
            var found = FindDescendant<T>(child, name);
            if (found is not null) return found;
        }
        return null;
    }

    // ── Settings ─────────────────────────────────────────────────

    /// <summary>Applies <paramref name="change"/> to the settings, normalizes, saves and re-applies the theme.</summary>
    internal void UpdateSettings(Func<AppSettings, AppSettings> change)
    {
        Settings = change(Settings).Normalized();
        _settingsStore.Save(Settings);
        ApplyTheme();
    }

    private void ApplyTheme()
    {
        if (Content is FrameworkElement root)
            root.RequestedTheme = Settings.Theme switch
            {
                AppTheme.Light => ElementTheme.Light,
                AppTheme.Dark => ElementTheme.Dark,
                _ => ElementTheme.Default,
            };
    }

    // ── Rendering ────────────────────────────────────────────────

    /// <summary>Schedules a re-render of the current view after <paramref name="debounceMs"/> ms of quiet.</summary>
    public void RequestRender(int debounceMs)
    {
        _viewVersion++;
        _renderTimer.Stop();
        _renderTimer.Interval = TimeSpan.FromMilliseconds(Math.Max(0, debounceMs));
        _renderTimer.Start();
    }

    private async void StartRender()
    {
        if (CanvasHost.XamlRoot is null) return;
        double scale = CanvasHost.XamlRoot.RasterizationScale;
        int w = (int)Math.Round(CanvasHost.ActualWidth * scale);
        int h = (int)Math.Round(CanvasHost.ActualHeight * scale);
        if (w <= 0 || h <= 0) return;

        _renderCts?.Cancel();
        var cts = new CancellationTokenSource();
        _renderCts = cts;
        int seq = ++_renderSeq;
        int version = _viewVersion;

        var (palT, palRgb) = Palette.ToNative();
        var req = new RenderRequest(View.Type, w, h, View.MaxIterations, View.X, View.Y, View.Zoom,
            View.Cx, View.Cy, 1, palT, palRgb);

        Busy.IsActive = true;
        int size = w * h * NativeRenderer.BytesPerPixel(PixelFormat.Bgra8);
        byte[]? buffer = null;
        try
        {
            // The frame buffer is rented inside the job, i.e. only once the GPU slot is ours.
            buffer = await Renderer.RunAsync(RenderLane.Interactive, ct =>
            {
                byte[] buf = ArrayPool<byte>.Shared.Rent(size);
                bool done;
                try
                {
                    done = NativeRenderer.Render(req, PixelFormat.Bgra8, buf.AsSpan(0, size), ct);
                }
                catch
                {
                    ArrayPool<byte>.Shared.Return(buf);
                    throw;
                }
                if (done) return buf;
                ArrayPool<byte>.Shared.Return(buf);
                return null;
            }, cts.Token);

            if (buffer is not null && seq == _renderSeq && version == _viewVersion)
            {
                if (_bitmap is null || _bitmap.PixelWidth != w || _bitmap.PixelHeight != h)
                    _bitmap = new WriteableBitmap(w, h);
                using (var stream = _bitmap.PixelBuffer.AsStream())
                    stream.Write(buffer, 0, size);
                _bitmap.Invalidate();
                FractalImage.Source = _bitmap;
                _preview.Reset();
                ApplyPreview();
                if (ErrorBar.Title == RenderErrorTitle) ErrorBar.IsOpen = false;   // keep save errors visible
            }
        }
        catch (OperationCanceledException)
        {
            // Superseded or cancelled render (incl. TaskCanceledException from the queue): ignore.
        }
        catch (Exception ex)
        {
            // FractalRenderException and anything else; async void must not let exceptions escape.
            if (seq == _renderSeq) ShowError(ex.Message);
        }
        finally
        {
            if (buffer is not null) ArrayPool<byte>.Shared.Return(buffer);
            if (seq == _renderSeq) Busy.IsActive = false;
            if (ReferenceEquals(_renderCts, cts)) _renderCts = null;
            cts.Dispose();
        }
    }

    /// <summary>Shows <paramref name="message"/> in the error InfoBar.</summary>
    internal void ShowError(string message, string title = RenderErrorTitle)
    {
        ErrorBar.Title = title;
        ErrorBar.Message = message;
        ErrorBar.IsOpen = true;
    }

    /// <summary>
    /// Shows <paramref name="message"/> in the success InfoBar. When <paramref name="openFolderPath"/>
    /// (a file path) is given, an "Open folder" link selects that file in Explorer.
    /// </summary>
    internal void ShowSuccess(string message, string? openFolderPath)
    {
        _successFilePath = openFolderPath;
        OpenFolderButton.Visibility = openFolderPath is null ? Visibility.Collapsed : Visibility.Visible;
        SuccessBar.Message = message;
        SuccessBar.IsOpen = true;
    }

    private void OpenFolderButton_Click(object sender, RoutedEventArgs e)
    {
        if (_successFilePath is null) return;
        try
        {
            System.Diagnostics.Process.Start("explorer.exe", $"/select,\"{_successFilePath}\"");
        }
        catch (Exception ex)
        {
            ShowError(ex.Message, "Could not open folder");
        }
    }

    /// <summary>Shows <paramref name="title"/> in the informational InfoBar.</summary>
    private void ShowInfo(string title)
    {
        InfoMessageBar.Title = title;
        InfoMessageBar.IsOpen = true;
    }

    // ── Saving ───────────────────────────────────────────────────

    private void SaveButton_Click(SplitButton sender, SplitButtonClickEventArgs args) => QuickSave();

    private void SaveAsItem_Click(object sender, RoutedEventArgs e) => SaveAs();

    private void QuickSaveAccelerator_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        QuickSave();
    }

    private void SaveAsAccelerator_Invoked(KeyboardAccelerator sender, KeyboardAcceleratorInvokedEventArgs args)
    {
        args.Handled = true;
        SaveAs();
    }

    private void CancelSave_Click(object sender, RoutedEventArgs e) => _saveCts?.Cancel();

    // Keep the top bars (progress strip, InfoBars) left of the Save button so they never overlap it.
    private void SaveHost_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        double right = e.NewSize.Width + SaveHost.Margin.Right + 16;
        TopBars.Margin = new Thickness(16, 16, right, 0);
    }

    /// <summary>Saves with the Settings defaults to a unique file in the save folder.</summary>
    private void QuickSave()
    {
        if (_saving || _saveDialogOpen) return;
        string path;
        try
        {
            path = SavePlanner.QuickSavePath(Settings, View.Type, DateTime.Now);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException
                                       or NotSupportedException)
        {
            ShowError($"Couldn't use the save folder: {ex.Message.TrimEnd('.', ' ')}. "
                      + "Try Save as… or change the folder in Settings.", "Save failed");
            return;
        }
        catch (Exception ex)
        {
            ShowError(ex.Message, "Save failed");
            return;
        }
        var spec = new SaveSpec(Settings.SaveWidth, Settings.SaveHeight,
            ExportMath.SupersampleForIndex(Settings.SaveSmoothingIndex), View.MaxIterations, path);
        RunSave(spec);
    }

    /// <summary>Opens the Save as dialog and, if confirmed, saves to the chosen file.</summary>
    private async void SaveAs()
    {
        if (_saving || _saveDialogOpen || Content?.XamlRoot is null) return;
        _saveDialogOpen = true;
        try
        {
            var dialog = new SaveAsDialog(Settings, View.Type, View.MaxIterations,
                WinRT.Interop.WindowNative.GetWindowHandle(this))
            {
                XamlRoot = Content.XamlRoot,
                RequestedTheme = (Content as FrameworkElement)?.RequestedTheme ?? ElementTheme.Default,
            };
            await dialog.ShowAsync();
            if (dialog.Result is { } spec) RunSave(spec);
        }
        catch (Exception ex)
        {
            ShowError(ex.Message, "Save failed");
        }
        finally
        {
            _saveDialogOpen = false;
        }
    }

    /// <summary>
    /// Renders <paramref name="spec"/> on the Save lane with a progress strip and Cancel, then writes the PNG
    /// to a temp file next to the target and moves it into place. On cancel or failure the temp file is
    /// deleted, and the target only if it is the save picker's 0-byte placeholder; a pre-existing file the
    /// user chose to replace survives any failure.
    /// </summary>
    private async void RunSave(SaveSpec spec)
    {
        if (_saving) return;
        _saving = true;
        SaveButton.IsEnabled = false;
        var cts = new CancellationTokenSource();
        _saveCts = cts;

        string path = spec.Path;
        int w = spec.Width, h = spec.Height;
        bool written = false;
        bool placeholder = IsEmptyFile(path);   // the save picker creates a 0-byte file for a new name
        string? tmp = null;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            // Snapshot everything the job needs on the UI thread.
            double viewAspect = CanvasHost.ActualHeight > 0 ? CanvasHost.ActualWidth / CanvasHost.ActualHeight : 0;
            var req = SavePlanner.BuildRequest(View, Palette.ToNative(), spec, viewAspect);
            var dq = DispatcherQueue;
            tmp = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".",
                $"{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
            string tmpPath = tmp;

            SuccessBar.IsOpen = false;
            InfoMessageBar.IsOpen = false;
            if (ErrorBar.Title == "Save failed") ErrorBar.IsOpen = false;   // a retry must not show both bars
            _saveWriting = false;
            SaveProgress.IsIndeterminate = false;
            CancelSaveButton.IsEnabled = true;
            UpdateSaveProgress(cts, 0, w, h);
            SaveStrip.Visibility = Visibility.Visible;

            int lastPct = -1;   // touched only by the render thread
            bool done = await Renderer.RunAsync(RenderLane.Save, ct =>
            {
                byte[] rgb = GC.AllocateUninitializedArray<byte>(checked(w * h * 3));
                // Runs on the render thread with the native render lock held: only post to the UI thread.
                bool completed = NativeRenderer.Render(req, PixelFormat.Rgb8, rgb, p =>
                {
                    int pct = (int)Math.Clamp(p * 100, 0, 100);
                    if (pct == lastPct) return;
                    lastPct = pct;
                    dq.TryEnqueue(() => UpdateSaveProgress(cts, pct, w, h));
                }, ct);
                // Cancel pressed after the last band: still a cancel, nothing is written.
                if (!completed || ct.IsCancellationRequested) return false;
                // Encoding cannot be interrupted, so the strip switches to an indeterminate,
                // non-cancellable "Writing PNG…" state.
                dq.TryEnqueue(() => ShowSaveWriting(cts));
                NativeRenderer.WritePng(tmpPath, w, h, rgb);
                File.Move(tmpPath, path, overwrite: true);
                return true;
            }, cts.Token);

            if (done)
            {
                written = true;
                ShowSuccess(string.Create(CultureInfo.InvariantCulture,
                    $"Saved {Path.GetFileName(path)} ({sw.Elapsed.TotalSeconds:F1}s)"), path);
            }
            else
            {
                ShowInfo("Save cancelled");
            }
        }
        catch (OperationCanceledException)
        {
            ShowInfo("Save cancelled");   // cancelled while waiting for the GPU
        }
        catch (OutOfMemoryException)
        {
            ShowError($"Not enough memory for a {w} × {h} save.", "Save failed");
        }
        catch (Exception ex)
        {
            ShowError(ex.Message, "Save failed");
        }
        finally
        {
            if (!written) CleanUpFailedSave(path, tmp, placeholder);
            SaveStrip.Visibility = Visibility.Collapsed;
            SaveProgress.IsIndeterminate = false;
            CancelSaveButton.IsEnabled = true;
            _saveWriting = false;
            SaveButton.IsEnabled = true;
            _saving = false;
            if (ReferenceEquals(_saveCts, cts)) _saveCts = null;
            cts.Dispose();
        }
    }

    private void UpdateSaveProgress(CancellationTokenSource cts, int pct, int w, int h)
    {
        if (!ReferenceEquals(_saveCts, cts) || _saveWriting) return;   // late update from a finished render
        SaveProgress.Value = pct;
        SaveProgressText.Text = string.Create(CultureInfo.InvariantCulture, $"Saving {w} × {h}… {pct}%");
    }

    private void ShowSaveWriting(CancellationTokenSource cts)
    {
        if (!ReferenceEquals(_saveCts, cts)) return;   // the save already finished
        _saveWriting = true;
        SaveProgress.IsIndeterminate = true;
        CancelSaveButton.IsEnabled = false;
        SaveProgressText.Text = "Writing PNG…";
    }

    private static bool IsEmptyFile(string path)
    {
        try
        {
            var fi = new FileInfo(path);
            return fi.Exists && fi.Length == 0;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Best-effort cleanup after a cancelled or failed save: deletes the temp file, and the target only
    /// if it was the save picker's 0-byte placeholder (still empty). Never throws.
    /// </summary>
    private static void CleanUpFailedSave(string path, string? tmp, bool placeholder)
    {
        try
        {
            if (tmp is not null && File.Exists(tmp)) File.Delete(tmp);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Could not delete temp file '{tmp}': {ex.Message}");
        }
        if (placeholder && IsEmptyFile(path))
        {
            try
            {
                File.Delete(path);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Could not delete placeholder '{path}': {ex.Message}");
            }
        }
    }

    private void ApplyPreview()
    {
        PreviewXform.CenterX = CanvasHost.ActualWidth / 2;
        PreviewXform.CenterY = CanvasHost.ActualHeight / 2;
        PreviewXform.TranslateX = _preview.TranslateX;
        PreviewXform.TranslateY = _preview.TranslateY;
        PreviewXform.ScaleX = _preview.Scale;
        PreviewXform.ScaleY = _preview.Scale;
    }

    // ── Pan / zoom ───────────────────────────────────────────────

    private void CanvasHost_PointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(CanvasHost);
        if (e.Pointer.PointerDeviceType == PointerDeviceType.Mouse && !pt.Properties.IsLeftButtonPressed) return;
        if (CanvasHost.CapturePointer(e.Pointer))
        {
            _panning = true;
            _lastPointer = pt.Position;
            e.Handled = true;
        }
    }

    private void CanvasHost_PointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (!_panning) return;
        var p = e.GetCurrentPoint(CanvasHost).Position;
        double dx = p.X - _lastPointer.X, dy = p.Y - _lastPointer.Y;
        _lastPointer = p;
        if (dx == 0 && dy == 0) return;

        View.Pan(dx, dy, CanvasHost.ActualWidth, CanvasHost.ActualHeight);
        _preview.ApplyPan(dx, dy);
        _viewVersion++;
        ApplyPreview();
        UpdateCoordinates();
        e.Handled = true;
    }

    private void CanvasHost_PointerReleased(object sender, PointerRoutedEventArgs e)
    {
        if (!_panning) return;
        CanvasHost.ReleasePointerCapture(e.Pointer);
        EndPan();
        e.Handled = true;
    }

    private void CanvasHost_PointerCaptureLost(object sender, PointerRoutedEventArgs e) => EndPan();

    private void EndPan()
    {
        if (!_panning) return;
        _panning = false;
        RequestRender(0);
    }

    private void CanvasHost_PointerWheelChanged(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(CanvasHost);
        int delta = pt.Properties.MouseWheelDelta;
        if (delta == 0) return;
        double w = CanvasHost.ActualWidth, h = CanvasHost.ActualHeight;
        if (w <= 0 || h <= 0) return;

        double factor = View.ZoomAt(delta, pt.Position.X, pt.Position.Y, w, h);
        _preview.ApplyZoom(pt.Position.X - w / 2, pt.Position.Y - h / 2, factor);
        ApplyPreview();
        UpdateCoordinates();
        RequestRender(150);
        e.Handled = true;
    }

    private void CanvasHost_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        _canvasClip.Rect = new Windows.Foundation.Rect(0, 0, e.NewSize.Width, e.NewSize.Height);
        ApplyPreview();
        if (_ready) RequestRender(500);
    }

    // ── State ────────────────────────────────────────────────────

    /// <summary>Resets view and palette to the v0.1.0 defaults and re-renders.</summary>
    internal void ResetView()
    {
        View.Reset();
        Palette.ResetToDefault();
        _preview.Reset();
        ApplyPreview();
        _fractalPane.SyncFromState();
        _coloursPane.SyncFromState();
        UpdateCoordinates();
        RequestRender(0);
    }

    private void UpdateCoordinates()
    {
        CoordsText.Text = string.Create(CultureInfo.InvariantCulture, $"x {View.X:F7}   y {View.Y:F7}   zoom ")
                          + View.ZoomText(CultureInfo.CurrentCulture);
    }
}
