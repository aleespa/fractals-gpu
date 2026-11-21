using System.Buffers;
using System.Globalization;
using System.Runtime.InteropServices.WindowsRuntime;
using FractalExplorer.Core;
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
            SettingsHost.Visibility = Visibility.Visible;
            return;
        }
        SettingsHost.Visibility = Visibility.Collapsed;
        if (ReferenceEquals(args.SelectedItem, ColoursNavItem)) PanelContent.Content = _coloursPane;
        else if (ReferenceEquals(args.SelectedItem, FractalNavItem)) PanelContent.Content = _fractalPane;
    }

    private void NavView_PaneOpening(NavigationView sender, object args) => SidePanel.Visibility = Visibility.Visible;

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
                ErrorBar.IsOpen = false;
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
    internal void ShowError(string message, string title = "Render failed")
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
