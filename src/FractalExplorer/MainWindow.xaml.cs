using System.Buffers;
using System.Globalization;
using System.Runtime.InteropServices.WindowsRuntime;
using FractalExplorer.Core;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;

namespace FractalExplorer;

public sealed partial class MainWindow : Window
{
    internal readonly ViewState _view = ViewState.CreateDefault();
    internal readonly Palette _palette = Palette.CreateDefault();

    private readonly PreviewTransform _preview = new();
    private readonly DispatcherQueueTimer _renderTimer;

    private CancellationTokenSource? _renderCts;
    private WriteableBitmap? _bitmap;
    private int _viewVersion;   // bumped on every view/parameter change; stale results are discarded
    private int _renderSeq;     // id of the most recently started render
    private bool _ready;        // false while XAML initialises / controls are being synced

    private bool _panning;
    private Windows.Foundation.Point _lastPointer;

    public MainWindow()
    {
        InitializeComponent();
        SystemBackdrop = new MicaBackdrop();
        AppWindow.Resize(new Windows.Graphics.SizeInt32(1400, 900));

        _renderTimer = DispatcherQueue.CreateTimer();
        _renderTimer.IsRepeating = false;
        _renderTimer.Tick += (_, _) => StartRender();

        PaletteEditor.Palette = _palette;
        _palette.Changed += (_, _) =>
        {
            if (!PaletteEditor.IsDragging) RequestRender(150);
        };
        PaletteEditor.DragCompleted += (_, _) => RequestRender(0);

        SyncControls();
        _ready = true;

        RootGrid.Loaded += (_, _) => RequestRender(0);
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
        if (ViewArea.XamlRoot is null) return;
        double scale = ViewArea.XamlRoot.RasterizationScale;
        int w = (int)Math.Round(ViewArea.ActualWidth * scale);
        int h = (int)Math.Round(ViewArea.ActualHeight * scale);
        if (w <= 0 || h <= 0) return;

        _renderCts?.Cancel();
        var cts = new CancellationTokenSource();
        _renderCts = cts;
        int seq = ++_renderSeq;
        int version = _viewVersion;

        var (palT, palRgb) = _palette.ToNative();
        var req = new RenderRequest(_view.Type, w, h, _view.MaxIterations, _view.X, _view.Y, _view.Zoom,
            _view.Cx, _view.Cy, 1, palT, palRgb);

        Busy.IsActive = true;
        int size = w * h * NativeRenderer.BytesPerPixel(PixelFormat.Bgra8);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(size);
        try
        {
            bool done = await Task.Run(() => NativeRenderer.Render(req, PixelFormat.Bgra8, buffer.AsSpan(0, size), cts.Token));
            if (done && seq == _renderSeq && version == _viewVersion)
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
        catch (FractalRenderException ex)
        {
            if (seq == _renderSeq)
            {
                ErrorBar.Message = ex.Message;
                ErrorBar.IsOpen = true;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
            if (seq == _renderSeq) Busy.IsActive = false;
            if (ReferenceEquals(_renderCts, cts)) _renderCts = null;
            cts.Dispose();
        }
    }

    private void ApplyPreview()
    {
        PreviewXform.CenterX = ViewArea.ActualWidth / 2;
        PreviewXform.CenterY = ViewArea.ActualHeight / 2;
        PreviewXform.TranslateX = _preview.TranslateX;
        PreviewXform.TranslateY = _preview.TranslateY;
        PreviewXform.ScaleX = _preview.Scale;
        PreviewXform.ScaleY = _preview.Scale;
    }

    // ── Pan / zoom ───────────────────────────────────────────────

    private void ViewArea_PointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(ViewArea);
        if (e.Pointer.PointerDeviceType == PointerDeviceType.Mouse && !pt.Properties.IsLeftButtonPressed) return;
        if (ViewArea.CapturePointer(e.Pointer))
        {
            _panning = true;
            _lastPointer = pt.Position;
            e.Handled = true;
        }
    }

    private void ViewArea_PointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (!_panning) return;
        var p = e.GetCurrentPoint(ViewArea).Position;
        double dx = p.X - _lastPointer.X, dy = p.Y - _lastPointer.Y;
        _lastPointer = p;
        if (dx == 0 && dy == 0) return;

        _view.Pan(dx, dy, ViewArea.ActualWidth, ViewArea.ActualHeight);
        _preview.ApplyPan(dx, dy);
        _viewVersion++;
        ApplyPreview();
        UpdateStats();
        e.Handled = true;
    }

    private void ViewArea_PointerReleased(object sender, PointerRoutedEventArgs e)
    {
        if (!_panning) return;
        ViewArea.ReleasePointerCapture(e.Pointer);
        EndPan();
        e.Handled = true;
    }

    private void ViewArea_PointerCaptureLost(object sender, PointerRoutedEventArgs e) => EndPan();

    private void EndPan()
    {
        if (!_panning) return;
        _panning = false;
        RequestRender(0);
    }

    private void ViewArea_PointerWheelChanged(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(ViewArea);
        int delta = pt.Properties.MouseWheelDelta;
        if (delta == 0) return;
        double w = ViewArea.ActualWidth, h = ViewArea.ActualHeight;
        if (w <= 0 || h <= 0) return;

        double factor = _view.ZoomAt(delta, pt.Position.X, pt.Position.Y, w, h);
        _preview.ApplyZoom(pt.Position.X - w / 2, pt.Position.Y - h / 2, factor);
        ApplyPreview();
        UpdateStats();
        RequestRender(150);
        e.Handled = true;
    }

    private void ViewArea_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        ApplyPreview();
        if (_ready) RequestRender(500);
    }

    // ── Controls ─────────────────────────────────────────────────

    private void TypeCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_ready) return;
        _view.Type = TypeCombo.SelectedIndex == 1 ? FractalType.Julia : FractalType.Mandelbrot;
        JuliaControls.Visibility = _view.Type == FractalType.Julia ? Visibility.Visible : Visibility.Collapsed;
        RequestRender(150);
    }

    private void IterSlider_ValueChanged(object sender, RangeBaseValueChangedEventArgs e)
    {
        if (!_ready) return;
        _view.MaxIterations = (int)Math.Round(IterSlider.Value);
        UpdateLabels();
        RequestRender(150);
    }

    private void JuliaSlider_ValueChanged(object sender, RangeBaseValueChangedEventArgs e)
    {
        if (!_ready) return;
        _view.Cx = CxSlider.Value;
        _view.Cy = CySlider.Value;
        UpdateLabels();
        RequestRender(150);
    }

    private void ResetButton_Click(object sender, RoutedEventArgs e)
    {
        _view.Reset();
        _palette.ResetToDefault();
        _preview.Reset();
        ApplyPreview();
        SyncControls();
        RequestRender(0);
    }

    // ── Export / CLI ─────────────────────────────────────────────

    private void ExportSizeCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (ExportCustomPanel is null) return;
        ExportCustomPanel.Visibility = ExportSizeCombo.SelectedIndex == 2 ? Visibility.Visible : Visibility.Collapsed;
    }

    private void SetExportStatus(string text, bool error)
    {
        ExportStatus.Text = text;
        ExportStatus.Foreground = error
            ? new SolidColorBrush(Microsoft.UI.Colors.Red)
            : (Brush)Application.Current.Resources["TextFillColorPrimaryBrush"];
        ExportStatus.Visibility = Visibility.Visible;
    }

    private async void ExportButton_Click(object sender, RoutedEventArgs e)
    {
        int w, h;
        switch (ExportSizeCombo.SelectedIndex)
        {
            case 1: w = 7680; h = 4320; break;
            case 2:
                w = double.IsNaN(ExportWidthBox.Value) ? 0 : (int)Math.Round(ExportWidthBox.Value);
                h = double.IsNaN(ExportHeightBox.Value) ? 0 : (int)Math.Round(ExportHeightBox.Value);
                break;
            default: w = 3840; h = 2160; break;
        }
        if (!ExportMath.IsValidSize(w, h))
        {
            SetExportStatus($"Width and height must be between {ExportMath.MinSize} and {ExportMath.MaxSize}.", true);
            return;
        }

        var picker = new Windows.Storage.Pickers.FileSavePicker();
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
        picker.FileTypeChoices.Add("PNG image", new List<string> { ".png" });
        picker.SuggestedFileName = Path.GetFileNameWithoutExtension(
            ExportMath.DefaultFileName(_view.Type, w, h, DateTime.Now));
        var file = await picker.PickSaveFileAsync();
        if (file is null) return;

        ExportButton.IsEnabled = false;
        ExportButton.Content = "Rendering…";
        SetExportStatus($"Rendering {w} × {h}…", false);

        // Snapshot UI state before leaving the UI thread.
        double viewAspect = ViewArea.ActualWidth / Math.Max(1, ViewArea.ActualHeight);
        double zoom = ExportMath.AdjustZoomForAspect(_view.Zoom, w, h, viewAspect);
        int ss = ExportMath.SupersampleForIndex(ExportSmoothCombo.SelectedIndex);
        int iter = double.IsNaN(ExportIterBox.Value) || ExportIterBox.Value <= 0
            ? _view.MaxIterations : (int)Math.Round(ExportIterBox.Value);
        var (palT, palRgb) = _palette.ToNative();
        var req = new RenderRequest(_view.Type, w, h, iter, _view.X, _view.Y, zoom, _view.Cx, _view.Cy, ss, palT, palRgb);
        string path = file.Path;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            await Task.Run(() =>
            {
                var rgb = GC.AllocateUninitializedArray<byte>(checked(w * h * 3));
                NativeRenderer.Render(req, PixelFormat.Rgb8, rgb, CancellationToken.None);
                NativeRenderer.WritePng(path, w, h, rgb);
            });
            SetExportStatus(string.Create(CultureInfo.InvariantCulture,
                $"Saved {file.Name} ({sw.Elapsed.TotalSeconds:F1}s)"), false);
        }
        catch (OutOfMemoryException)
        {
            SetExportStatus($"Not enough memory for a {w} × {h} export.", true);
        }
        catch (Exception ex) when (ex is FractalRenderException or ArgumentException
                                       or IOException or UnauthorizedAccessException)
        {
            SetExportStatus(ex.Message, true);
        }
        finally
        {
            ExportButton.Content = "Export PNG";
            ExportButton.IsEnabled = true;
        }
    }

    private async void CopyCliButton_Click(object sender, RoutedEventArgs e)
    {
        string cmd = CliCommand.Build(Path.Combine(AppContext.BaseDirectory, "CppFractals.exe"),
            _view, _palette, DateTime.UtcNow);
        var dp = new Windows.ApplicationModel.DataTransfer.DataPackage();
        dp.SetText(cmd);
        Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(dp);
        CopyCliButton.Content = "Copied!";
        await Task.Delay(1500);
        CopyCliButton.Content = "Copy CLI Command";
    }

    /// <summary>Pushes <see cref="_view"/> into every control without triggering change handlers.</summary>
    private void SyncControls()
    {
        bool wasReady = _ready;
        _ready = false;
        try
        {
            TypeCombo.SelectedIndex = (int)_view.Type;
            JuliaControls.Visibility = _view.Type == FractalType.Julia ? Visibility.Visible : Visibility.Collapsed;
            IterSlider.Value = _view.MaxIterations;
            CxSlider.Value = _view.Cx;
            CySlider.Value = _view.Cy;
            UpdateLabels();
            UpdateStats();
        }
        finally
        {
            _ready = wasReady;
        }
    }

    private void UpdateLabels()
    {
        var inv = CultureInfo.InvariantCulture;
        IterLabel.Text = "Max Iterations: " + _view.MaxIterations.ToString(inv);
        CxLabel.Text = "Cx: " + _view.Cx.ToString("F3", inv);
        CyLabel.Text = "Cy: " + _view.Cy.ToString("F3", inv);
    }

    private void UpdateStats()
    {
        var inv = CultureInfo.InvariantCulture;
        StatX.Text = _view.X.ToString("F6", inv);
        StatY.Text = _view.Y.ToString("F6", inv);
        StatZoom.Text = _view.ZoomText(CultureInfo.CurrentCulture);
    }
}
