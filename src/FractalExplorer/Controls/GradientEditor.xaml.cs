using FractalExplorer.Core;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using Windows.UI;

namespace FractalExplorer.Controls;

public sealed partial class GradientEditor : UserControl
{
    private const double HandleSize = 14;
    private const double DragThreshold = 5;

    private Palette? _palette;
    private readonly Dictionary<PaletteStop, Ellipse> _handles = new();

    private PaletteStop? _pressStop;
    private Windows.Foundation.Point _pressPoint;
    private bool _dragging;

    public GradientEditor()
    {
        InitializeComponent();
    }

    /// <summary>Raised while the colour picker edits a stop.</summary>
    public event EventHandler? LivePreviewChanged;

    /// <summary>Raised after a handle drag finishes.</summary>
    public event EventHandler? DragCompleted;

    /// <summary>True while a handle is being dragged.</summary>
    public bool IsDragging => _dragging;

    public Palette? Palette
    {
        get => _palette;
        set
        {
            if (_palette is not null) _palette.Changed -= OnPaletteChanged;
            _palette = value;
            if (_palette is not null) _palette.Changed += OnPaletteChanged;
            Relayout();
        }
    }

    private void OnPaletteChanged(object? sender, EventArgs e) => Relayout();

    private void Host_SizeChanged(object sender, SizeChangedEventArgs e) => Relayout();

    private static Color ToColor(uint rgb) =>
        Color.FromArgb(255, (byte)(rgb >> 16), (byte)(rgb >> 8), (byte)rgb);

    private static uint ToRgb(Color c) => ((uint)c.R << 16) | ((uint)c.G << 8) | c.B;

    private void Relayout()
    {
        if (_palette is null) return;
        double w = Host.ActualWidth;

        var brush = new LinearGradientBrush { StartPoint = new(0, 0), EndPoint = new(1, 0) };
        foreach (var s in _palette.Stops)
            brush.GradientStops.Add(new GradientStop { Offset = s.T, Color = ToColor(s.Rgb) });
        Bar.Fill = brush;

        bool sameSet = _handles.Count == _palette.Stops.Count && _palette.Stops.All(_handles.ContainsKey);
        if (!sameSet)
        {
            HandleCanvas.Children.Clear();
            _handles.Clear();
            foreach (var s in _palette.Stops)
            {
                var h = new Ellipse
                {
                    Width = HandleSize,
                    Height = HandleSize,
                    Fill = new SolidColorBrush(ToColor(s.Rgb)),
                    Stroke = new SolidColorBrush(Colors.White),
                    StrokeThickness = 2,
                    Tag = s,
                };
                h.PointerPressed += Handle_PointerPressed;
                h.RightTapped += Handle_RightTapped;
                HandleCanvas.Children.Add(h);
                _handles[s] = h;
            }
        }

        foreach (var (s, h) in _handles)
        {
            ((SolidColorBrush)h.Fill).Color = ToColor(s.Rgb);
            Canvas.SetLeft(h, s.T * w - HandleSize / 2);
            Canvas.SetTop(h, (Host.ActualHeight - HandleSize) / 2);
        }
    }

    private void Canvas_PointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(HandleCanvas);
        if (!pt.Properties.IsLeftButtonPressed || _palette is null || Host.ActualWidth <= 0) return;
        _palette.AddAt(pt.Position.X / Host.ActualWidth);
        e.Handled = true;
    }

    private void Handle_PointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(HandleCanvas);
        if (!pt.Properties.IsLeftButtonPressed) return;
        _pressStop = (PaletteStop)((FrameworkElement)sender).Tag;
        _pressPoint = pt.Position;
        _dragging = false;
        HandleCanvas.CapturePointer(e.Pointer);
        e.Handled = true;
    }

    private void Canvas_PointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (_pressStop is null || _palette is null) return;
        if (!e.GetCurrentPoint(HandleCanvas).Properties.IsLeftButtonPressed) return;
        var p = e.GetCurrentPoint(HandleCanvas).Position;
        if (!_dragging)
        {
            if (Math.Abs(p.X - _pressPoint.X) <= DragThreshold && Math.Abs(p.Y - _pressPoint.Y) <= DragThreshold) return;
            _dragging = true;
        }
        if (Host.ActualWidth > 0) _palette.Move(_pressStop, p.X / Host.ActualWidth);
        e.Handled = true;
    }

    private void Canvas_PointerReleased(object sender, PointerRoutedEventArgs e)
    {
        if (_pressStop is null) return;
        var stop = _pressStop;
        bool wasDrag = _dragging;
        EndPress(e.Pointer);
        if (wasDrag) DragCompleted?.Invoke(this, EventArgs.Empty);
        else ShowPicker(stop);
        e.Handled = true;
    }

    private void Canvas_PointerCaptureLost(object sender, PointerRoutedEventArgs e)
    {
        if (_pressStop is null) return;
        bool wasDrag = _dragging;
        EndPress(e.Pointer);
        if (wasDrag) DragCompleted?.Invoke(this, EventArgs.Empty);
    }

    private void Canvas_PointerCanceled(object sender, PointerRoutedEventArgs e)
    {
        bool wasDrag = _dragging;
        EndPress(e.Pointer);
        if (wasDrag) DragCompleted?.Invoke(this, EventArgs.Empty);
    }

    private void EndPress(Microsoft.UI.Xaml.Input.Pointer pointer)
    {
        _pressStop = null;
        _dragging = false;
        HandleCanvas.ReleasePointerCapture(pointer);
    }

    private void Handle_RightTapped(object sender, RightTappedRoutedEventArgs e)
    {
        var stop = (PaletteStop)((FrameworkElement)sender).Tag;
        _palette?.Remove(stop);
        e.Handled = true;
    }

    private void ShowPicker(PaletteStop stop)
    {
        if (_palette is null || !_handles.TryGetValue(stop, out var handle)) return;
        var picker = new ColorPicker
        {
            IsAlphaEnabled = false,
            Color = ToColor(stop.Rgb),
        };
        picker.ColorChanged += (_, args) =>
        {
            if (_palette is null) return;
            _palette.SetColor(stop, ToRgb(args.NewColor));
            LivePreviewChanged?.Invoke(this, EventArgs.Empty);
        };
        var flyout = new Flyout { Content = picker };
        flyout.ShowAt(handle);
    }
}
