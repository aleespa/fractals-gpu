using System.Globalization;

namespace FractalExplorer.Core;

public sealed class ViewState
{
    public FractalType Type { get; set; }
    public double X { get; set; }
    public double Y { get; set; }
    public double Zoom { get; set; }
    public int MaxIterations { get; set; }
    public double Cx { get; set; }
    public double Cy { get; set; }

    public static ViewState CreateDefault() => new()
    {
        Type = FractalType.Mandelbrot, X = -1.4002, Y = 0, Zoom = 0.10,
        MaxIterations = 400, Cx = -0.4, Cy = 0.6,
    };

    public void Reset()
    {
        var d = CreateDefault();
        X = d.X; Y = d.Y; Zoom = d.Zoom; MaxIterations = d.MaxIterations; Cx = d.Cx; Cy = d.Cy;
        if (Type == FractalType.Julia) { X = 0; Zoom = 1.5; }
    }

    public void Pan(double dx, double dy, double viewW, double viewH)
    {
        double aspect = viewW / viewH;
        X -= (dx / viewW) * (2 * Zoom);
        Y -= (dy / viewH) * (2 * Zoom / aspect);
    }

    public double ZoomAt(int wheelDelta, double pointerX, double pointerY, double viewW, double viewH)
    {
        double deltaY = -wheelDelta * 100.0 / 120.0;
        double zoomDelta = Math.Abs(deltaY) * 0.001 * 0.5 + 0.1;
        double mathFactor = deltaY > 0 ? 1 + zoomDelta : 1 - zoomDelta;
        double aspect = viewW / viewH;
        double offX = pointerX - viewW / 2;
        double offY = pointerY - viewH / 2;
        double mouseCx = X + (offX / viewW) * (2 * Zoom);
        double mouseCy = Y + (offY / viewH) * (2 * Zoom / aspect);
        Zoom *= mathFactor;
        X = mouseCx - (offX / viewW) * (2 * Zoom);
        Y = mouseCy - (offY / viewH) * (2 * Zoom / aspect);
        return 1 / mathFactor;
    }

    public string ZoomText(CultureInfo culture) =>
        Math.Round(1 / Zoom, 2).ToString("#,0.##", culture) + "x";
}
