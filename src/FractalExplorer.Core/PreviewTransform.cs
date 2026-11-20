namespace FractalExplorer.Core;

public sealed class PreviewTransform
{
    public double TranslateX { get; private set; }
    public double TranslateY { get; private set; }
    public double Scale { get; private set; } = 1;

    public void ApplyPan(double dx, double dy) { TranslateX += dx; TranslateY += dy; }

    public void ApplyZoom(double offsetX, double offsetY, double factor)
    {
        TranslateX = offsetX - (offsetX - TranslateX) * factor;
        TranslateY = offsetY - (offsetY - TranslateY) * factor;
        Scale *= factor;
    }

    public void Reset() { TranslateX = 0; TranslateY = 0; Scale = 1; }
}
