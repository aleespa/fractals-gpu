namespace FractalExplorer.Core;

public sealed record SaveSpec(int Width, int Height, int Supersample, int MaxIterations, string Path);

public static class SavePlanner
{
    public static string UniquePath(string folder, string fileName)
    {
        var p = Path.Combine(folder, fileName);
        if (!File.Exists(p)) return p;
        var name = Path.GetFileNameWithoutExtension(fileName);
        var ext = Path.GetExtension(fileName);
        for (int i = 2; ; i++)
        {
            p = Path.Combine(folder, $"{name} ({i}){ext}");
            if (!File.Exists(p)) return p;
        }
    }

    public static string QuickSavePath(AppSettings s, FractalType type, DateTime local)
    {
        Directory.CreateDirectory(s.SaveFolder);
        return UniquePath(s.SaveFolder, ExportMath.DefaultFileName(type, s.SaveWidth, s.SaveHeight, local));
    }

    public static RenderRequest BuildRequest(ViewState view, (float[] T, float[] Rgb) palette, SaveSpec spec, double viewAspect)
    {
        double aspect = viewAspect <= 0 ? spec.Width / (double)spec.Height : viewAspect;
        double zoom = ExportMath.AdjustZoomForAspect(view.Zoom, spec.Width, spec.Height, aspect);
        return new RenderRequest(view.Type, spec.Width, spec.Height, spec.MaxIterations,
            view.X, view.Y, zoom, view.Cx, view.Cy, spec.Supersample, palette.T, palette.Rgb);
    }
}
