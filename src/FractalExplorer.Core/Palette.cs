using System.Globalization;

namespace FractalExplorer.Core;

public sealed class PaletteStop
{
    public double T { get; internal set; }
    public uint Rgb { get; internal set; }
    public bool IsPinned => T == 0.0 || T == 1.0;
}

public sealed class Palette
{
    public const double MinMovableT = 0.001, MaxMovableT = 0.999;

    private readonly List<PaletteStop> _stops = new();

    public event EventHandler? Changed;

    public IReadOnlyList<PaletteStop> Stops => _stops;

    private Palette() { }

    public static Palette CreateDefault()
    {
        var p = new Palette();
        p.Fill();
        return p;
    }

    private void Fill()
    {
        _stops.Clear();
        _stops.Add(new PaletteStop { T = 0.0, Rgb = 0x000000 });
        _stops.Add(new PaletteStop { T = 0.25, Rgb = 0x2130C8 });
        _stops.Add(new PaletteStop { T = 0.5, Rgb = 0x8C0021 });
        _stops.Add(new PaletteStop { T = 0.75, Rgb = 0xFF9900 });
        _stops.Add(new PaletteStop { T = 1.0, Rgb = 0xFFFFFF });
    }

    private static double Clamp(double t) => Math.Max(MinMovableT, Math.Min(MaxMovableT, t));

    private void Sort() => _stops.Sort((a, b) => a.T.CompareTo(b.T));

    public uint Sample(double t)
    {
        t = Math.Max(0, Math.Min(1, t));
        int i = 0;
        for (int k = 0; k < _stops.Count - 1; k++)
            if (_stops[k].T <= t) i = k;
        int j = Math.Min(i + 1, _stops.Count - 1);
        double span = _stops[j].T - _stops[i].T;
        double frac = span > 0 ? (t - _stops[i].T) / span : 0;
        uint a = _stops[i].Rgb, b = _stops[j].Rgb;
        uint Ch(int shift)
        {
            double x = (a >> shift) & 0xFF, y = (b >> shift) & 0xFF;
            return (uint)Math.Round(x + frac * (y - x), MidpointRounding.AwayFromZero);
        }
        return (Ch(16) << 16) | (Ch(8) << 8) | Ch(0);
    }

    public PaletteStop AddAt(double t)
    {
        t = Clamp(t);
        var s = new PaletteStop { T = t, Rgb = Sample(t) };
        _stops.Add(s);
        Sort();
        Changed?.Invoke(this, EventArgs.Empty);
        return s;
    }

    public void Move(PaletteStop stop, double t)
    {
        if (stop.IsPinned || !_stops.Contains(stop)) return;
        stop.T = Clamp(t);
        Sort();
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public bool Remove(PaletteStop stop)
    {
        if (stop.IsPinned || _stops.Count <= 2 || !_stops.Remove(stop)) return false;
        Changed?.Invoke(this, EventArgs.Empty);
        return true;
    }

    public void SetColor(PaletteStop stop, uint rgb)
    {
        stop.Rgb = rgb & 0xFFFFFF;
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void ResetToDefault()
    {
        Fill();
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public (float[] T, float[] Rgb) ToNative()
    {
        var t = new float[_stops.Count];
        var rgb = new float[_stops.Count * 3];
        for (int i = 0; i < _stops.Count; i++)
        {
            t[i] = (float)Math.Round(_stops[i].T, 6);
            uint c = _stops[i].Rgb;
            rgb[i * 3] = ((c >> 16) & 0xFF) / 255f;
            rgb[i * 3 + 1] = ((c >> 8) & 0xFF) / 255f;
            rgb[i * 3 + 2] = (c & 0xFF) / 255f;
        }
        return (t, rgb);
    }

    public IReadOnlyList<string> ToCliArgs()
    {
        var r = new List<string>(_stops.Count * 2);
        foreach (var s in _stops)
        {
            r.Add(s.T.ToString("F6", CultureInfo.InvariantCulture));
            r.Add(s.Rgb.ToString("X6", CultureInfo.InvariantCulture));
        }
        return r;
    }
}
