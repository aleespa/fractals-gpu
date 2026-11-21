using Microsoft.UI.Xaml.Controls;

namespace FractalExplorer.Panes;

/// <summary>Side panel for the "Colours" rail item: gradient editor and (reserved) presets.</summary>
public sealed partial class ColoursPane : UserControl
{
    private readonly MainWindow _owner;

    public ColoursPane(MainWindow owner)
    {
        _owner = owner;
        InitializeComponent();

        PaletteEditor.Palette = owner.Palette;
        // Suppress renders while a stop is dragged; render once when the drag finishes.
        owner.Palette.Changed += (_, _) =>
        {
            if (!PaletteEditor.IsDragging) _owner.RequestRender(150);
        };
        PaletteEditor.DragCompleted += (_, _) => _owner.RequestRender(0);
    }

    /// <summary>Re-binds the editor to the owner's palette (the editor also follows palette changes itself).</summary>
    public void SyncFromState() => PaletteEditor.Palette = _owner.Palette;
}
