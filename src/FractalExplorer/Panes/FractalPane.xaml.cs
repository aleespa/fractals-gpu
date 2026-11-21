using System.Globalization;
using FractalExplorer.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;

namespace FractalExplorer.Panes;

/// <summary>Side panel for the "Fractal" rail item: type, iterations, Julia constant, reset.</summary>
public sealed partial class FractalPane : UserControl
{
    private readonly MainWindow _owner;
    private bool _syncing;

    public FractalPane(MainWindow owner)
    {
        _owner = owner;
        _syncing = true;   // XAML initialisation raises ValueChanged/SelectionChanged
        InitializeComponent();
        _syncing = false;
        SyncFromState();
    }

    private ViewState View => _owner.View;

    /// <summary>Pushes the owner's view state into every control without triggering change handlers.</summary>
    public void SyncFromState()
    {
        bool was = _syncing;
        _syncing = true;
        try
        {
            TypeCombo.SelectedIndex = (int)View.Type;
            IterSlider.Value = View.MaxIterations;
            CxSlider.Value = View.Cx;
            CySlider.Value = View.Cy;
            UpdateJuliaVisibility();
            UpdateLabels();
        }
        finally
        {
            _syncing = was;
        }
    }

    private void TypeCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_syncing) return;
        View.Type = TypeCombo.SelectedIndex == 1 ? FractalType.Julia : FractalType.Mandelbrot;
        UpdateJuliaVisibility();
        _owner.RequestRender(150);
    }

    private void IterSlider_ValueChanged(object sender, RangeBaseValueChangedEventArgs e)
    {
        if (_syncing) return;
        View.MaxIterations = (int)Math.Round(IterSlider.Value);
        UpdateLabels();
        _owner.RequestRender(150);
    }

    private void JuliaSlider_ValueChanged(object sender, RangeBaseValueChangedEventArgs e)
    {
        if (_syncing) return;
        View.Cx = CxSlider.Value;
        View.Cy = CySlider.Value;
        UpdateLabels();
        _owner.RequestRender(150);
    }

    private void ResetButton_Click(object sender, RoutedEventArgs e) => _owner.ResetView();

    private void UpdateJuliaVisibility() =>
        JuliaControls.Visibility = View.Type == FractalType.Julia ? Visibility.Visible : Visibility.Collapsed;

    private void UpdateLabels()
    {
        var inv = CultureInfo.InvariantCulture;
        IterLabel.Text = "Max iterations · " + View.MaxIterations.ToString(inv);
        CxLabel.Text = "Cx · " + View.Cx.ToString("F3", inv);
        CyLabel.Text = "Cy · " + View.Cy.ToString("F3", inv);
    }
}
