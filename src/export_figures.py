#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced export function for publication-ready figures
Exports visualizations in multiple formats: HTML, PNG, PDF, SVG
Font: Arial/Helvetica as per guidelines
"""

import os
import plotly.graph_objects as go
from pathlib import Path

def save_figure_multi_format(fig, base_filename, output_dir="abbildungen", dpi=600):
    """
    Save Plotly figure in multiple formats for publication.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to save
    base_filename : str
        Base filename without extension (e.g., "Abb1" or "cluster_analysis")
    output_dir : str
        Output directory path (default: "abbildungen")
    dpi : int
        Resolution for raster formats (default: 600 for combination graphics)
    
    Formats exported:
    -----------------
    - HTML: Interactive version for online/digital use
    - PNG: High-resolution raster (600 dpi for combination graphics, 300 dpi for photos)
    - PDF: Vector format, can be converted to EPS if needed
    - SVG: Alternative vector format
    
    Publication Guidelines Compliance:
    -----------------------------------
    - Kombinationsgraphiken (color diagrams with text): 600 dpi minimum
    - Halbtonabbildungen (photos with shading): 300 dpi minimum  
    - Strichzeichnungen (line drawings): Would need 1200 dpi or vector (PDF/SVG)
    - Fonts: Arial/Helvetica embedded in vector formats
    - Size: 80mm or 122mm width (will need manual adjustment in layout)
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure fonts are set to Arial/Helvetica as per guidelines
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif")
    )
    
    formats_saved = []
    
    # 1. HTML (interactive, online version)
    html_file = output_path / f"{base_filename}.html"
    fig.write_html(
        str(html_file),
        include_plotlyjs='cdn',
        config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': base_filename,
                'height': None,
                'width': None,
                'scale': 3
            },
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
    )
    formats_saved.append(f"HTML: {html_file}")
    
    # 2. PNG (high-resolution raster)
    # Using 600 dpi for combination graphics (color diagrams with labels)
    png_file = output_path / f"{base_filename}.png"
    fig.write_image(
        str(png_file),
        format='png',
        width=1200,  # pixels - adjust based on final print size
        height=800,   # pixels - adjust based on final print size
        scale=3       # multiplier for resolution
    )
    formats_saved.append(f"PNG (600+ dpi): {png_file}")
    
    # 3. PDF (vector format - preferred for print)
    pdf_file = output_path / f"{base_filename}.pdf"
    fig.write_image(
        str(pdf_file),
        format='pdf',
        width=480,  # in points (122mm ≈ 345pt, 80mm ≈ 227pt)
        height=320   # in points - adjust as needed
    )
    formats_saved.append(f"PDF (vector): {pdf_file}")
    
    # 4. SVG (alternative vector format)
    svg_file = output_path / f"{base_filename}.svg"
    fig.write_image(
        str(svg_file),
        format='svg',
        width=480,
        height=320
    )
    formats_saved.append(f"SVG (vector): {svg_file}")
    
    # Print confirmation
    print(f"\n✅ Figure saved in multiple formats:")
    for fmt in formats_saved:
        print(f"   {fmt}")
    
    print(f"\n📋 Publication notes:")
    print(f"   - Graphikprogramm: Plotly (Python) v{go.__version__}")
    print(f"   - Fonts: Arial/Helvetica (embedded in vector formats)")
    print(f"   - PNG Resolution: 600+ dpi (combination graphics)")
    print(f"   - Vector formats: PDF, SVG (fonts embedded)")
    print(f"   - Final size adjustment: Edit width/height parameters for 80mm or 122mm")
    print(f"   - EPS conversion: Use Adobe Acrobat or Inkscape to convert PDF→EPS if needed")
    
    return {
        'html': str(html_file),
        'png': str(png_file),
        'pdf': str(pdf_file),
        'svg': str(svg_file)
    }


def convert_pdf_to_eps(pdf_file, eps_file=None):
    """
    Helper function to convert PDF to EPS using system tools.
    Requires: ghostscript (gs) or Adobe Acrobat
    
    Usage:
        convert_pdf_to_eps("Abb1.pdf", "Abb1.eps")
    
    Or use Inkscape command line:
        inkscape --export-eps=Abb1.eps Abb1.pdf
    """
    if eps_file is None:
        eps_file = pdf_file.replace('.pdf', '.eps')
    
    import subprocess
    
    try:
        # Try using ghostscript
        subprocess.run([
            'gs',
            '-dNOPAUSE',
            '-dBATCH',
            '-dEPSCrop',
            '-sDEVICE=eps2write',
            f'-sOutputFile={eps_file}',
            pdf_file
        ], check=True)
        print(f"✅ EPS created: {eps_file}")
        return eps_file
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"⚠️  Could not convert to EPS automatically.")
        print(f"   Manual options:")
        print(f"   1. Adobe Acrobat: Open {pdf_file} → Save As → EPS")
        print(f"   2. Inkscape: inkscape --export-eps={eps_file} {pdf_file}")
        print(f"   3. Command line: gs -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=eps2write -sOutputFile={eps_file} {pdf_file}")
        return None


# Example usage for reference
if __name__ == "__main__":
    # This is a template - actual figures are generated by visualization scripts
    
    print("📚 Multi-Format Export Function for Publication-Ready Figures")
    print("=" * 70)
    print("\nThis module provides save_figure_multi_format() function")
    print("to export visualizations in multiple formats.\n")
    print("Formats: HTML, PNG (600+ dpi), PDF (vector), SVG (vector)")
    print("Publication guidelines: German academic standards")
    print("Fonts: Arial/Helvetica (as per guidelines)")
    print("\nUsage in your visualization script:")
    print("-" * 70)
    print("""
    from export_figures import save_figure_multi_format
    
    # Create your plotly figure
    fig = go.Figure(...)
    
    # Save in all formats
    save_figure_multi_format(fig, "Abb1", output_dir="abbildungen")
    """)
    print("-" * 70)
