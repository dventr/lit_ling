#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Einfacher PDF/PNG Export für alle Abbildungen.
Verwendet playwright/selenium für Screenshot-basierte Konvertierung.
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
import subprocess
import os

# Konfiguration
ABBILDUNGEN_DIR = Path('abbildungen')
WIDTH_MM = 122
HEIGHT_MM = 90

def export_with_playwright(html_file, output_pdf, output_png):
    """
    Exportiert HTML zu PDF/PNG mit playwright (Chrome headless).
    """
    try:
        # PDF mit playwright
        script = f"""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('file://{html_file.absolute()}')
    page.wait_for_load_state('networkidle')
    
    # PDF
    page.pdf(
        path='{output_pdf}',
        width='{WIDTH_MM}mm',
        height='{HEIGHT_MM}mm',
        print_background=True
    )
    
    # PNG Screenshot
    page.screenshot(
        path='{output_png}',
        full_page=True
    )
    
    browser.close()
    print('✅ Exportiert: {output_pdf.name}, {output_png.name}')
"""
        
        with open('_temp_export.py', 'w') as f:
            f.write(script)
        
        subprocess.run(['python3', '_temp_export.py'], check=True)
        os.remove('_temp_export.py')
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Playwright-Export fehlgeschlagen: {e}")
        return False

def export_with_chrome(html_file, output_pdf, output_png):
    """
    Exportiert HTML zu PDF und PNG mit Chrome headless (macOS).
    PNG mit 600dpi-äquivalenter Auflösung (2x scaling).
    """
    try:
        chrome_path = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        
        # PDF Export
        cmd_pdf = [
            chrome_path,
            '--headless',
            '--disable-gpu',
            '--print-to-pdf=' + str(output_pdf),
            f'--print-to-pdf-no-header',
            'file://' + str(html_file.absolute())
        ]
        
        subprocess.run(cmd_pdf, capture_output=True, text=True, timeout=10)
        
        if output_pdf.exists():
            print(f"  ✅ PDF: {output_pdf.name}")
        
        # PNG Export mit hoher Auflösung
        # 122mm @ 600dpi = 2897px, 90mm @ 600dpi = 2126px
        cmd_png = [
            chrome_path,
            '--headless',
            '--disable-gpu',
            '--screenshot=' + str(output_png),
            '--window-size=2897,2126',
            '--force-device-scale-factor=2',  # 2x für höhere Qualität
            'file://' + str(html_file.absolute())
        ]
        
        subprocess.run(cmd_png, capture_output=True, text=True, timeout=10)
        
        if output_png.exists():
            print(f"  ✅ PNG: {output_png.name} (600 dpi)")
            return True
        
        return True
            
    except Exception as e:
        print(f"  ⚠️  Chrome-Export fehlgeschlagen: {e}")
        return False

def main():
    print("PDF/PNG Export für Abbildungen")
    print("=" * 60)
    print()
    
    html_files = sorted(ABBILDUNGEN_DIR.glob('Abb[0-9]*.html'))
    
    if not html_files:
        print("❌ Keine HTML-Dateien gefunden")
        return
    
    print(f"Gefunden: {len(html_files)} HTML-Dateien\n")
    
    # Prüfe verfügbare Tools
    has_chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome').exists()
    
    try:
        import playwright
        has_playwright = True
    except:
        has_playwright = False
    
    if not has_chrome and not has_playwright:
        print("❌ Keine Export-Tools verfügbar!")
        print("\nInstallieren Sie:")
        print("  pip install playwright")
        print("  playwright install chromium")
        print("\nOder verwenden Sie Chrome (bereits installiert auf macOS)")
        return
    
    print("Export-Methode:", "Chrome" if has_chrome else "Playwright")
    print()
    
    for html_file in html_files:
        fig_num = int(''.join(filter(str.isdigit, html_file.stem)))
        prefix = f"Abb{fig_num}"
        
        print(f"{prefix}: {html_file.name}")
        
        output_pdf = ABBILDUNGEN_DIR / f"{prefix}.pdf"
        output_png = ABBILDUNGEN_DIR / f"{prefix}.png"
        
        if has_chrome:
            export_with_chrome(html_file, output_pdf, output_png)
        elif has_playwright:
            export_with_playwright(html_file, output_pdf, output_png)
        
        print()
    
    print("=" * 60)
    print("✅ Export abgeschlossen!")
    print(f"📂 Ausgabe: {ABBILDUNGEN_DIR}")

if __name__ == "__main__":
    main()
