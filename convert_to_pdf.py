#!/usr/bin/env python3
"""
Convert markdown report to PDF using markdown-pdf library
"""

import sys
import os
from pathlib import Path

def convert_markdown_to_pdf(input_file, output_file):
    """Convert markdown file to PDF"""
    try:
        from markdown_pdf import MarkdownPdf

        # Initialize PDF converter
        pdf = MarkdownPdf()

        # Read markdown content
        with open(input_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Convert to PDF
        pdf.add_section(markdown_content, user_css="""
            body { font-family: 'Times New Roman', serif; font-size: 12pt; }
            h1 { font-size: 24pt; font-weight: bold; margin-top: 30pt; margin-bottom: 15pt; }
            h2 { font-size: 18pt; font-weight: bold; margin-top: 25pt; margin-bottom: 12pt; }
            h3 { font-size: 14pt; font-weight: bold; margin-top: 20pt; margin-bottom: 10pt; }
            table { border-collapse: collapse; width: 100%; margin: 15pt 0; }
            th, td { border: 1pt solid black; padding: 8pt; text-align: left; }
            th { background-color: #f5f5f5; font-weight: bold; }
            code { font-family: 'Courier New', monospace; background-color: #f5f5f5; padding: 2pt 4pt; }
            pre { background-color: #f5f5f5; padding: 10pt; border-left: 3pt solid #ccc; }
            .page-break { page-break-before: always; }
        """)

        # Save PDF
        pdf.save(output_file)
        print(f"✅ PDF report created successfully: {output_file}")
        return True

    except ImportError as e:
        print(f"❌ Error: markdown-pdf library not found. {e}")
        print("Please install it with: pip install markdown-pdf")
        return False
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)

    success = convert_markdown_to_pdf(input_file, output_file)
    if success:
        # Get file size
        file_size = os.path.getsize(output_file)
        print(".1f"
        print("\n📄 PDF Generation Summary:")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        print(".1f"        print("   Ready for submission! 📋"
    else:
        sys.exit(1)
