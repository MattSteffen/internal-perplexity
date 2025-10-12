#!/usr/bin/env python3
"""
Simple test script for document conversion.

This script tests the converter package by loading test PDFs, converting them using
different converter backends, saving outputs to files, and reporting on the success of each conversion.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path


from crawler.converter import *

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_converter(converter_name: str, config, test_pdf_path: Path, output_dir: Path):
    """Test a specific converter with the given configuration and save outputs."""
    print(f"\n{'='*60}")
    print(f"Testing {converter_name} on {test_pdf_path.name}")
    print(f"{'='*60}")
    
    # Create converter-specific output directory
    converter_output_dir = output_dir / converter_name.replace(" ", "_").replace(":", "_")
    converter_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create converter
        converter = create_converter(config)
        print(f"✅ {converter_name} converter created successfully")
        
        # Load test PDF
        doc = DocumentInput.from_path(test_pdf_path)
        print(f"✅ Test PDF loaded: {doc.filename}")
        
        # Set up conversion options
        options = ConvertOptions(
            include_metadata=True,
            include_page_numbers=True,
            include_images=True,
            describe_images=True,
            extract_tables=True,
        )
        
        # Convert document
        print(f"🔄 Starting conversion with {converter_name}...")
        result = converter.convert(doc, options=options)
        
        # Report results
        print(f"✅ Conversion completed successfully!")
        print(f"📊 Results:")
        print(f"   • Output length: {len(result.markdown)} characters")
        print(f"   • Images found: {len(result.images)}")
        print(f"   • Tables found: {len(result.tables)}")
        print(f"   • Processing time: {result.stats.total_time_sec:.2f}s")
        
        if result.images:
            print(f"   • Images described: {result.stats.images_described}")
        
        # Save outputs to files
        pdf_name = test_pdf_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save markdown output
        markdown_file = converter_output_dir / f"{pdf_name}_{timestamp}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(result.markdown)
        print(f"💾 Markdown saved to: {markdown_file}")
        
        # Save metadata
        metadata_file = converter_output_dir / f"{pdf_name}_{timestamp}_metadata.json"
        metadata = {
            "converter": converter_name,
            "source_file": str(test_pdf_path),
            "conversion_time": timestamp,
            "output_length": len(result.markdown),
            "images_count": len(result.images),
            "tables_count": len(result.tables),
            "processing_time_sec": result.stats.total_time_sec,
            "images_described": result.stats.images_described if hasattr(result.stats, 'images_described') else 0,
            "markdown_file": str(markdown_file),
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Metadata saved to: {metadata_file}")
        
        # Save images if any
        if result.images:
            images_dir = converter_output_dir / f"{pdf_name}_{timestamp}_images"
            images_dir.mkdir(exist_ok=True)
            for i, image in enumerate(result.images):
                image_file = images_dir / f"image_{i:03d}.png"
                with open(image_file, 'wb') as f:
                    f.write(image.data)
            print(f"💾 {len(result.images)} images saved to: {images_dir}")
        
        # Save tables if any
        if result.tables:
            tables_file = converter_output_dir / f"{pdf_name}_{timestamp}_tables.json"
            tables_data = []
            for i, table in enumerate(result.tables):
                tables_data.append({
                    "index": i,
                    "data": table.data,
                    "metadata": table.metadata if hasattr(table, 'metadata') else {}
                })
            with open(tables_file, 'w', encoding='utf-8') as f:
                json.dump(tables_data, f, indent=2)
            print(f"💾 {len(result.tables)} tables saved to: {tables_file}")
        
        # Show a preview of the markdown output
        preview = result.markdown[:500] + "..." if len(result.markdown) > 500 else result.markdown
        print(f"\n📝 Markdown preview:")
        print("-" * 40)
        print(preview)
        print("-" * 40)
        
        return True, converter_output_dir
        
    except Exception as e:
        print(f"❌ {converter_name} conversion failed: {e}")
        return False, None
    
    finally:
        # Clean up converter resources
        if 'converter' in locals():
            converter.close()


def main():
    """Main test function."""
    setup_logging()
    
    # Recreate output directory, wiping it if it exists
    output_dir = Path(__file__).parent / "test_outputs"
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
            else:
                item.unlink()
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory (wiped): {output_dir}")
    
    # Test PDF files
    test_pdfs = [
        Path(__file__).parent / "image.pdf",
        Path(__file__).parent / "multicolumn.pdf"
    ]
    
    # Verify test PDFs exist
    missing_pdfs = [pdf for pdf in test_pdfs if not pdf.exists()]
    if missing_pdfs:
        print(f"❌ Missing test PDFs: {[str(p) for p in missing_pdfs]}")
        return 1
    
    print(f"📄 Found {len(test_pdfs)} test PDFs:")
    for pdf in test_pdfs:
        print(f"   • {pdf.name} ({pdf.stat().st_size} bytes)")
    
    # Test configurations
    test_configs = [
        ("PyMuPDF with Ollama VLM", PyMuPDFConfig(
            type="pymupdf",
            image_describer={
                "type": "ollama",
                "model": "granite3.2-vision:latest",
                "base_url": "http://localhost:11434"
            }
        )),
        # ("Docling with VLM", DoclingConfig(
        #     type="docling",
        #     use_vlm=True,
        #     vlm_base_url="http://localhost:11434",
        #     vlm_model="granite3.2-vision:latest"
        # )),
        # ("Docling without VLM", DoclingConfig(
        #     type="docling",
        #     use_vlm=False
        # )),
        ("Docling API with VLM", DoclingAPIConfig(
            type="docling_api",
            base_url="http://localhost:5001",
            vlm_url="http://localhost:11434/v1/chat/completions",
            vlm_model="granite3.2-vision:latest",
            timeout=600,
            do_picture_description=True,
            include_images=True
        )),
        ("MarkItDown", MarkItDownConfig(
            type="markitdown",
            llm_base_url="http://localhost:11434",
            llm_model="granite3.2-vision:latest"
        )),
    ]
    
    # Run tests for each PDF and converter combination
    all_results = {}
    output_dirs = set()
    
    for pdf_path in test_pdfs:
        print(f"\n{'='*80}")
        print(f"Testing PDF: {pdf_path.name}")
        print(f"{'='*80}")
        
        pdf_results = {}
        for name, config in test_configs:
            success, converter_output_dir = test_converter(name, config, pdf_path, output_dir)
            pdf_results[name] = success
            if converter_output_dir:
                output_dirs.add(converter_output_dir)
        
        all_results[pdf_path.name] = pdf_results
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = 0
    successful_tests = 0
    
    for pdf_name, pdf_results in all_results.items():
        print(f"\n📄 {pdf_name}:")
        for converter_name, success in pdf_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {converter_name}")
            total_tests += 1
            if success:
                successful_tests += 1
    
    print(f"\n📊 Overall Results: {successful_tests}/{total_tests} tests passed")
    print(f"📁 Output directories created:")
    for output_dir_path in sorted(output_dirs):
        print(f"   • {output_dir_path}")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
