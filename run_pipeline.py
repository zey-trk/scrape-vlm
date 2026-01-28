"""
Unified RAG Pipeline Runner
Executes the complete workflow:
1. Web Scraping (comprehensive_scraper.py)
2. PDF Processing (process_pdfs.py)
3. RAG Chunking & Export (rag_chunker_optimized.py + export_to_excel.py)
"""
import asyncio
import time
import os
from datetime import datetime

# Import workflow components
from utils.web_scraper import comprehensive_crawl
from process_pdfs import integrate_pdfs
from utils.rag_chunker_optimized import OptimizedRAGChunker
from utils.export_to_excel import ExcelExporter
from utils.config import OUTPUT_FILE, EXCEL_FILE

def print_header(text, char="="):
    """Print a formatted header"""
    print("\n" + char * 70)
    print(f"  {text}")
    print(char * 70)

def run_chunking_step():
    """Run the chunking and export step"""
    print_header("✂️  STEP 3: OPTIMIZED RAG CHUNKING", "-")
    
    output_file = OUTPUT_FILE
    if not os.path.exists(output_file):
        print(f"❌ Error: {output_file} not found. Scraping must have failed.")
        return

    try:
        # Initialize chunker
        chunker = OptimizedRAGChunker(output_file)
        
        # Process chunks
        chunks = chunker.process_all_strategies()
        stats = chunker.get_strategy_stats()
        
        print("\n📊 CHUNKING STATISTICS:")
        for strategy, stat in stats.items():
            print(f"\n  {strategy}:")
            print(f"    • Total chunks: {stat['total_chunks']}")
            print(f"    • Avg chars: {stat['avg_chars']:.1f}")
        
        # Export to Excel
        print_header("📊 STEP 4: EXCEL EXPORT", "-")
        excel_file = EXCEL_FILE
        exporter = ExcelExporter(chunks, stats, excel_file)
        exporter.export()
        
        print(f"\n✅ Export complete: {excel_file}")
        
    except Exception as e:
        print(f"❌ Error during chunking/export: {e}")
        import traceback
        traceback.print_exc()

async def run_pipeline():
    """Execute the full pipeline sequentially"""
    start_time = time.time()
    
    print_header("🚀 STARTING FULL RAG PIPELINE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # STEP 1: Comprehensive Scraping
    print_header("📥 STEP 1: WEB SCRAPING (comprehensive_scraper)", "-")
    # Remove existing output to ensure fresh scrape
    if os.path.exists("rag_results/output.json"):
        os.remove("rag_results/output.json")
        print("🗑️  Removed old rag_results/output.json")
        
    await comprehensive_crawl()
    
    # STEP 2: PDF Processing
    print_header("📄 STEP 2: PDF INTEGRATION (process_pdfs)", "-")
    await integrate_pdfs()
    
    # STEP 3 & 4: Chunking and Export
    # (Synchronous wrapper around chunking logic)
    run_chunking_step()
    
    # Final Summary
    elapsed = time.time() - start_time
    print_header("✨ PIPELINE COMPLETE")
    print(f"Total Workflow Time: {elapsed:.1f} seconds")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
