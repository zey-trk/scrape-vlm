"""
Main Pipeline - Orchestrates web scraping, RAG chunking, and Excel export
"""
import asyncio
import os
import time
from datetime import datetime
from crawl_hadi import crawl
from rag_chunker import RAGChunker
from export_to_excel import ExcelExporter


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    """Main execution pipeline"""
    start_time = time.time()
    
    print_header("🚀 RAG CHUNKING PIPELINE FOR TOMBANKHADI.COM")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Web Scraping
    print_header("📥 STEP 1: WEB SCRAPING")
    print("Crawling tombankhadi.com and extracting content...")
    
    try:
        asyncio.run(crawl())
        print("✅ Web scraping completed successfully!")
    except Exception as e:
        print(f"❌ Error during web scraping: {e}")
        return
    
    # Verify output file exists
    if not os.path.exists("output.json"):
        print("❌ Error: output.json not found after scraping")
        return
    
    # Step 2: RAG Chunking
    print_header("✂️  STEP 2: RAG CHUNKING")
    print("Applying multiple LangChain text splitting strategies...")
    
    try:
        chunker = RAGChunker("output.json")
        chunks = chunker.process_all_strategies()
        stats = chunker.get_strategy_stats()
        
        print("\n📊 CHUNKING STATISTICS:")
        for strategy, stat in stats.items():
            print(f"\n  {strategy}:")
            print(f"    • Total chunks: {stat['total_chunks']}")
            print(f"    • Avg chars: {stat['avg_chars']:.1f}")
            print(f"    • Avg words: {stat['avg_words']:.1f}")
        
        print(f"\n✅ Created {len(chunks)} total chunks across all strategies!")
        
    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Excel Export
    print_header("📊 STEP 3: EXCEL EXPORT")
    print("Exporting chunks to Excel with analysis sheets...")
    
    try:
        exporter = ExcelExporter(chunks, stats, "rag_chunks_analysis.xlsx")
        exporter.export()
        
        print("\n✅ Excel export completed successfully!")
        print(f"   Output file: rag_chunks_analysis.xlsx")
        
    except Exception as e:
        print(f"❌ Error during Excel export: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final Summary
    elapsed_time = time.time() - start_time
    
    print_header("✨ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {elapsed_time:.1f} seconds")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Generated Files:")
    print("   • output.json - Raw scraped data")
    print("   • rag_chunks_analysis.xlsx - Complete chunk analysis")
    print("\n💡 Next Steps:")
    print("   1. Open rag_chunks_analysis.xlsx")
    print("   2. Review the 'Strategy Comparison' sheet")
    print("   3. Analyze chunks from different strategies")
    print("   4. Choose the best strategy for your RAG chatbot")
    print("\n🎯 RECOMMENDATION: Start with 'RecursiveCharacterTextSplitter (1000/200)'")
    print("   This is the most commonly used strategy for RAG applications.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
