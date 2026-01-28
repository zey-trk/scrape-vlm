"""
Optimized Main Pipeline - Generates high-quality RAG chunks with content cleaning
"""
import asyncio
import os
import time
from datetime import datetime
from crawl_hadi import crawl
from utils.rag_chunker_optimized import OptimizedRAGChunker
from utils.export_to_excel import ExcelExporter


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    """Main execution pipeline with optimized chunking"""
    start_time = time.time()
    
    print_header("🚀 OPTIMIZED RAG CHUNKING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✨ Key improvements in this version:")
    print("   • URL and markdown artifact removal")
    print("   • Sentence-aware splitting (no mid-sentence cuts)")
    print("   • Boilerplate/navigation removal")
    print("   • Low-quality chunk filtering")
    
    # Check if we need to scrape or use existing data
    if os.path.exists("output.json"):
        print("\n📁 Using existing scraped data (output.json)")
        print("   Delete output.json and re-run to scrape fresh data")
    else:
        # Step 1: Web Scraping
        print_header("📥 STEP 1: WEB SCRAPING")
        try:
            asyncio.run(crawl())
            print("✅ Web scraping completed!")
        except Exception as e:
            print(f"❌ Error during web scraping: {e}")
            return
    
    # Step 2: Optimized RAG Chunking
    print_header("✂️  STEP 2: OPTIMIZED RAG CHUNKING")
    
    try:
        chunker = OptimizedRAGChunker("output.json")
        chunks = chunker.process_all_strategies()
        stats = chunker.get_strategy_stats()
        
        print("\n📊 CHUNKING STATISTICS:")
        for strategy, stat in stats.items():
            print(f"\n  {strategy}:")
            print(f"    • Total chunks: {stat['total_chunks']}")
            print(f"    • Avg chars: {stat['avg_chars']:.1f}")
            print(f"    • Avg words: {stat['avg_words']:.1f}")
        
        # Show sample chunks
        chunker.show_sample_chunks(2)
        
    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Excel Export
    print_header("📊 STEP 3: EXCEL EXPORT")
    
    try:
        output_file = "rag_chunks_optimized.xlsx"
        exporter = ExcelExporter(chunks, stats, output_file)
        exporter.export()
        
        print(f"\n✅ Export complete: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during Excel export: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final Summary
    elapsed_time = time.time() - start_time
    
    print_header("✨ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {elapsed_time:.1f} seconds")
    
    print("\n📁 Generated Files:")
    print("   • output.json - Raw scraped data")
    print("   • rag_chunks_optimized.xlsx - Optimized chunks for RAG")
    
    print("\n🎯 Quality Improvements Applied:")
    print("   ✓ URLs removed (won't corrupt embeddings)")
    print("   ✓ Markdown artifacts cleaned")
    print("   ✓ Navigation/boilerplate removed")
    print("   ✓ Sentence completeness ensured")
    print("   ✓ Low-quality chunks filtered out")
    
    print("\n💡 Next Steps:")
    print("   1. Open rag_chunks_optimized.xlsx")
    print("   2. Review the chunks - they should be cleaner!")
    print("   3. Use 'Optimized_RAG' strategy for your chatbot")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
