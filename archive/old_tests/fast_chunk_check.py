"""
Fast Path: Process Single PDF and Show Chunks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import os
import aiohttp
from utils.pdf_vlm_processor import VLMPDFProcessor, load_dotenv
from utils.rag_chunker_optimized import OptimizedRAGChunker

async def run_fast_check():
    load_dotenv()
    url = "https://www.tombank.com.tr/assets/images/doc/urun_ve_hizmet_ucretleri.pdf"
    name = "urun_ve_hizmet_ucretleri.pdf"
    
    print(f"🚀 Processing fast path for: {name}")
    
    # 1. Download & VLM Extract
    processor = VLMPDFProcessor(provider="ubicloud")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            content_bytes = await resp.read()
            
    print("🧠 Extracting with VLM (Ubicloud)...")
    result = await processor.process_pdf(content_bytes, name)
    text = result['content']
    
    # 2. Save to temp JSON
    mini_data = [{
        "url": url,
        "title": "Fast Path VLM extract",
        "content": text,
        "media": {},
        "metadata": {"type": "pdf", "source": "fast_path"}
    }]
    
    with open("rag_results/temp_fast_output.json", "w") as f:
        json.dump(mini_data, f)
        
    # 3. Chunk
    print("\n✂️  Chunking...")
    chunker = OptimizedRAGChunker("rag_results/temp_fast_output.json")
    chunks = chunker.process_all_strategies()
    
    # 4. Save Chunks to JSON (User Request)
    output_json = "rag_results/urun_ve_hizmet_ucretleri_chunks.json"
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*60)
    print(f"✅ Saved {len(chunks)} chunks to {output_json}")
    print("="*60)
    
    # 5. Show Preview
    for i, chunk in enumerate(chunks):
        if i >= 3: break # Show first 3
        print(f"\n--- Chunk {i+1} ({chunk['strategy']}) ---")
        print(f"Size: {len(chunk['chunk_text'])} chars")
        print("-" * 30)
        print(chunk['chunk_text'][:200] + "...")

if __name__ == "__main__":
    asyncio.run(run_fast_check())
