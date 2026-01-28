"""
Strict Verification Script for Ubicloud VLM Integration
TARGET: urun_ve_hizmet_ucretleri.pdf
GOAL: Verify content extraction and measure EXACT token consumption.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import aiohttp
import os
from utils.pdf_vlm_processor import VLMPDFProcessor, load_dotenv

async def verify_ubicloud():
    # Force reload env
    load_dotenv(override=True)
    
    print("\n" + "="*60)
    print("🧪 UBICLOUD VLM VERIFICATION TEST")
    print("="*60)
    
    # Check Key
    key = os.getenv("ubicloud_api_key")
    if not key:
        print("❌ CRITICAL: 'ubicloud_api_key' not found in .env")
        return
    print(f"✅ Key found: {key[:5]}...{key[-3:]}")
    
    # Initialize Processor
    processor = VLMPDFProcessor(provider="ubicloud")
    if not processor.is_available():
        print("❌ Processor initialization failed (Check dependencies)")
        return
    print(f"✅ Processor initialized for model: {processor.model}")
    print(f"   Endpoint: {processor.client.base_url}")

    # Download PDF
    url = "https://www.tombank.com.tr/assets/images/doc/urun_ve_hizmet_ucretleri.pdf"
    print(f"\n📥 Downloading target PDF: {url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"❌ Download failed: {response.status}")
                return
            pdf_bytes = await response.read()
            print(f"✅ Downloaded {len(pdf_bytes)} bytes")

    # Run Analysis
    print("\n🧠 Sending to VLM (Qwen/Qwen3-VL-235B)...")
    print("   WARNING: This consumes real tokens via Ubicloud API.")
    
    result = await processor.process_pdf(pdf_bytes, "urun_ve_hizmet_ucretleri.pdf")
    
    content = result.get("content", "")
    usage = result.get("total_usage", {})
    
    print("\n" + "="*60)
    print("📊 TOKEN CONSUMPTION REPORT")
    print("="*60)
    print(f"Prompt Tokens:      {usage.get('prompt_tokens', 0):,}")
    print(f"Completion Tokens:  {usage.get('completion_tokens', 0):,}")
    print(f"TOTAL TOKENS:       {usage.get('total_tokens', 0):,}")
    print("="*60)
    
    if content:
        print("\n📄 EXTRACTION PREVIEW (First 500 chars):")
        print("-" * 40)
        print(content[:500])
        print("..." + "-" * 40)
        
        # Check for key terms
        keywords = ["Havale", "EFT", "FAST", "Ücretsiz"]
        found = [k for k in keywords if k in content]
        print(f"\n🔍 Keyword Check: Found {found}/{len(keywords)}")
    else:
        print("\n❌ Extraction failed: No content returned")

if __name__ == "__main__":
    asyncio.run(verify_ubicloud())
