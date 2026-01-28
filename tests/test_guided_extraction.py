"""
Test script for VLM-Guided Extraction
Demonstrates extraction with custom Turkish instruction for campaign data.
"""
import asyncio
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vlm_extractor import VLMExtractor


async def test_guided_extraction():
    """Test guided extraction with user's example instruction."""
    print("\n" + "=" * 70)
    print("🎯 VLM-GUIDED EXTRACTION TEST")
    print("=" * 70)
    
    # Initialize extractor
    extractor = VLMExtractor()
    print(f"✅ Model: {extractor.model}")
    print(f"📁 Output dir: {extractor.output_dir}")
    
    # Target URL
    url = "https://tombankhadi.com/hadi-ozel-bankacilik/ozel-bankacilik-segmentlerimiz"
    
    # User's instruction (in Turkish)
    instruction = """
    Sayfadaki 'Segmentler Bazında Ayrıcalıklar ve Kazanımlar' tablosundan aşağıdaki bilgileri çıkar:
    
    1. Tüm kampanya isimlerini (örn: Restoran Kampanyası, Uçak Bileti Kampanyası, vb.)
    2. Her kampanya için iade/indirim oranlarını
    3. Elite, Elite Plus ve Prestige segmentleri için:
       - İndirim/iade oranları
       - Günlük limit
       - Aylık limit (varsa)
    
    Sonucu aşağıdaki JSON yapısında döndür:
    {
        "kampanyalar": [
            {
                "kampanya_adi": "...",
                "elite": {"iade_orani": "...", "gunluk_limit": "...", "aylik_limit": "..."},
                "elite_plus": {"iade_orani": "...", "gunluk_limit": "...", "aylik_limit": "..."},
                "prestige": {"iade_orani": "...", "gunluk_limit": "...", "aylik_limit": "..."}
            }
        ],
        "segment_kriterleri": {
            "elite": "...",
            "elite_plus": "...",
            "prestige": "..."
        }
    }
    """
    
    print(f"\n🔗 URL: {url}")
    print(f"\n📝 Instruction:\n{instruction[:200]}...")
    
    print("\n⏳ Extracting data (this may take a minute)...")
    
    # Perform extraction
    result = await extractor.extract(
        url=url,
        instruction=instruction,
        output_format="json",
        screenshot_name="segment_campaigns"
    )
    
    if result.error:
        print(f"\n❌ Error: {result.error}")
        return
    
    print("\n" + "=" * 70)
    print("📊 EXTRACTION RESULT")
    print("=" * 70)
    
    # Pretty print the structured data
    if result.structured_data:
        print(json.dumps(result.structured_data, ensure_ascii=False, indent=2))
    else:
        print("Raw response:")
        print(result.raw_response)
    
    # Token usage
    print("\n" + "-" * 40)
    print("📈 TOKEN USAGE:")
    print(f"   Prompt tokens: {result.usage.get('prompt_tokens', 0):,}")
    print(f"   Completion tokens: {result.usage.get('completion_tokens', 0):,}")
    print(f"   Total tokens: {result.usage.get('total_tokens', 0):,}")
    
    # Save results
    output_path = extractor.save_results(result, "campaign_extraction.json")
    print(f"\n💾 Results saved to: {output_path}")
    print(f"📸 Screenshot: {result.screenshot_path}")
    
    print("\n✅ Extraction complete!")


if __name__ == "__main__":
    asyncio.run(test_guided_extraction())
