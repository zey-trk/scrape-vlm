"""
Web Scraper with Playwright + Qwen3 VL Integration
Extracts page content including images and uses VLM for understanding.
"""
import asyncio
import base64
import os
import sys
import json
from pathlib import Path
from playwright.async_api import async_playwright
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# VLM Configuration
VLM_ENDPOINT = "https://us-east-a2.ai.ubicloud.com/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
API_KEY = os.getenv("ubicloud_api_key")

# Target URL
TARGET_URL = "https://tombankhadi.com/hadi-ozel-bankacilik/ozel-bankacilik-segmentlerimiz"


async def capture_screenshot(page, filepath: str) -> bytes:
    """Capture full page screenshot and return bytes."""
    await page.screenshot(path=filepath, full_page=True)
    with open(filepath, "rb") as f:
        return f.read()


async def extract_images(page) -> list[dict]:
    """Extract all images from the page with their URLs and alt text."""
    images = await page.evaluate("""
        () => {
            const imgs = Array.from(document.querySelectorAll('img'));
            return imgs.map(img => ({
                src: img.src,
                alt: img.alt || '',
                title: img.title || '',
                width: img.naturalWidth,
                height: img.naturalHeight
            }));
        }
    """)
    return images


async def extract_text_content(page) -> dict:
    """Extract structured text content from the page."""
    content = await page.evaluate("""
        () => {
            // Get page title
            const title = document.title;
            
            // Get main headings
            const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4')).map(h => ({
                level: h.tagName,
                text: h.innerText.trim()
            }));
            
            // Get paragraphs
            const paragraphs = Array.from(document.querySelectorAll('p')).map(p => p.innerText.trim()).filter(t => t.length > 0);
            
            // Get list items
            const listItems = Array.from(document.querySelectorAll('li')).map(li => li.innerText.trim()).filter(t => t.length > 0);
            
            // Get all visible text
            const bodyText = document.body.innerText;
            
            return {
                title,
                headings,
                paragraphs,
                listItems,
                bodyText
            };
        }
    """)
    return content


async def analyze_with_vlm(image_base64: str, context: str) -> dict:
    """Send image to Qwen3 VL for analysis."""
    if not API_KEY:
        return {"error": "API key not found in environment"}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Create message with image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"""Analyze this webpage screenshot from a Turkish bank's private banking segment page.

Context from the page:
{context[:1000]}

Please provide:
1. A summary of what you see in the image (in Turkish if the content is Turkish)
2. Key visual elements and their purpose
3. Any important information visible in the image
4. The overall design and user experience observations

Respond in a structured format."""
                }
            ]
        }
    ]
    
    payload = {
        "model": VLM_MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(VLM_ENDPOINT, headers=headers, json=payload, timeout=120) as response:
            if response.status != 200:
                error_text = await response.text()
                return {"error": f"VLM request failed: {response.status} - {error_text}"}
            
            result = await response.json()
            return {
                "analysis": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": result.get("usage", {})
            }


async def scrape_with_vlm():
    """Main scraping function."""
    print("\n" + "=" * 60)
    print("🌐 PLAYWRIGHT + QWEN3 VL WEB SCRAPER")
    print("=" * 60)
    print(f"Target URL: {TARGET_URL}")
    print(f"VLM Model: {VLM_MODEL}")
    
    # Check API key
    if not API_KEY:
        print("❌ CRITICAL: 'ubicloud_api_key' not found in .env")
        return
    print(f"✅ API Key found: {API_KEY[:5]}...{API_KEY[-3:]}")
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "scrape_output"
    output_dir.mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        print("\n📱 Launching browser...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = await context.new_page()
        
        # Navigate to the page
        print(f"🔗 Navigating to {TARGET_URL}...")
        try:
            await page.goto(TARGET_URL, wait_until="networkidle", timeout=60000)
            print("✅ Page loaded successfully")
        except Exception as e:
            print(f"❌ Navigation failed: {e}")
            await browser.close()
            return
        
        # Wait for content to render
        await asyncio.sleep(2)
        
        # Extract text content
        print("\n📄 Extracting text content...")
        text_content = await extract_text_content(page)
        print(f"   Title: {text_content['title']}")
        print(f"   Headings found: {len(text_content['headings'])}")
        print(f"   Paragraphs found: {len(text_content['paragraphs'])}")
        
        # Extract images
        print("\n🖼️ Extracting images...")
        images = await extract_images(page)
        print(f"   Images found: {len(images)}")
        for i, img in enumerate(images[:5]):  # Show first 5
            print(f"   [{i+1}] {img['alt'] or 'No alt text'} - {img['src'][:50]}...")
        
        # Capture screenshot
        print("\n📸 Capturing full page screenshot...")
        screenshot_path = str(output_dir / "page_screenshot.png")
        screenshot_bytes = await capture_screenshot(page, screenshot_path)
        print(f"   Saved to: {screenshot_path}")
        print(f"   Size: {len(screenshot_bytes):,} bytes")
        
        # Prepare context for VLM
        context_text = f"""
Page Title: {text_content['title']}

Headings:
{chr(10).join([f"- {h['level']}: {h['text']}" for h in text_content['headings'][:10]])}

Key Content:
{chr(10).join(text_content['paragraphs'][:5])}
"""
        
        # Analyze with VLM
        print("\n🧠 Sending to Qwen3 VL for analysis...")
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        vlm_result = await analyze_with_vlm(screenshot_b64, context_text)
        
        if "error" in vlm_result:
            print(f"❌ VLM Error: {vlm_result['error']}")
        else:
            print("\n" + "=" * 60)
            print("🤖 VLM ANALYSIS RESULT")
            print("=" * 60)
            print(vlm_result.get("analysis", "No analysis returned"))
            
            # Token usage
            usage = vlm_result.get("usage", {})
            if usage:
                print("\n" + "-" * 40)
                print("📊 TOKEN USAGE:")
                print(f"   Prompt tokens: {usage.get('prompt_tokens', 0):,}")
                print(f"   Completion tokens: {usage.get('completion_tokens', 0):,}")
                print(f"   Total tokens: {usage.get('total_tokens', 0):,}")
        
        # Save all extracted data
        output_data = {
            "url": TARGET_URL,
            "text_content": text_content,
            "images": images,
            "vlm_analysis": vlm_result
        }
        
        output_json_path = output_dir / "scraped_data.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Data saved to: {output_json_path}")
        
        await browser.close()
        print("\n✅ Scraping complete!")


if __name__ == "__main__":
    asyncio.run(scrape_with_vlm())
