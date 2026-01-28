"""
PDF Content Integrator
Scans for PDF links in scraped data, downloads them, extracts text,
and adds them as new entries to the dataset.
"""
import asyncio
import json
import re
import os
from utils.pdf_processor import PDFProcessor
from utils.config import OUTPUT_FILE, BACKUP_FILE

INPUT_FILE = OUTPUT_FILE

async def integrate_pdfs():
    """Find PDFs in scraped data and add their content"""
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file {INPUT_FILE} not found!")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Find all Unique PDF Links
    pdf_links = set()
    print("Scanning for PDF links...")
    
    for page in data:
        content = page.get('content', '')
        # Regex for .pdf links in markdown: [text](url.pdf) or just url.pdf
        # We look for http...pdf pattern inside parentheses or standalone
        
        # Pattern 1: Markdown links [text](http...pdf)
        md_links = re.findall(r'\]\((https?://[^)]+\.pdf)\)', content, re.IGNORECASE)
        for link in md_links:
            pdf_links.add(link)
            
        # Pattern 2: Explicit internal links from crawl4ai structure (if any)
        # (Already handled by regex on content usually, but good to be thorough if they exist in metadata)

    print(f"Found {len(pdf_links)} unique PDF links.")
    
    if not pdf_links:
        print("No PDFs found to process.")
        return

    # 2. Filter out PDFs we already have (if any)
    existing_urls = set(p['url'] for p in data)
    new_links = [link for link in pdf_links if link not in existing_urls]
    
    print(f"New PDFs to process: {len(new_links)}")
    if not new_links:
        print("All PDFs already in dataset.")
        return

    # 3. Process PDFs
    from utils.pdf_processor import PDFProcessor
    from utils.pdf_vlm_processor import VLMPDFProcessor
    
    standard_processor = PDFProcessor()
    
    # Initialize VLM (Prefer Ubicloud, fallback to OpenAI)
    vlm_provider = "ubicloud" if os.getenv("ubicloud_api_key") else "openai"
    vlm_processor = VLMPDFProcessor(provider=vlm_provider)
    
    # Check if VLM is available
    use_vlm = vlm_processor.is_available()
    if use_vlm:
        print(f"✨ VLM Processor is ENABLED using {vlm_provider.upper()} (Table extraction will be improved)")
    else:
        print("⚠️ VLM Processor is DISABLED (Standard text extraction only)")

    new_pages = []
    
    # Process sequentially to allow mixing strategies
    for url in new_links:
        print(f"\n📄 Processing: {url.split('/')[-1]}")
        
        # Decide strategy: Use VLM for known complex docs or if it looks like a table/fee doc
        is_complex = "ucret" in url.lower() or "tarife" in url.lower() or "tablo" in url.lower()
        
        processed_page = None
        
        # Strategy 1: VLM (if complex and available)
        if use_vlm and is_complex:
            print("   ↳ 🧠 Strategy: VLM (Complex Document)")
            # Download first
            async with aiohttp.ClientSession() as session:
                content = await standard_processor.download_file(session, url)
                if content:
                    text = await vlm_processor.process_pdf(content, url.split('/')[-1])
                    if text:
                         processed_page = {
                            "url": url,
                            "title": standard_processor._extract_filename(url) + " (VLM)",
                            "content": text,
                            "media": {},
                            "metadata": {"type": "pdf", "source": "embedded_asset", "extraction": "vlm_gpt4o"}
                        }
        
        # Strategy 2: Standard (Fallback or default)
        if not processed_page:
            strategy_name = "Standard" if not is_complex else "Standard (Fallback)"
            print(f"   ↳ 📄 Strategy: {strategy_name}")
            # Use the bulk processor's logic for single file
            batch_result = await standard_processor.process_urls([url])
            if batch_result:
                processed_page = batch_result[0]

        if processed_page:
            new_pages.append(processed_page)
        
    if not new_pages:
        print("No content extracted from PDFs.")
        return

    # 4. Integrate and Save
    print(f"Adding {len(new_pages)} PDF documents to dataset...")
    data.extend(new_pages)
    
    # Backup original
    if not os.path.exists(BACKUP_FILE):
        import shutil
        shutil.copy(INPUT_FILE, BACKUP_FILE)
        print(f"Backed up original data to {BACKUP_FILE}")
    
    # Save updated
    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Successfully updated {INPUT_FILE} with PDF content!")
    print(f"   Total pages now: {len(data)}")

if __name__ == "__main__":
    asyncio.run(integrate_pdfs())
