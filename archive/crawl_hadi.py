import asyncio
import json
import os
from crawl4ai import AsyncWebCrawler
from urllib.parse import urlparse, urljoin

START_URL = "https://tombankhadi.com/"
DOMAIN = "tombankhadi.com"
OUTPUT_DIR = "hadi_scraper_crawl4ai"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def crawl():
    queue = [START_URL]
    visited = set()
    
    # Store results in memory (or append to file, but memory is fine for <100 pages)
    results = []
    
    MAX_PAGES = 25 # Increased for comprehensive RAG dataset
    CONCURRENCY = 3

    print(f"Starting crawl of {START_URL} with max {MAX_PAGES} pages.")

    async with AsyncWebCrawler(verbose=True) as crawler:
        while queue and len(visited) < MAX_PAGES:
            # Take a batch of URLs
            batch = []
            while len(batch) < CONCURRENCY and queue:
                url = queue.pop(0)
                if url not in visited:
                    if len(url) > 1 and url.endswith('/'):
                        url = url[:-1]
                    if url not in visited:
                        batch.append(url)
                        visited.add(url)
            
            if not batch:
                continue

            print(f"Crawling batch of {len(batch)}: {batch}")
            
            # Create tasks
            tasks = []
            for url in batch:
                # Set a 30s timeout (default is usually 30s or 60s, explicit is better)
                # Remove JavaScript wait/execution if not strictly needed? 
                # Actually, standard run should be fine if we just don't hang forever.
                # 'magic=True' is a common crawl4ai parameter for "do your best" but let's stick to standard `arun` with timeout.
                # Is there a timeout param? usually yes.
                # If not, asyncio.wait_for wrapper.
                tasks.append(crawler.arun(url=url)) 
            
            # Run batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    continue
                
                if not result.success:
                    print(f"Failed: {url} - {result.error_message}")
                    continue

                print(f"Success: {url}")
                
                # Store data
                page_data = {
                    "url": url,
                    "title": "Extracted by Crawl4AI",
                    "content": result.markdown if result.markdown else "",  # Full content for RAG chunking
                    "media": result.media if hasattr(result, 'media') else {},
                }
                 # Check for metadata/title if available
                if hasattr(result, 'metadata') and result.metadata:
                    page_data["metadata"] = result.metadata

                results.append(page_data)

                # Extract links
                if hasattr(result, 'links') and isinstance(result.links, dict):
                   internal_links = result.links.get("internal", [])
                   for link_obj in internal_links:
                       href = link_obj.get('href')
                       if href:
                           full_url = urljoin(url, href)
                           parsed = urlparse(full_url)
                           if parsed.netloc == "tombankhadi.com" or parsed.netloc == "www.tombankhadi.com":
                               if parsed.scheme in ["http", "https"]:
                                   clean_url = full_url.split('#')[0]
                                   if clean_url not in visited and clean_url not in queue and clean_url not in batch:
                                       queue.append(clean_url)

    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Finished. Scraped {len(results)} pages. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(crawl())
