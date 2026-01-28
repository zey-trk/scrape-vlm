"""
Comprehensive Web Scraper for TomBankHadi.com
Scrapes ALL pages including FAQ sections with proper timeout handling
"""
import asyncio
import json
import os
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from urllib.parse import urlparse, urljoin

from .config import BASE_URL as START_URL, PRIORITY_URLS, OUTPUT_FILE, MAX_PAGES, DOMAIN

# Configuration (Loaded from config.py)
# START_URL, PRIORITY_URLS, OUTPUT_FILE are imported directly


async def comprehensive_crawl():
    """
    Comprehensive crawl with better timeout handling and priority URL support
    """
    # Start with priority URLs + queue for discovered URLs
    queue = PRIORITY_URLS.copy()
    visited = set()
    results = []
    
    MAX_PAGES = 100  # Increased significantly
    CONCURRENCY = 2  # Lower concurrency for stability
    PAGE_TIMEOUT = 30000  # 30 seconds timeout
    
    print(f"🚀 Starting comprehensive crawl of {START_URL}")
    print(f"   Max pages: {MAX_PAGES}")
    print(f"   Priority URLs: {len(PRIORITY_URLS)}")
    
    # Browser config for better compatibility
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )
    
    # Crawler run config with timeout
    run_config = CrawlerRunConfig(
        page_timeout=PAGE_TIMEOUT,
        wait_until="domcontentloaded",
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        success_count = 0
        error_count = 0
        
        while queue and len(visited) < MAX_PAGES:
            # Take a batch of URLs
            batch = []
            while len(batch) < CONCURRENCY and queue:
                url = queue.pop(0)
                # Normalize URL
                if url.endswith('/') and len(url) > 1:
                    url = url[:-1]
                if url not in visited:
                    batch.append(url)
                    visited.add(url)
            
            if not batch:
                continue
            
            print(f"\n📥 Crawling batch [{len(visited)}/{MAX_PAGES}]: {len(batch)} URLs")
            for u in batch:
                print(f"   → {u}")
            
            # Create tasks with individual timeouts
            for url in batch:
                try:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=run_config),
                        timeout=45  # 45 second overall timeout
                    )
                    
                    if result.success:
                        success_count += 1
                        content = result.markdown if result.markdown else ""
                        
                        print(f"   ✅ {url} ({len(content)} chars)")
                        
                        # Store page data
                        page_data = {
                            "url": url,
                            "title": result.metadata.get("title", "No Title") if result.metadata else "No Title",
                            "content": content,
                            "media": result.media if hasattr(result, 'media') else {},
                            "metadata": result.metadata if hasattr(result, 'metadata') else {},
                        }
                        results.append(page_data)
                        
                        # Extract new links
                        if hasattr(result, 'links') and isinstance(result.links, dict):
                            internal_links = result.links.get("internal", [])
                            for link_obj in internal_links:
                                href = link_obj.get('href')
                                if href:
                                    full_url = urljoin(url, href)
                                    parsed = urlparse(full_url)
                                    if parsed.netloc in [DOMAIN, f"www.{DOMAIN}"]:
                                        if parsed.scheme in ["http", "https"]:
                                            clean_url = full_url.split('#')[0]
                                            if clean_url.endswith('/'):
                                                clean_url = clean_url[:-1]
                                            if clean_url not in visited and clean_url not in queue:
                                                queue.append(clean_url)
                    else:
                        error_count += 1
                        print(f"   ❌ {url} - Failed: {result.error_message}")
                        
                except asyncio.TimeoutError:
                    error_count += 1
                    print(f"   ⏰ {url} - Timeout (skipping)")
                except Exception as e:
                    error_count += 1
                    print(f"   ❌ {url} - Error: {str(e)[:50]}")
            
            # Small delay between batches
            await asyncio.sleep(0.5)
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✨ SCRAPING COMPLETE")
    print(f"   Success: {success_count} pages")
    print(f"   Errors: {error_count} pages")
    print(f"   Total content: {sum(len(p['content']) for p in results):,} characters")
    print(f"   Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    asyncio.run(comprehensive_crawl())
