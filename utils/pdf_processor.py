"""
PDF Processor - Downloads and extracts text from embedded PDFs
"""
import asyncio
import os
import aiohttp
from pypdf import PdfReader
from io import BytesIO
import re
from .content_cleaner import ContentCleaner

class PDFProcessor:
    """Handles downloading and text extraction from PDFs"""
    
    def __init__(self, output_dir: str = "pdf_downloads"):
        self.output_dir = output_dir
        self.cleaner = ContentCleaner()
        os.makedirs(output_dir, exist_ok=True)
        
    async def download_file(self, session: aiohttp.ClientSession, url: str) -> bytes:
        """Download a file from URL"""
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.read()
                print(f"❌ Failed to download {url}: Status {response.status}")
                return None
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text content from PDF bytes"""
        try:
            pdf_file = BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            text = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            
            return "\n\n".join(text)
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""

    async def process_urls(self, urls: list) -> list:
        """
        Download and process a list of PDF URLs
        Returns list of dict items compatible with scraped data structure
        """
        print(f"\n📄 Processing {len(urls)} PDFs...")
        results = []
        
        async with aiohttp.ClientSession() as session:
            for url in urls:
                print(f"   Downloading: {url}")
                content_bytes = await self.download_file(session, url)
                
                if content_bytes:
                    # Extract text
                    raw_text = self.extract_text_from_pdf(content_bytes)
                    
                    if raw_text.strip():
                        # Clean the text using our existing cleaner
                        clean_text = self.cleaner.clean_content(raw_text)
                        
                        pdf_data = {
                            "url": url,
                            "title": self._extract_filename(url),
                            "content": clean_text,
                            "media": {},
                            "metadata": {"type": "pdf", "source": "embedded_asset"}
                        }
                        results.append(pdf_data)
                        print(f"   ✅ Extracted {len(clean_text)} chars from PDF")
                    else:
                        print(f"   ⚠️ No text content found in PDF")
                
                # Modest delay
                await asyncio.sleep(0.5)
                
        return results

    def _extract_filename(self, url: str) -> str:
        """Get readable name from URL"""
        try:
            name = url.split('/')[-1].replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
            return f"{name} (PDF)"
        except:
            return "PDF Document"

if __name__ == "__main__":
    # Test with the specific requested PDF
    processor = PDFProcessor()
    url = "https://www.tombank.com.tr/assets/images/doc/urun_ve_hizmet_ucretleri.pdf"
    
    async def test():
        results = await processor.process_urls([url])
        if results:
            print("Successfully processed PDF!")
            print(results[0]['content'][:200])
            
    asyncio.run(test())
