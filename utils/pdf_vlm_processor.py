"""
VLM PDF Processor
Uses Vision-Language Models (like GPT-4o) to extract structured content from PDFs.
Converts PDF pages to images -> Sends to VLM -> Returns Markdown
"""
import asyncio
import os
import base64
from io import BytesIO
from typing import List, Optional, Dict
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image not installed")
except Exception:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ poppler not installed or not found in PATH")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VLMPDFProcessor:
    """Processor for extracting content from PDFs using Vision Models (OpenAI or Ubicloud)"""
    
    def __init__(self, provider: str = "ubicloud"):
        self.provider = provider
        self.client = None
        self.model = None
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and OPENAI_AVAILABLE:
                self.client = AsyncOpenAI(api_key=api_key)
                self.model = "gpt-4o"
            else:
                print("⚠️ OpenAI API Key missing or package not installed")

        elif provider == "ubicloud":
            api_key = os.getenv("ubicloud_api_key")
            if api_key and OPENAI_AVAILABLE:
                # Ubicloud is OpenAI-compatible but needs custom base_url
                self.client = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://us-east-a2.ai.ubicloud.com/v1"
                )
                self.model = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
            else:
                print("⚠️ Ubicloud API Key missing (ubicloud_api_key)")
        
    def is_available(self) -> bool:
        """Check if VLM processing is available"""
        return bool(self.client and PDF2IMAGE_AVAILABLE)

    def _encode_image(self, image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def _process_page(self, image, page_num: int) -> dict:
        """Process a single page and return text + usage stats"""
        if not self.client:
            return {"text": "", "usage": {}}

        base64_image = self._encode_image(image)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a Data Extraction Expert optimizing content for RAG (Retrieval Augmented Generation). "
                            "Do NOT output Markdown tables. Instead, convert all information into self-contained, explicit semantic sentences. "
                            "Ensure every sentence includes the full context (Subject + Action + Value). "
                            "Example: instead of '| 500 TL | Free |', write 'For FAST transfers under 500 TL, the transaction fee is Free.' "
                            "Structure with clear headers (##) and bullet points."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Extract content from page {page_num}. Flatten tables into sentences."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}
            
            return {"text": content, "usage": usage}
            
        except Exception as e:
            print(f"❌ Error processing page {page_num} with VLM: {e}")
            return {"text": "", "usage": {}}

    async def process_pdf(self, pdf_content: bytes, document_name: str = "document") -> dict:
        """
        Convert PDF to images and extract text using VLM.
        Returns dict with 'content' (str) and 'total_usage' (dict).
        """
        if not self.is_available():
            print("❌ VLM Processor not available")
            return {"content": "", "total_usage": {}}

        print(f"🔍 VLM Processing ({self.provider}): Converting {document_name}...")
        
        try:
            images = convert_from_bytes(pdf_content, dpi=200, fmt='jpeg', thread_count=2)
            print(f"   Converted to {len(images)} images. Using model: {self.model}")
            
            full_text = []
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for i, image in enumerate(images):
                print(f"   Analyzing page {i+1}/{len(images)}...")
                result = await self._process_page(image, i+1)
                
                text = result["text"]
                usage = result["usage"]
                
                full_text.append(f"## Page {i+1}\n\n{text}")
                
                # Aggregate usage
                if usage:
                    total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    total_usage["total_tokens"] += usage.get("total_tokens", 0)
            
            combined_text = "\n\n---\n\n".join(full_text)
            
            return {
                "content": combined_text,
                "total_usage": total_usage
            }
            
        except Exception as e:
            print(f"❌ Error in VLM pipeline for {document_name}: {e}")
            return {"content": "", "total_usage": {}}

# Implementation helper to test VLM independently
if __name__ == "__main__":
    import asyncio
    
    async def test():
        processor = VLMPDFProcessor()
        if processor.is_available():
            print("✅ VLM Processor is ready!")
        else:
            print("❌ VLM Processor is NOT ready.")
            if not PDF2IMAGE_AVAILABLE: print("   - Missing pdf2image/poppler")
            if not processor.client: print("   - Missing OpenAI API Key")

    asyncio.run(test())
