"""
VLM-Guided Web Extraction System
Extracts structured data from web pages using Playwright + Qwen3 VL with custom instruction prompts.
"""
import asyncio
import base64
import os
import json
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass, field
from playwright.async_api import async_playwright
import aiohttp
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExtractionResult:
    """Result of a VLM extraction."""
    url: str
    instruction: str
    raw_response: str
    structured_data: dict | list | None
    usage: dict = field(default_factory=dict)
    screenshot_path: Optional[str] = None
    error: Optional[str] = None


class PromptBuilder:
    """Builds optimized prompts for VLM extraction."""
    
    SYSTEM_PROMPTS = {
        "json": """You are a precise data extraction assistant. Your task is to extract specific information from the provided webpage screenshot.

CRITICAL RULES:
1. Extract ONLY the information requested in the user's instruction
2. Return your response as valid JSON - no markdown code blocks, no explanations outside the JSON
3. Use descriptive field names in Turkish if the content is in Turkish
4. If a value is not found, use null instead of making up data
5. For numerical values, extract them as numbers when possible
6. For percentages, include the % symbol in the string

Your response must be ONLY valid JSON, starting with { or [""",

        "table": """You are a precise data extraction assistant. Your task is to extract specific information from the provided webpage screenshot.

CRITICAL RULES:
1. Extract ONLY the information requested in the user's instruction
2. Return your response as a markdown table
3. Use Turkish headers if the content is in Turkish
4. Include all relevant columns for comparison
5. If a value is not found, use "-" in the cell

Your response must be ONLY a markdown table, starting with |""",

        "markdown": """You are a precise data extraction assistant. Your task is to extract specific information from the provided webpage screenshot.

CRITICAL RULES:
1. Extract ONLY the information requested in the user's instruction
2. Return your response as structured markdown
3. Use Turkish headings if the content is in Turkish
4. Be concise but comprehensive
5. Use bullet points and sub-sections for clarity"""
    }
    
    def build(
        self, 
        instruction: str, 
        output_format: Literal["json", "table", "markdown"],
        context: Optional[str] = None
    ) -> list[dict]:
        """Build the message array for VLM API call."""
        system_prompt = self.SYSTEM_PROMPTS.get(output_format, self.SYSTEM_PROMPTS["json"])
        
        user_content = f"""EXTRACTION INSTRUCTION:
{instruction}

Please analyze the screenshot and extract the requested information."""
        
        if context:
            user_content += f"\n\nPAGE CONTEXT (for reference):\n{context[:500]}"
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]


class VLMExtractor:
    """Main class for VLM-guided web data extraction."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://us-east-a2.ai.ubicloud.com/v1/chat/completions",
        model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        output_dir: Optional[Path] = None
    ):
        self.api_key = api_key or os.getenv("ubicloud_api_key")
        self.endpoint = endpoint
        self.model = model
        self.output_dir = output_dir or Path(__file__).parent.parent / "scrape_output"
        self.output_dir.mkdir(exist_ok=True)
        self.prompt_builder = PromptBuilder()
        
        if not self.api_key:
            raise ValueError("API key not found. Set 'ubicloud_api_key' in .env or pass api_key parameter.")
    
    async def scrape_page(self, url: str, screenshot_name: str = "page") -> tuple[bytes, str, dict]:
        """
        Scrape a page with Playwright and return screenshot + text content.
        
        Returns:
            tuple: (screenshot_bytes, text_content, page_metadata)
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(2)  # Allow dynamic content to render
            
            # Capture screenshot
            screenshot_path = self.output_dir / f"{screenshot_name}.png"
            screenshot_bytes = await page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Extract text for context
            text_content = await page.evaluate("() => document.body.innerText")
            title = await page.title()
            
            metadata = {
                "title": title,
                "url": url,
                "screenshot_path": str(screenshot_path)
            }
            
            await browser.close()
            return screenshot_bytes, text_content, metadata
    
    async def _call_vlm(
        self,
        image_base64: str,
        messages: list[dict],
        max_tokens: int = 4000
    ) -> dict:
        """Make API call to VLM."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Build the content with image
        user_message = messages[-1]
        multimodal_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            },
            {
                "type": "text",
                "text": user_message["content"]
            }
        ]
        
        # Reconstruct messages with image
        api_messages = []
        for msg in messages[:-1]:  # System and other messages
            api_messages.append(msg)
        api_messages.append({"role": "user", "content": multimodal_content})
        
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint, 
                headers=headers, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=180)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {"error": f"VLM request failed: {response.status} - {error_text}"}
                
                result = await response.json()
                return {
                    "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "usage": result.get("usage", {})
                }
    
    async def extract(
        self,
        url: str,
        instruction: str,
        output_format: Literal["json", "table", "markdown"] = "json",
        screenshot_name: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract structured data from a webpage using VLM.
        
        Args:
            url: The webpage URL to scrape
            instruction: Custom extraction instruction (can be in Turkish)
            output_format: Desired output format - "json", "table", or "markdown"
            screenshot_name: Optional name for the screenshot file
            
        Returns:
            ExtractionResult with extracted data
        """
        # Generate screenshot name from URL if not provided
        if not screenshot_name:
            screenshot_name = url.split("/")[-1].replace("-", "_")[:30] or "page"
        
        # Scrape the page
        screenshot_bytes, text_content, metadata = await self.scrape_page(url, screenshot_name)
        
        # Build prompt
        messages = self.prompt_builder.build(
            instruction=instruction,
            output_format=output_format,
            context=text_content[:1000]
        )
        
        # Call VLM
        image_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        vlm_response = await self._call_vlm(image_base64, messages)
        
        if "error" in vlm_response:
            return ExtractionResult(
                url=url,
                instruction=instruction,
                raw_response="",
                structured_data=None,
                error=vlm_response["error"],
                screenshot_path=metadata["screenshot_path"]
            )
        
        raw_response = vlm_response["content"]
        
        # Parse structured data for JSON format
        structured_data = None
        if output_format == "json":
            try:
                # Clean up response - remove any markdown code blocks
                clean_response = raw_response.strip()
                if clean_response.startswith("```"):
                    clean_response = clean_response.split("\n", 1)[1]
                if clean_response.endswith("```"):
                    clean_response = clean_response.rsplit("```", 1)[0]
                clean_response = clean_response.strip()
                
                structured_data = json.loads(clean_response)
            except json.JSONDecodeError:
                # Keep raw response if JSON parsing fails
                structured_data = {"raw": raw_response}
        else:
            structured_data = {"content": raw_response}
        
        return ExtractionResult(
            url=url,
            instruction=instruction,
            raw_response=raw_response,
            structured_data=structured_data,
            usage=vlm_response.get("usage", {}),
            screenshot_path=metadata["screenshot_path"]
        )
    
    async def batch_extract(
        self,
        urls: list[str],
        instruction: str,
        output_format: Literal["json", "table", "markdown"] = "json"
    ) -> list[ExtractionResult]:
        """Extract data from multiple URLs with the same instruction."""
        results = []
        for i, url in enumerate(urls):
            result = await self.extract(
                url=url,
                instruction=instruction,
                output_format=output_format,
                screenshot_name=f"page_{i+1}"
            )
            results.append(result)
        return results
    
    def save_results(self, result: ExtractionResult, filename: str = "extraction_result.json"):
        """Save extraction results to a JSON file."""
        output_path = self.output_dir / filename
        
        data = {
            "url": result.url,
            "instruction": result.instruction,
            "structured_data": result.structured_data,
            "raw_response": result.raw_response,
            "usage": result.usage,
            "screenshot_path": result.screenshot_path,
            "error": result.error
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return output_path
