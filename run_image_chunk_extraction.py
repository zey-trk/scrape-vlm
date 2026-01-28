"""
VLM Web Extraction Pipeline
A reusable system for extracting structured data from any URL using custom instructions.

Usage:
    python run_extraction.py --url "https://example.com/page" --instruction "Extract all products..."
    
Or in code:
    from run_extraction import VLMPipeline
    
    pipeline = VLMPipeline()
    result = await pipeline.run(
        url="https://example.com",
        instruction="Your extraction instruction...",
        output_name="my_extraction"
    )
"""
import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.vlm_extractor import VLMExtractor
import pandas as pd


@dataclass
class Chunk:
    """A RAG-ready chunk."""
    content: str
    title: str
    category: str
    source_url: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of the extraction pipeline."""
    url: str
    instruction: str
    chunks: list[Chunk]
    json_path: str
    xlsx_path: str
    screenshot_path: str
    token_usage: dict
    

class GenericChunker:
    """Creates RAG-ready chunks from VLM extraction results."""
    
    @staticmethod
    def create_chunks(
        extraction_data: dict,
        chunk_strategy: str = "auto"
    ) -> list[Chunk]:
        """
        Create chunks from extraction data.
        
        Args:
            extraction_data: The structured_data from VLM extraction
            chunk_strategy: "auto" | "flat" | "nested"
            
        Returns:
            List of Chunk objects
        """
        source_url = extraction_data.get("url", "")
        structured = extraction_data.get("structured_data", {})
        
        if not structured:
            # Fallback to raw response
            return [Chunk(
                content=extraction_data.get("raw_response", "No content"),
                title="Extracted Content",
                category="raw",
                source_url=source_url
            )]
        
        # Auto-detect structure and chunk accordingly
        chunks = []
        
        # Handle list of items (most common pattern)
        for key, value in structured.items():
            if isinstance(value, list):
                # Each list item becomes a chunk
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        # Find a title field
                        title = (
                            item.get("title") or 
                            item.get("name") or 
                            item.get("kampanya_adi") or
                            item.get("baslik") or
                            f"{key} #{i+1}"
                        )
                        
                        # Create readable content
                        content_lines = [f"# {title}", ""]
                        for k, v in item.items():
                            if isinstance(v, dict):
                                content_lines.append(f"## {k.replace('_', ' ').title()}")
                                for sub_k, sub_v in v.items():
                                    label = sub_k.replace('_', ' ').title()
                                    content_lines.append(f"- {label}: {sub_v}")
                                content_lines.append("")
                            elif k not in ["title", "name", "kampanya_adi", "baslik"]:
                                label = k.replace('_', ' ').title()
                                content_lines.append(f"- {label}: {v}")
                        
                        chunks.append(Chunk(
                            content="\n".join(content_lines),
                            title=str(title),
                            category=key,
                            source_url=source_url,
                            metadata={"raw_item": item, "list_key": key}
                        ))
                    else:
                        # Simple list item
                        chunks.append(Chunk(
                            content=str(item),
                            title=f"{key} #{i+1}",
                            category=key,
                            source_url=source_url
                        ))
            
            elif isinstance(value, dict):
                # Dictionary becomes one chunk
                content_lines = [f"# {key.replace('_', ' ').title()}", ""]
                for k, v in value.items():
                    label = k.replace('_', ' ').title()
                    content_lines.append(f"- {label}: {v}")
                
                chunks.append(Chunk(
                    content="\n".join(content_lines),
                    title=key.replace('_', ' ').title(),
                    category=key,
                    source_url=source_url,
                    metadata={"raw_data": value}
                ))
            
            elif isinstance(value, (str, int, float)):
                # Simple value
                chunks.append(Chunk(
                    content=f"{key.replace('_', ' ').title()}: {value}",
                    title=key.replace('_', ' ').title(),
                    category="info",
                    source_url=source_url
                ))
        
        return chunks
    
    @staticmethod
    def export_to_xlsx(chunks: list[Chunk], output_path: str) -> str:
        """Export chunks to Excel file."""
        
        # Main chunks sheet
        main_data = []
        for i, chunk in enumerate(chunks, 1):
            main_data.append({
                "chunk_id": i,
                "title": chunk.title,
                "category": chunk.category,
                "content": chunk.content,
                "char_count": len(chunk.content),
                "word_count": len(chunk.content.split()),
                "source_url": chunk.source_url
            })
        
        df_main = pd.DataFrame(main_data)
        
        # Statistics
        stats = {
            "Metric": ["Total Chunks", "Avg Chars", "Avg Words", "Categories"],
            "Value": [
                len(chunks),
                round(df_main["char_count"].mean(), 1) if len(chunks) > 0 else 0,
                round(df_main["word_count"].mean(), 1) if len(chunks) > 0 else 0,
                ", ".join(df_main["category"].unique())
            ]
        }
        df_stats = pd.DataFrame(stats)
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='Chunks', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value or "")) for cell in column)
                    column_letter = column[0].column_letter
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 60)
        
        return output_path


class VLMPipeline:
    """
    Complete VLM extraction pipeline.
    
    Takes a URL and instruction, extracts data using Qwen3 VL,
    creates RAG-ready chunks, and exports to JSON + XLSX.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "scrape_output"
        self.output_dir.mkdir(exist_ok=True)
        self.extractor = VLMExtractor(output_dir=self.output_dir)
        self.chunker = GenericChunker()
    
    async def run(
        self,
        url: str,
        instruction: str,
        output_name: str = "extraction",
        output_format: str = "json"
    ) -> PipelineResult:
        """
        Run the complete extraction pipeline.
        
        Args:
            url: The webpage URL to extract from
            instruction: Custom extraction instruction (can be in Turkish)
            output_name: Base name for output files (without extension)
            output_format: VLM output format - "json", "table", or "markdown"
            
        Returns:
            PipelineResult with all outputs and paths
        """
        print("\n" + "=" * 70)
        print("🚀 VLM EXTRACTION PIPELINE")
        print("=" * 70)
        print(f"📍 URL: {url}")
        print(f"📝 Instruction: {instruction[:100]}...")
        
        # Step 1: Extract with VLM
        print("\n[1/3] 🧠 Extracting with Qwen3 VL...")
        extraction_result = await self.extractor.extract(
            url=url,
            instruction=instruction,
            output_format=output_format,
            screenshot_name=output_name
        )
        
        if extraction_result.error:
            raise RuntimeError(f"Extraction failed: {extraction_result.error}")
        
        print(f"      ✅ Extracted (tokens: {extraction_result.usage.get('total_tokens', 0):,})")
        
        # Step 2: Create chunks
        print("[2/3] 🔧 Creating RAG chunks...")
        
        extraction_data = {
            "url": url,
            "structured_data": extraction_result.structured_data,
            "raw_response": extraction_result.raw_response
        }
        
        chunks = self.chunker.create_chunks(extraction_data)
        print(f"      ✅ Created {len(chunks)} chunks")
        
        # Step 3: Export
        print("[3/3] 💾 Exporting...")
        
        # Save JSON
        json_path = self.output_dir / f"{output_name}.json"
        json_data = {
            "url": url,
            "instruction": instruction,
            "timestamp": datetime.now().isoformat(),
            "structured_data": extraction_result.structured_data,
            "raw_response": extraction_result.raw_response,
            "usage": extraction_result.usage,
            "chunks": [
                {
                    "title": c.title,
                    "category": c.category,
                    "content": c.content,
                    "source_url": c.source_url
                }
                for c in chunks
            ]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"      📄 JSON: {json_path}")
        
        # Save XLSX
        xlsx_path = self.output_dir / f"{output_name}_chunks.xlsx"
        self.chunker.export_to_xlsx(chunks, str(xlsx_path))
        print(f"      📊 XLSX: {xlsx_path}")
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        
        return PipelineResult(
            url=url,
            instruction=instruction,
            chunks=chunks,
            json_path=str(json_path),
            xlsx_path=str(xlsx_path),
            screenshot_path=extraction_result.screenshot_path,
            token_usage=extraction_result.usage
        )


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VLM Web Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract campaigns from a bank page
  python run_extraction.py \\
    --url "https://tombankhadi.com/hadi-ozel-bankacilik/ozel-bankacilik-segmentlerimiz" \\
    --instruction "Sayfadaki kampanyaları ve segment bilgilerini çıkar" \\
    --output campaigns
    
  # Extract product info
  python run_extraction.py \\
    --url "https://example.com/products" \\
    --instruction "Extract all product names, prices, and descriptions" \\
    --output products
        """
    )
    
    parser.add_argument("--url", "-u", required=True, help="URL to extract from")
    parser.add_argument("--instruction", "-i", required=True, help="Extraction instruction")
    parser.add_argument("--output", "-o", default="extraction", help="Output file base name")
    parser.add_argument("--format", "-f", default="json", choices=["json", "table", "markdown"],
                       help="VLM output format")
    
    args = parser.parse_args()
    
    pipeline = VLMPipeline()
    result = await pipeline.run(
        url=args.url,
        instruction=args.instruction,
        output_name=args.output,
        output_format=args.format
    )
    
    print(f"\n📊 Created {len(result.chunks)} chunks")
    print(f"📄 JSON: {result.json_path}")
    print(f"📊 XLSX: {result.xlsx_path}")
    print(f"📸 Screenshot: {result.screenshot_path}")


if __name__ == "__main__":
    asyncio.run(main())
