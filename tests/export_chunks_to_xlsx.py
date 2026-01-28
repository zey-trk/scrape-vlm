"""
Export VLM Extraction to RAG Chunks (XLSX)
Transforms extracted JSON data into searchable chunks for RAG systems.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Chunk:
    """A RAG-ready chunk."""
    content: str
    title: str
    category: str
    source_url: str
    metadata: dict = field(default_factory=dict)


class ExtractionChunker:
    """Transforms VLM extraction JSON into RAG-ready chunks."""
    
    @staticmethod
    def chunk_campaign_extraction(json_path: str) -> list[Chunk]:
        """
        Create chunks from campaign extraction JSON.
        
        Strategy:
        - One chunk per campaign with full segment details
        - One chunk for segment criteria overview
        - Optional: granular chunks per campaign+segment combination
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        source_url = data.get("url", "")
        structured = data.get("structured_data", {})
        chunks = []
        
        # --- Campaign chunks (comprehensive) ---
        for campaign in structured.get("kampanyalar", []):
            campaign_name = campaign.get("kampanya_adi", "Kampanya")
            
            # Build comprehensive content
            content_lines = [
                f"# {campaign_name}",
                "",
                "Bu kampanya, Hadi Özel Bankacılık müşterilerine özel avantajlar sunar.",
                ""
            ]
            
            for segment, segment_name in [
                ("elite", "Elite"),
                ("elite_plus", "Elite Plus"),
                ("prestige", "Prestige")
            ]:
                details = campaign.get(segment, {})
                if details:
                    content_lines.append(f"## {segment_name} Segmenti")
                    content_lines.append(f"- İade Oranı: {details.get('iade_orani', '-')}")
                    content_lines.append(f"- Günlük Limit: {details.get('gunluk_limit', '-')}")
                    content_lines.append(f"- Aylık Limit: {details.get('aylik_limit', '-')}")
                    content_lines.append("")
            
            chunks.append(Chunk(
                content="\n".join(content_lines),
                title=campaign_name,
                category="kampanya",
                source_url=source_url,
                metadata={
                    "extraction_type": "campaign",
                    "campaign_name": campaign_name,
                    "raw_data": campaign
                }
            ))
        
        # --- Granular chunks (per campaign + segment) ---
        for campaign in structured.get("kampanyalar", []):
            campaign_name = campaign.get("kampanya_adi", "Kampanya")
            
            for segment, segment_name in [
                ("elite", "Elite"),
                ("elite_plus", "Elite Plus"),
                ("prestige", "Prestige")
            ]:
                details = campaign.get(segment, {})
                if details:
                    content = (
                        f"{campaign_name} - {segment_name} Segment Avantajları:\n"
                        f"İade oranı {details.get('iade_orani', '-')} olarak uygulanır. "
                        f"Günlük harcama limiti {details.get('gunluk_limit', '-')}, "
                        f"aylık limit ise {details.get('aylik_limit', '-')} şeklindedir."
                    )
                    
                    chunks.append(Chunk(
                        content=content,
                        title=f"{campaign_name} - {segment_name}",
                        category="kampanya_segment",
                        source_url=source_url,
                        metadata={
                            "campaign_name": campaign_name,
                            "segment": segment_name,
                            "discount_rate": details.get('iade_orani', ''),
                            "daily_limit": details.get('gunluk_limit', ''),
                            "monthly_limit": details.get('aylik_limit', '')
                        }
                    ))
        
        # --- Segment criteria chunk ---
        segment_criteria = structured.get("segment_kriterleri", {})
        if segment_criteria:
            content_lines = [
                "# Hadi Özel Bankacılık Segment Kriterleri",
                "",
                "Müşteriler, aşağıdaki bakiye kriterlerine göre segmentlere ayrılır:",
                ""
            ]
            
            for segment, segment_name in [
                ("elite", "Elite"),
                ("elite_plus", "Elite Plus"),
                ("prestige", "Prestige")
            ]:
                criteria = segment_criteria.get(segment, "")
                if criteria:
                    content_lines.append(f"- **{segment_name}**: {criteria}")
            
            content_lines.append("")
            content_lines.append("Birikim hesaplaması TL, yabancı para ve altın hesapları toplamı üzerinden yapılır.")
            
            chunks.append(Chunk(
                content="\n".join(content_lines),
                title="Segment Kriterleri",
                category="segment",
                source_url=source_url,
                metadata={
                    "extraction_type": "segment_criteria",
                    "raw_data": segment_criteria
                }
            ))
        
        # --- Q&A style chunks (good for RAG retrieval) ---
        qa_chunks = [
            {
                "question": "Özel bankacılık için minimum ne kadar bakiye gerekiyor?",
                "answer": f"Özel bankacılık Elite segmenti için minimum {segment_criteria.get('elite', '2.5 Milyon TL')} bakiye gerekmektedir."
            },
            {
                "question": "Prestige segmentine nasıl geçilir?",
                "answer": f"Prestige segmenti için {segment_criteria.get('prestige', '15 Milyon TL ve üzeri')} bakiye gerekmektedir."
            }
        ]
        
        for qa in qa_chunks:
            chunks.append(Chunk(
                content=f"Soru: {qa['question']}\nCevap: {qa['answer']}",
                title=qa['question'][:50],
                category="qa",
                source_url=source_url,
                metadata={"qa_type": "faq"}
            ))
        
        return chunks
    
    @staticmethod
    def export_to_xlsx(chunks: list[Chunk], output_path: str) -> str:
        """Export chunks to Excel file with multiple sheets."""
        
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
        
        # Metadata sheet
        meta_data = []
        for i, chunk in enumerate(chunks, 1):
            meta_data.append({
                "chunk_id": i,
                "title": chunk.title,
                "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False)
            })
        
        df_meta = pd.DataFrame(meta_data)
        
        # Statistics sheet
        stats = {
            "Metric": [
                "Total Chunks",
                "Total Characters",
                "Total Words",
                "Avg Chars per Chunk",
                "Avg Words per Chunk",
                "Categories"
            ],
            "Value": [
                len(chunks),
                sum(len(c.content) for c in chunks),
                sum(len(c.content.split()) for c in chunks),
                round(sum(len(c.content) for c in chunks) / len(chunks), 1),
                round(sum(len(c.content.split()) for c in chunks) / len(chunks), 1),
                ", ".join(set(c.category for c in chunks))
            ]
        }
        df_stats = pd.DataFrame(stats)
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='Chunks', index=False)
            df_meta.to_excel(writer, sheet_name='Metadata', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value or "")) for cell in column)
                    column_letter = column[0].column_letter
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 60)
        
        return output_path


def main():
    """Process campaign extraction and export to XLSX."""
    print("\n" + "=" * 60)
    print("📊 VLM EXTRACTION → RAG CHUNKS → XLSX")
    print("=" * 60)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "scrape_output" / "campaign_extraction.json"
    output_path = base_dir / "scrape_output" / "rag_chunks.xlsx"
    
    if not json_path.exists():
        print(f"❌ Extraction file not found: {json_path}")
        print("   Run test_guided_extraction.py first!")
        return
    
    print(f"📄 Input: {json_path}")
    
    # Create chunks
    print("\n🔧 Creating chunks...")
    chunker = ExtractionChunker()
    chunks = chunker.chunk_campaign_extraction(str(json_path))
    
    print(f"   Created {len(chunks)} chunks:")
    for category in set(c.category for c in chunks):
        count = sum(1 for c in chunks if c.category == category)
        print(f"   - {category}: {count} chunks")
    
    # Export to XLSX
    print(f"\n📝 Exporting to: {output_path}")
    chunker.export_to_xlsx(chunks, str(output_path))
    
    # Preview
    print("\n" + "-" * 60)
    print("📋 CHUNK PREVIEW")
    print("-" * 60)
    
    for chunk in chunks[:3]:
        print(f"\n[{chunk.category}] {chunk.title}")
        print(f"   {chunk.content[:100]}...")
    
    print(f"\n✅ Exported {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
