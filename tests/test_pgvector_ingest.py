"""
Test script for PGVector Knowledge Base ingestion and search.
Demonstrates ingesting campaign extraction and querying in Turkish.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pgvector_store import PGVectorStore


async def test_pgvector_ingest():
    """Test ingestion and search of campaign extraction."""
    print("\n" + "=" * 70)
    print("🗄️ PGVECTOR KNOWLEDGE BASE TEST")
    print("=" * 70)
    
    # Path to extraction file
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scrape_output",
        "campaign_extraction.json"
    )
    
    if not os.path.exists(json_path):
        print(f"❌ Extraction file not found: {json_path}")
        print("   Run test_guided_extraction.py first!")
        return
    
    print(f"📄 Source file: {json_path}")
    
    # Initialize store
    try:
        store = PGVectorStore()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install: pip install asyncpg sentence-transformers")
        return
    
    try:
        # Connect to database
        print("\n🔌 Connecting to PostgreSQL...")
        await store.connect()
        
        # Create schema
        print("📐 Creating schema...")
        await store.create_schema()
        
        # Ingest extraction
        print("\n📥 Ingesting campaign extraction...")
        ids = await store.ingest_extraction(json_path, extraction_type="campaign")
        print(f"   Inserted {len(ids)} documents")
        
        # Get document count
        count = await store.get_document_count()
        print(f"\n📊 Total documents in KB: {count}")
        
        # Test searches
        print("\n" + "=" * 70)
        print("🔍 TESTING SEMANTIC SEARCH")
        print("=" * 70)
        
        test_queries = [
            "Elite müşteriler için restoran indirimi nedir?",
            "Prestige segmenti için ne kadar bakiye gerekiyor?",
            "Uçak bileti kampanyasında iade oranları",
            "Otopark indirim oranları",
        ]
        
        for query in test_queries:
            print(f"\n🔎 Query: \"{query}\"")
            print("-" * 50)
            
            results = await store.search(query, k=2)
            
            for i, result in enumerate(results, 1):
                print(f"\n  [{i}] {result['title']} (similarity: {result['similarity']:.3f})")
                print(f"      Category: {result['category']}")
                # Show first 150 chars of content
                content_preview = result['content'][:150].replace('\n', ' ')
                print(f"      Content: {content_preview}...")
        
        # Test category-filtered search
        print("\n" + "-" * 50)
        print("🏷️ Category-filtered search (category='kampanya'):")
        results = await store.search("indirim oranları", k=3, category="kampanya")
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result['title']} (sim: {result['similarity']:.3f})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure PostgreSQL with pgvector is running")
        print("   2. Check your .env file has correct credentials:")
        print("      PGVECTOR_HOST=localhost")
        print("      PGVECTOR_PORT=5432")
        print("      PGVECTOR_DATABASE=knowledge_base")
        print("      PGVECTOR_USER=postgres")
        print("      PGVECTOR_PASSWORD=your_password")
        print("\n   Quick setup with Docker:")
        print("   docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16")
        print("   docker exec -it pgvector psql -U postgres -c 'CREATE DATABASE knowledge_base;'")
        raise
    finally:
        await store.close()
    
    print("\n" + "=" * 70)
    print("✅ Knowledge base test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_pgvector_ingest())
