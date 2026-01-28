"""
PGVector Knowledge Base Store
Stores VLM-extracted data in PostgreSQL with pgvector for semantic search.

Setup (Docker):
    docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16

Required env vars in .env:
    PGVECTOR_HOST=localhost
    PGVECTOR_PORT=5432
    PGVECTOR_DATABASE=knowledge_base
    PGVECTOR_USER=postgres
    PGVECTOR_PASSWORD=postgres
"""
import os
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Try to import required packages
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    print("⚠️ asyncpg not installed. Run: pip install asyncpg")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")


@dataclass
class Document:
    """A document to store in the knowledge base."""
    content: str
    title: str = ""
    category: str = ""
    source_url: str = ""
    metadata: dict = field(default_factory=dict)
    id: Optional[str] = None
    embedding: Optional[list] = None


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model name. Default is multilingual-e5-large
                        which works well for Turkish and has 1024 dimensions.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Run: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Loaded embedding model: {model_name} (dim={self.dimension})")
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # For E5 models, prefix with "passage: " for documents
        if "e5" in self.model_name.lower():
            text = f"passage: {text}"
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        # For E5 models, prefix with "query: " for queries
        if "e5" in self.model_name.lower():
            query = f"query: {query}"
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class DocumentTransformer:
    """Transforms VLM-extracted JSON into searchable documents."""
    
    @staticmethod
    def transform_campaign_extraction(json_path: str) -> list[Document]:
        """
        Transform campaign extraction JSON into documents.
        
        Creates one document per campaign with segment details,
        plus one document for segment criteria.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        source_url = data.get("url", "")
        structured = data.get("structured_data", {})
        
        documents = []
        
        # Process campaigns
        for campaign in structured.get("kampanyalar", []):
            campaign_name = campaign.get("kampanya_adi", "Kampanya")
            
            # Build comprehensive content for each campaign
            content_parts = [f"**{campaign_name}**\n"]
            
            for segment, details in [
                ("Elite", campaign.get("elite", {})),
                ("Elite Plus", campaign.get("elite_plus", {})),
                ("Prestige", campaign.get("prestige", {}))
            ]:
                if details:
                    content_parts.append(
                        f"- {segment} müşterileri için: "
                        f"İade oranı {details.get('iade_orani', '-')}, "
                        f"günlük limit {details.get('gunluk_limit', '-')}, "
                        f"aylık limit {details.get('aylik_limit', '-')}."
                    )
            
            doc = Document(
                content="\n".join(content_parts),
                title=campaign_name,
                category="kampanya",
                source_url=source_url,
                metadata={
                    "campaign_details": campaign,
                    "extraction_date": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        # Process segment criteria
        segment_criteria = structured.get("segment_kriterleri", {})
        if segment_criteria:
            content_parts = ["**Özel Bankacılık Segment Kriterleri**\n"]
            for segment, criteria in [
                ("Elite", segment_criteria.get("elite", "")),
                ("Elite Plus", segment_criteria.get("elite_plus", "")),
                ("Prestige", segment_criteria.get("prestige", ""))
            ]:
                if criteria:
                    content_parts.append(f"- {segment}: {criteria}")
            
            doc = Document(
                content="\n".join(content_parts),
                title="Segment Kriterleri",
                category="segment",
                source_url=source_url,
                metadata={
                    "segment_details": segment_criteria,
                    "extraction_date": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def transform_generic(json_path: str) -> list[Document]:
        """Transform any extraction JSON into a single document."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        content = data.get("raw_response", "")
        if not content:
            content = json.dumps(data.get("structured_data", {}), ensure_ascii=False, indent=2)
        
        return [Document(
            content=content,
            title="Extracted Content",
            category="extraction",
            source_url=data.get("url", ""),
            metadata={"extraction_date": datetime.now().isoformat()}
        )]


class PGVectorStore:
    """PostgreSQL + pgvector knowledge base store."""
    
    # SQL for creating the schema
    SCHEMA_SQL = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    CREATE TABLE IF NOT EXISTS kb_documents (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        source_url TEXT,
        category TEXT,
        title TEXT,
        content TEXT NOT NULL,
        metadata JSONB DEFAULT '{}',
        embedding VECTOR({dimension}),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Create index if not exists
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'kb_documents_embedding_idx') THEN
            CREATE INDEX kb_documents_embedding_idx ON kb_documents 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        END IF;
    END $$;
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        embedding_model: str = "intfloat/multilingual-e5-large"
    ):
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg is required. Run: pip install asyncpg")
        
        self.host = host or os.getenv("PGVECTOR_HOST", "localhost")
        self.port = port or int(os.getenv("PGVECTOR_PORT", "5432"))
        self.database = database or os.getenv("PGVECTOR_DATABASE", "knowledge_base")
        self.user = user or os.getenv("PGVECTOR_USER", "postgres")
        self.password = password or os.getenv("PGVECTOR_PASSWORD", "postgres")
        
        self.pool: Optional[asyncpg.Pool] = None
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.transformer = DocumentTransformer()
    
    async def connect(self):
        """Establish connection pool to PostgreSQL."""
        self.pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=2,
            max_size=10
        )
        print(f"✅ Connected to PostgreSQL: {self.host}:{self.port}/{self.database}")
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            print("✅ PostgreSQL connection closed")
    
    async def create_schema(self):
        """Create the database schema."""
        async with self.pool.acquire() as conn:
            schema_sql = self.SCHEMA_SQL.format(dimension=self.embedding_generator.dimension)
            await conn.execute(schema_sql)
            print(f"✅ Schema created (embedding dimension: {self.embedding_generator.dimension})")
    
    async def insert_document(self, doc: Document) -> str:
        """Insert a single document with embedding."""
        if doc.embedding is None:
            doc.embedding = self.embedding_generator.embed(doc.content)
        
        doc_id = doc.id or str(uuid.uuid4())
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO kb_documents (id, source_url, category, title, content, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                """,
                uuid.UUID(doc_id),
                doc.source_url,
                doc.category,
                doc.title,
                doc.content,
                json.dumps(doc.metadata),
                str(doc.embedding)  # pgvector accepts string format
            )
        
        return doc_id
    
    async def insert_documents(self, docs: list[Document]) -> list[str]:
        """Insert multiple documents with batch embedding."""
        # Generate embeddings in batch
        contents = [doc.content for doc in docs]
        embeddings = self.embedding_generator.embed_batch(contents)
        
        for doc, embedding in zip(docs, embeddings):
            doc.embedding = embedding
        
        # Insert documents
        ids = []
        for doc in docs:
            doc_id = await self.insert_document(doc)
            ids.append(doc_id)
        
        return ids
    
    async def search(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None
    ) -> list[dict]:
        """
        Semantic similarity search.
        
        Args:
            query: Search query (can be in Turkish)
            k: Number of results to return
            category: Optional category filter
            
        Returns:
            List of matching documents with scores
        """
        query_embedding = self.embedding_generator.embed_query(query)
        
        if category:
            sql = """
                SELECT id, source_url, category, title, content, metadata,
                       1 - (embedding <=> $1) as similarity
                FROM kb_documents
                WHERE category = $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """
            params = [str(query_embedding), category, k]
        else:
            sql = """
                SELECT id, source_url, category, title, content, metadata,
                       1 - (embedding <=> $1) as similarity
                FROM kb_documents
                ORDER BY embedding <=> $1
                LIMIT $2
            """
            params = [str(query_embedding), k]
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        results = []
        for row in rows:
            results.append({
                "id": str(row["id"]),
                "title": row["title"],
                "content": row["content"],
                "category": row["category"],
                "source_url": row["source_url"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "similarity": row["similarity"]
            })
        
        return results
    
    async def ingest_extraction(
        self,
        json_path: str,
        extraction_type: Literal["campaign", "generic"] = "campaign"
    ) -> list[str]:
        """
        Ingest VLM extraction JSON into the knowledge base.
        
        Args:
            json_path: Path to the extraction JSON file
            extraction_type: Type of extraction for proper transformation
            
        Returns:
            List of inserted document IDs
        """
        if extraction_type == "campaign":
            docs = self.transformer.transform_campaign_extraction(json_path)
        else:
            docs = self.transformer.transform_generic(json_path)
        
        print(f"📄 Transformed {len(docs)} documents from {json_path}")
        
        ids = await self.insert_documents(docs)
        print(f"✅ Inserted {len(ids)} documents into knowledge base")
        
        return ids
    
    async def get_document_count(self) -> int:
        """Get total number of documents in the knowledge base."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM kb_documents")
            return result
    
    async def delete_by_category(self, category: str) -> int:
        """Delete all documents in a category."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM kb_documents WHERE category = $1",
                category
            )
            count = int(result.split()[-1])
            print(f"🗑️ Deleted {count} documents from category: {category}")
            return count
    
    async def clear_all(self) -> int:
        """Delete all documents. Use with caution!"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM kb_documents")
            count = int(result.split()[-1])
            print(f"🗑️ Cleared all {count} documents from knowledge base")
            return count


# Convenience function for quick usage
async def quick_ingest(json_path: str, **pg_kwargs) -> list[str]:
    """Quick helper to ingest an extraction file."""
    store = PGVectorStore(**pg_kwargs)
    await store.connect()
    await store.create_schema()
    
    ids = await store.ingest_extraction(json_path)
    
    await store.close()
    return ids
