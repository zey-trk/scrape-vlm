"""
Optimized RAG Chunker - Creates high-quality chunks for RAG with clean content

Key improvements over basic chunker:
1. Content cleaning before chunking (removes URLs, markdown artifacts)
2. Sentence-aware splitting (no mid-sentence cuts)
3. Section-based chunking (respects document structure)
4. Minimum chunk quality filtering
5. Semantic completeness validation
"""
import json
import re
from typing import List, Dict, Optional
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter
)
from .content_cleaner import ContentCleaner
from .config import CHUNK_SIZE_OPTIMIZED, CHUNK_OVERLAP_OPTIMIZED, CHUNK_SIZE_GRANULAR, CHUNK_OVERLAP_GRANULAR


class OptimizedRAGChunker:
    """Creates high-quality chunks optimized for RAG retrieval"""
    
    def __init__(self, input_file: str = "rag_results/output.json"):
        self.input_file = input_file
        self.scraped_data = []
        self.all_chunks = []
        self.cleaner = ContentCleaner()
        
        # Configuration for optimal RAG chunks
        self.config = {
            'primary_chunk_size': 800,      # Slightly smaller for better precision
            'primary_overlap': 150,          # Good overlap for context continuity
            'secondary_chunk_size': 400,     # Smaller chunks for granular retrieval
            'secondary_overlap': 80,
            'min_chunk_length': 50,          # Minimum chars for a valid chunk
            'min_words': 10,                 # Minimum words for a valid chunk
        }
        
    def load_data(self):
        """Load scraped data from JSON file"""
        print(f"Loading data from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.scraped_data = json.load(f)
        print(f"Loaded {len(self.scraped_data)} pages")
        
    def clean_content(self, content: str) -> str:
        """Apply comprehensive content cleaning"""
        return self.cleaner.clean_content(content)
    
    def is_valid_chunk(self, chunk: str) -> bool:
        """Check if chunk meets minimum quality requirements"""
        chunk = chunk.strip()
        
        if len(chunk) < self.config['min_chunk_length']:
            return False
            
        word_count = len(chunk.split())
        if word_count < self.config['min_words']:
            return False
        
        # Reject chunks that are mostly special characters or URLs
        alphanumeric_ratio = sum(1 for c in chunk if c.isalnum() or c.isspace()) / max(len(chunk), 1)
        if alphanumeric_ratio < 0.6:
            return False
        
        # Reject chunks that end with incomplete patterns
        if chunk.rstrip().endswith('...') or chunk.rstrip().endswith('…'):
            # Try to detect if it's actually incomplete vs intentional ellipsis
            if len(chunk) < 100:  # Short chunks with ellipsis are likely incomplete
                return False
        
        return True
    
    def ensure_complete_sentences(self, chunks: List[str], original_content: str) -> List[str]:
        """
        Post-process chunks to ensure sentence completeness
        Tries to extend chunks that end mid-sentence
        """
        completed_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            
            if not chunk:
                continue
            
            # Check if chunk ends with sentence-ending punctuation
            last_char = chunk.rstrip()[-1] if chunk.rstrip() else ''
            
            if last_char not in '.!?:—':
                # Try to find chunk in original and extend to sentence end
                pos = original_content.find(chunk[:50])  # Search using first 50 chars
                if pos != -1:
                    end_pos = pos + len(chunk)
                    # Look for sentence ending within next 150 chars
                    for i in range(end_pos, min(end_pos + 150, len(original_content))):
                        if original_content[i] in '.!?':
                            chunk = original_content[pos:i+1].strip()
                            break
            
            if self.is_valid_chunk(chunk):
                completed_chunks.append(chunk)
        
        return completed_chunks
    
    def extract_sections(self, content: str) -> List[Dict]:
        """
        Extract logical sections from content based on headers
        Returns list of sections with title and content
        """
        sections = []
        
        # Split by header patterns (####, ###, ##, #)
        header_pattern = r'(?:^|\n)(#{1,4}\s+[^\n]+)'
        parts = re.split(header_pattern, content)
        
        current_section = {'title': 'Introduction', 'content': ''}
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            if part.startswith('#'):
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                # Start new section
                current_section = {
                    'title': re.sub(r'^#+\s*', '', part).strip(),
                    'content': ''
                }
            else:
                current_section['content'] += ' ' + part
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        # If no sections found, treat entire content as one section
        if not sections:
            sections = [{'title': 'Main Content', 'content': content}]
        
        return sections
    
    def create_optimized_chunks(self) -> List[Dict]:
        """
        Create chunks using optimized strategy:
        1. Clean content first
        2. Use RecursiveCharacterTextSplitter with sentence separators
        3. Post-process for sentence completeness
        4. Filter low-quality chunks
        """
        print("\n=== Creating Optimized RAG Chunks ===")
        print(f"Config: chunk_size={self.config['primary_chunk_size']}, overlap={self.config['primary_overlap']}")
        
        # Use separators that respect sentence boundaries better
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['primary_chunk_size'],
            chunk_overlap=self.config['primary_overlap'],
            length_function=len,
            separators=[
                "\n\n",      # Paragraph breaks (highest priority)
                "\n",        # Line breaks
                ". ",        # Sentence ends
                "! ",        # Exclamation ends
                "? ",        # Question ends
                "; ",        # Semicolon breaks
                ", ",        # Clause breaks
                " ",         # Word breaks (last resort)
            ],
            is_separator_regex=False,
        )
        
        chunks = []
        stats = {
            'pages_processed': 0,
            'chunks_created': 0,
            'chunks_filtered': 0,
            'chars_original': 0,
            'chars_cleaned': 0,
        }
        
        for page_idx, page in enumerate(self.scraped_data):
            raw_content = page.get('content', '')
            if not raw_content:
                continue
            
            stats['pages_processed'] += 1
            stats['chars_original'] += len(raw_content)
            
            # Step 1: Clean content
            cleaned_content = self.clean_content(raw_content)
            stats['chars_cleaned'] += len(cleaned_content)
            
            if not cleaned_content or len(cleaned_content) < 50:
                continue
            
            # Step 2: Split into chunks
            raw_chunks = splitter.split_text(cleaned_content)
            
            # Step 3: Ensure sentence completeness
            completed_chunks = self.ensure_complete_sentences(raw_chunks, cleaned_content)
            
            # Step 4: Create chunk entries with metadata
            for chunk_idx, chunk_text in enumerate(completed_chunks):
                if not self.is_valid_chunk(chunk_text):
                    stats['chunks_filtered'] += 1
                    continue
                
                chunks.append({
                    'chunk_id': f"optimized_{page_idx}_{chunk_idx}",
                    'strategy': 'Optimized_RAG',
                    'chunk_size_param': CHUNK_SIZE_OPTIMIZED,
                    'overlap_param': CHUNK_OVERLAP_OPTIMIZED,
                    'source_url': page.get('url', ''),
                    'page_title': page.get('metadata', {}).get('title', 'No Title'),
                    'chunk_text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'position_in_doc': chunk_idx,
                    'is_cleaned': True,
                })
                stats['chunks_created'] += 1
        
        print(f"\n📊 Optimization Statistics:")
        print(f"  Pages processed: {stats['pages_processed']}")
        print(f"  Chunks created: {stats['chunks_created']}")
        print(f"  Chunks filtered (low quality): {stats['chunks_filtered']}")
        reduction = (1 - stats['chars_cleaned'] / max(stats['chars_original'], 1)) * 100
        print(f"  Content reduction: {reduction:.1f}% (noise removed)")
        
        return chunks
    
    def create_granular_chunks(self) -> List[Dict]:
        """
        Create smaller chunks for more precise retrieval
        Uses secondary configuration with smaller chunk size
        """
        print("\n=== Creating Granular Chunks (for precise retrieval) ===")
        print(f"Config: chunk_size={self.config['secondary_chunk_size']}, overlap={self.config['secondary_overlap']}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['secondary_chunk_size'],
            chunk_overlap=self.config['secondary_overlap'],
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            is_separator_regex=False,
        )
        
        chunks = []
        
        for page_idx, page in enumerate(self.scraped_data):
            raw_content = page.get('content', '')
            if not raw_content:
                continue
            
            cleaned_content = self.clean_content(raw_content)
            if not cleaned_content or len(cleaned_content) < 50:
                continue
            
            raw_chunks = splitter.split_text(cleaned_content)
            completed_chunks = self.ensure_complete_sentences(raw_chunks, cleaned_content)
            
            for chunk_idx, chunk_text in enumerate(completed_chunks):
                if not self.is_valid_chunk(chunk_text):
                    continue
                
                chunks.append({
                    'chunk_id': f"granular_{page_idx}_{chunk_idx}",
                    'strategy': 'Granular_Precise',
                    'chunk_size_param': CHUNK_SIZE_GRANULAR,
                    'overlap_param': CHUNK_OVERLAP_GRANULAR,
                    'source_url': page.get('url', ''),
                    'page_title': page.get('metadata', {}).get('title', 'No Title'),
                    'chunk_text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'position_in_doc': chunk_idx,
                    'is_cleaned': True,
                })
        
        print(f"Created {len(chunks)} granular chunks")
        return chunks
    
    def process_all_strategies(self) -> List[Dict]:
        """Process with all optimized strategies"""
        self.load_data()
        
        optimized_chunks = self.create_optimized_chunks()
        granular_chunks = self.create_granular_chunks()
        
        self.all_chunks = optimized_chunks + granular_chunks
        
        print(f"\n=== TOTAL: {len(self.all_chunks)} high-quality chunks ===")
        
        return self.all_chunks
    
    def get_strategy_stats(self) -> Dict:
        """Calculate statistics for each strategy"""
        stats = {}
        
        for chunk in self.all_chunks:
            strategy = chunk['strategy']
            params = f"{chunk['chunk_size_param']}/{chunk['overlap_param']}"
            key = f"{strategy} ({params})"
            
            if key not in stats:
                stats[key] = {
                    'total_chunks': 0,
                    'total_chars': 0,
                    'total_words': 0,
                    'avg_chars': 0,
                    'avg_words': 0,
                }
            
            stats[key]['total_chunks'] += 1
            stats[key]['total_chars'] += chunk['char_count']
            stats[key]['total_words'] += chunk['word_count']
        
        # Calculate averages
        for key in stats:
            if stats[key]['total_chunks'] > 0:
                stats[key]['avg_chars'] = stats[key]['total_chars'] / stats[key]['total_chunks']
                stats[key]['avg_words'] = stats[key]['total_words'] / stats[key]['total_chunks']
        
        return stats
    
    def show_sample_chunks(self, n: int = 3):
        """Display sample chunks for review"""
        print("\n=== SAMPLE OPTIMIZED CHUNKS ===")
        for i, chunk in enumerate(self.all_chunks[:n]):
            print(f"\n--- Chunk {i+1} ({chunk['strategy']}) ---")
            print(f"Source: {chunk['source_url']}")
            print(f"Length: {chunk['char_count']} chars, {chunk['word_count']} words")
            print(f"Text preview:\n{chunk['chunk_text'][:300]}...")


if __name__ == "__main__":
    chunker = OptimizedRAGChunker("output.json")
    chunks = chunker.process_all_strategies()
    stats = chunker.get_strategy_stats()
    
    print("\n=== STRATEGY STATISTICS ===")
    for strategy, stat in stats.items():
        print(f"\n{strategy}:")
        print(f"  Total chunks: {stat['total_chunks']}")
        print(f"  Avg chars: {stat['avg_chars']:.1f}")
        print(f"  Avg words: {stat['avg_words']:.1f}")
    
    chunker.show_sample_chunks(3)
