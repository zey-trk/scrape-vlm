"""
RAG Chunker - Creates optimized chunks for RAG using multiple LangChain strategies
"""
import json
from typing import List, Dict
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)


class RAGChunker:
    """Handles text chunking with multiple LangChain strategies for RAG optimization"""
    
    def __init__(self, input_file: str = "output.json"):
        """
        Initialize the RAG chunker
        
        Args:
            input_file: Path to the crawled data JSON file
        """
        self.input_file = input_file
        self.scraped_data = []
        self.all_chunks = []
        
    def load_data(self):
        """Load scraped data from JSON file"""
        print(f"Loading data from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.scraped_data = json.load(f)
        print(f"Loaded {len(self.scraped_data)} pages")
        
    def create_chunks_recursive_1000(self) -> List[Dict]:
        """
        Create chunks using RecursiveCharacterTextSplitter 
        with chunk_size=1000, overlap=200 (RECOMMENDED FOR RAG)
        """
        print("\n=== Strategy 1: RecursiveCharacterTextSplitter (1000/200) ===")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = []
        for page_idx, page in enumerate(self.scraped_data):
            content = page.get('content', '')
            if not content:
                continue
                
            page_chunks = splitter.split_text(content)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunks.append({
                    'chunk_id': f"recursive_1000_{page_idx}_{chunk_idx}",
                    'strategy': 'RecursiveCharacterTextSplitter',
                    'chunk_size_param': 1000,
                    'overlap_param': 200,
                    'source_url': page.get('url', ''),
                    'page_title': page.get('metadata', {}).get('title', 'No Title'),
                    'chunk_text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'position_in_doc': chunk_idx,
                })
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_chunks_recursive_500(self) -> List[Dict]:
        """
        Create chunks using RecursiveCharacterTextSplitter 
        with chunk_size=500, overlap=100 (SMALLER CHUNKS)
        """
        print("\n=== Strategy 2: RecursiveCharacterTextSplitter (500/100) ===")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = []
        for page_idx, page in enumerate(self.scraped_data):
            content = page.get('content', '')
            if not content:
                continue
                
            page_chunks = splitter.split_text(content)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunks.append({
                    'chunk_id': f"recursive_500_{page_idx}_{chunk_idx}",
                    'strategy': 'RecursiveCharacterTextSplitter',
                    'chunk_size_param': 500,
                    'overlap_param': 100,
                    'source_url': page.get('url', ''),
                    'page_title': page.get('metadata', {}).get('title', 'No Title'),
                    'chunk_text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'position_in_doc': chunk_idx,
                })
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_chunks_character(self) -> List[Dict]:
        """
        Create chunks using CharacterTextSplitter 
        with chunk_size=1000, overlap=200
        """
        print("\n=== Strategy 3: CharacterTextSplitter (1000/200) ===")
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = []
        for page_idx, page in enumerate(self.scraped_data):
            content = page.get('content', '')
            if not content:
                continue
                
            page_chunks = splitter.split_text(content)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunks.append({
                    'chunk_id': f"character_{page_idx}_{chunk_idx}",
                    'strategy': 'CharacterTextSplitter',
                    'chunk_size_param': 1000,
                    'overlap_param': 200,
                    'source_url': page.get('url', ''),
                    'page_title': page.get('metadata', {}).get('title', 'No Title'),
                    'chunk_text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'position_in_doc': chunk_idx,
                })
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_chunks_markdown(self) -> List[Dict]:
        """
        Create chunks using MarkdownTextSplitter 
        with chunk_size=1000, overlap=200 (OPTIMIZED FOR MARKDOWN)
        """
        print("\n=== Strategy 4: MarkdownTextSplitter (1000/200) ===")
        splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        
        chunks = []
        for page_idx, page in enumerate(self.scraped_data):
            content = page.get('content', '')
            if not content:
                continue
                
            page_chunks = splitter.split_text(content)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunks.append({
                    'chunk_id': f"markdown_{page_idx}_{chunk_idx}",
                    'strategy': 'MarkdownTextSplitter',
                    'chunk_size_param': 1000,
                    'overlap_param': 200,
                    'source_url': page.get('url', ''),
                    'page_title': page.get('metadata', {}).get('title', 'No Title'),
                    'chunk_text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'position_in_doc': chunk_idx,
                })
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_all_strategies(self) -> List[Dict]:
        """
        Process all chunking strategies and combine results
        
        Returns:
            List of all chunks from all strategies
        """
        self.load_data()
        
        # Apply all strategies
        chunks_recursive_1000 = self.create_chunks_recursive_1000()
        chunks_recursive_500 = self.create_chunks_recursive_500()
        chunks_character = self.create_chunks_character()
        chunks_markdown = self.create_chunks_markdown()
        
        # Combine all chunks
        self.all_chunks = (
            chunks_recursive_1000 + 
            chunks_recursive_500 + 
            chunks_character + 
            chunks_markdown
        )
        
        print(f"\n=== TOTAL: {len(self.all_chunks)} chunks created ===")
        
        return self.all_chunks
    
    def get_strategy_stats(self) -> Dict:
        """
        Calculate statistics for each chunking strategy
        
        Returns:
            Dictionary with stats per strategy
        """
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


if __name__ == "__main__":
    chunker = RAGChunker("output.json")
    chunks = chunker.process_all_strategies()
    stats = chunker.get_strategy_stats()
    
    print("\n=== STRATEGY STATISTICS ===")
    for strategy, stat in stats.items():
        print(f"\n{strategy}:")
        print(f"  Total chunks: {stat['total_chunks']}")
        print(f"  Avg chars: {stat['avg_chars']:.1f}")
        print(f"  Avg words: {stat['avg_words']:.1f}")
