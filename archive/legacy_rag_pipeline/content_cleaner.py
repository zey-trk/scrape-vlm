"""
Content Cleaner - Prepares raw web content for optimal RAG embedding

This module handles:
1. URL removal (corrupts embeddings)
2. Markdown artifact cleanup
3. Navigation/boilerplate removal
4. Incomplete text detection and filtering
5. Sentence boundary preservation
"""
import re
from typing import List, Tuple


class ContentCleaner:
    """Cleans web content for optimal RAG embedding quality"""
    
    # Common boilerplate patterns in Turkish bank websites
    BOILERPLATE_PATTERNS = [
        r'\[ Ücretsiz İndir \]\([^)]+\)',
        r'\[ \]\(javascript:[^)]*\)',
        r'Önerilen Aramalar',
        r'Hadi fırsatları nelerdir\?',
        r'Hadi\'de nasıl hesap oluşturulur\?',
        r'Ücretler ve limitler için tıklayınız\.',
        r'\* \* \*',  # Section dividers
        r'\[x\]|\[ \]',  # Checkbox markers
    ]
    
    # Patterns that indicate incomplete/truncated content
    INCOMPLETE_PATTERNS = [
        r'\.\.\.$',  # Ends with ...
        r'…$',  # Ends with ellipsis character
        r'\.\.\.\s*$',  # Ends with ... and whitespace
    ]
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'https?://[^\s\)\]]+|'  # Full URLs
            r'www\.[^\s\)\]]+',  # www URLs
            re.IGNORECASE
        )
        
        self.markdown_image_pattern = re.compile(
            r'!\[[^\]]*\]\([^)]*\)',  # ![alt](url) format
            re.IGNORECASE
        )
        
        self.markdown_link_pattern = re.compile(
            r'\[([^\]]*)\]\([^)]*\)',  # [text](url) - keep text, remove link
            re.IGNORECASE
        )
        
        self.empty_brackets_pattern = re.compile(
            r'\[\s*\]|\(\s*\)',  # Empty [] or ()
        )
        
        self.hash_header_pattern = re.compile(
            r'^#{1,6}\s+',  # Markdown headers
            re.MULTILINE
        )
        
        self.multiple_newlines_pattern = re.compile(r'\n{3,}')
        self.multiple_spaces_pattern = re.compile(r'[ \t]{2,}')
        
        # Pattern for truncated preview lines (content ending with ...)
        self.truncated_line_pattern = re.compile(
            r'^[^\n]*\.\.\.\s*$',  # Lines ending with ...
            re.MULTILINE
        )
    
    def remove_truncated_sections(self, text: str) -> str:
        """
        Remove lines that end with ... (truncated preview text)
        These are incomplete and will hurt RAG quality
        """
        # Split into lines and filter
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            stripped = line.rstrip()
            # Skip lines that end with truncation markers
            if stripped.endswith('...') or stripped.endswith('…'):
                # Keep if it's a very long line (might be intentional ellipsis)
                if len(stripped) < 200:
                    continue  # Skip truncated preview text
            clean_lines.append(line)
        
        return '\n'.join(clean_lines)
        
    def remove_urls(self, text: str) -> str:
        """Remove all URLs from text"""
        return self.url_pattern.sub('', text)
    
    def remove_markdown_images(self, text: str) -> str:
        """Remove markdown image references"""
        return self.markdown_image_pattern.sub('', text)
    
    def extract_link_text(self, text: str) -> str:
        """Convert markdown links to plain text (keep link text, remove URL)"""
        return self.markdown_link_pattern.sub(r'\1', text)
    
    def remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate/navigation content"""
        for pattern in self.BOILERPLATE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    def clean_markdown_artifacts(self, text: str) -> str:
        """Clean up markdown formatting artifacts"""
        # Remove empty brackets
        text = self.empty_brackets_pattern.sub('', text)
        
        # Convert headers to plain text with emphasis
        text = self.hash_header_pattern.sub('', text)
        
        # Remove bold/italic markers but keep text
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for clean output"""
        # Replace multiple newlines with double newline
        text = self.multiple_newlines_pattern.sub('\n\n', text)
        
        # Replace multiple spaces with single space
        text = self.multiple_spaces_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def is_incomplete_text(self, text: str) -> bool:
        """Check if text appears to be truncated/incomplete"""
        text = text.strip()
        
        for pattern in self.INCOMPLETE_PATTERNS:
            if re.search(pattern, text):
                return True
        
        return False
    
    def expand_incomplete_text(self, text: str, full_content: str) -> str:
        """
        Try to expand incomplete text by finding the full sentence in original content
        """
        if not self.is_incomplete_text(text):
            return text
        
        # Remove the ellipsis to search for the text
        search_text = re.sub(r'\.{3}|…', '', text).strip()
        
        if not search_text or len(search_text) < 20:
            return text
        
        # Find the position in full content
        pos = full_content.find(search_text)
        if pos == -1:
            return text
        
        # Find the end of the sentence (., !, ?, or newline)
        end_pos = pos + len(search_text)
        sentence_end = None
        
        for i in range(end_pos, min(end_pos + 200, len(full_content))):
            if full_content[i] in '.!?\n':
                sentence_end = i + 1
                break
        
        if sentence_end:
            return full_content[pos:sentence_end].strip()
        
        return text
    
    def clean_content(self, text: str) -> str:
        """
        Apply full cleaning pipeline to content
        
        Order matters! We process in this sequence:
        1. Remove images first (most disruptive markdown)
        2. Extract link text (keep meaningful text)
        3. Remove remaining URLs
        4. Remove boilerplate
        5. Clean markdown artifacts
        6. Remove truncated preview sections
        7. Normalize whitespace
        """
        if not text:
            return ''
        
        # Step 1: Remove markdown images
        text = self.remove_markdown_images(text)
        
        # Step 2: Extract text from links
        text = self.extract_link_text(text)
        
        # Step 3: Remove any remaining URLs
        text = self.remove_urls(text)
        
        # Step 4: Remove boilerplate navigation
        text = self.remove_boilerplate(text)
        
        # Step 5: Clean markdown artifacts
        text = self.clean_markdown_artifacts(text)
        
        # Step 6: Remove truncated preview sections (lines ending with ...)
        text = self.remove_truncated_sections(text)
        
        # Step 7: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        return text
    
    def get_clean_stats(self, original: str, cleaned: str) -> dict:
        """Get statistics about the cleaning process"""
        return {
            'original_chars': len(original),
            'cleaned_chars': len(cleaned),
            'reduction_percent': round((1 - len(cleaned) / max(len(original), 1)) * 100, 1),
            'original_words': len(original.split()),
            'cleaned_words': len(cleaned.split()),
        }


def demonstrate_cleaning():
    """Demonstrate the cleaning on sample problematic content"""
    
    sample_text = """
    [ Ücretsiz İndir ](https://haditombank.com/hadi-indir)
    [ ](javascript:;)
    Önerilen Aramalar
    * [Hadi fırsatları nelerdir?](https://haditombank.com/hadi-kazan/kampanyalar)
    
    #### Hadi Kredi Kartı
    **Hadi Taksitli Alışveriş limitinle alışverişlerin...
    
    [](https://haditombank.com/hadi-kartlarim/hadi-black-kredi-karti) ![](https://haditombank.com/medium/Page/Image/64d14980-bb0e-4407-b1ff-618dae2accdc)
    
    Hadi Kredi Kartında ömür boyu kart ücreti yok, harcadıkça kazandıran kampanyalar var!
    """
    
    cleaner = ContentCleaner()
    cleaned = cleaner.clean_content(sample_text)
    stats = cleaner.get_clean_stats(sample_text, cleaned)
    
    print("=== BEFORE CLEANING ===")
    print(sample_text)
    print("\n=== AFTER CLEANING ===")
    print(cleaned)
    print("\n=== STATISTICS ===")
    print(f"Reduction: {stats['reduction_percent']}%")
    print(f"Original: {stats['original_chars']} chars -> Cleaned: {stats['cleaned_chars']} chars")


if __name__ == "__main__":
    demonstrate_cleaning()
