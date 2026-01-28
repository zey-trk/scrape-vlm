import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
from io import BytesIO
from pypdf import PdfReader

def estimate_cost():
    print("Analyze PDF Cost for VLM...")
    
    # Cost Constants (GPT-4o)
    INPUT_COST_PER_1M = 2.50
    OUTPUT_COST_PER_1M = 10.00
    
    # Token Estimates per Page
    IMG_TOKENS = 765      # Standard high-res image cost
    TEXT_TOKENS = 200     # System prompt + user prompt
    OUTPUT_TOKENS = 800   # Markdown table output (conservative estimate)
    
    TOTAL_INPUT_PER_PAGE = IMG_TOKENS + TEXT_TOKENS
    
    try:
        data = json.load(open('rag_results/output.json'))
        
        # Filter for complex PDFs
        complex_pdfs = []
        for page in data:
            url = page.get('url', '')
            if url.endswith('.pdf'):
                # Same logic as process_pdfs.py
                is_complex = 'ucret' in url.lower() or 'tarife' in url.lower() or 'tablo' in url.lower()
                if is_complex:
                    complex_pdfs.append(url)
        
        print(f"Found {len(complex_pdfs)} complex PDFs matching VLM criteria.")
        
        if not complex_pdfs:
            print("No complex PDFs found. Cost: $0.00")
            return

        total_pages = 0
        print("\nCounting pages...")
        
        for i, url in enumerate(complex_pdfs):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    reader = PdfReader(BytesIO(response.content))
                    pages = len(reader.pages)
                    total_pages += pages
                    print(f"  {i+1}. {url.split('/')[-1]}: {pages} pages")
            except Exception as e:
                print(f"  {i+1}. {url}: Error counting pages ({e})")
        
        print(f"\nTotal Pages in Complex Docs: {total_pages}")
        
        # Calculate Costs
        total_input_tokens = total_pages * TOTAL_INPUT_PER_PAGE
        total_output_tokens = total_pages * OUTPUT_TOKENS
        
        cost_input = (total_input_tokens / 1_000_000) * INPUT_COST_PER_1M
        cost_output = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
        total_cost = cost_input + cost_output
        
        print("\n" + "="*50)
        print(f"💰 ESTIMATED VLM COST")
        print("="*50)
        print(f"Total Pages:      {total_pages}")
        print(f"Input Tokens:     {total_input_tokens:,}")
        print(f"Output Tokens:    {total_output_tokens:,}")
        print("-" * 30)
        print(f"Input Cost:       ${cost_input:.4f}")
        print(f"Output Cost:      ${cost_output:.4f}")
        print("="*50)
        print(f"TOTAL ESTIMATE:   ${total_cost:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    estimate_cost()
