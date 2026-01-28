# VLM Image Chunk Extraction

Extract structured data from any webpage using Playwright screenshots + Qwen3 VL, then export as RAG-ready chunks.

## Features

- 📸 **Screenshot-based extraction** - Captures full webpage as image
- 🧠 **VLM-powered understanding** - Uses Qwen3 VL to extract structured data
- 📝 **Custom instructions** - Write extraction prompts in Turkish or English
- 📊 **RAG-ready output** - Exports chunks to JSON + XLSX

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run extraction
python run_image_chunk_extraction.py \
  --url "https://example.com/page" \
  --instruction "Extract all products with prices" \
  --output products
```

## Usage

### CLI
```bash
python run_image_chunk_extraction.py \
  --url "https://tombank.com.tr/veresiye.html" \
  --instruction "Veresiye ürünü detaylarını çıkar" \
  --output veresiye
```

### Python API
```python
from run_image_chunk_extraction import VLMPipeline

pipeline = VLMPipeline()
result = await pipeline.run(
    url="https://example.com",
    instruction="Your extraction instruction...",
    output_name="my_extraction"
)
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-u`, `--url` | URL to extract from (required) |
| `-i`, `--instruction` | Extraction instruction (required) |
| `-o`, `--output` | Output file base name |
| `-f`, `--format` | VLM format: json, table, markdown |

## Output

Each extraction creates:

```
scrape_output/
├── {name}.json          # Full extraction + chunks
├── {name}_chunks.xlsx   # RAG-ready spreadsheet
└── {name}.png           # Page screenshot
```

## Project Structure

```
├── run_image_chunk_extraction.py   # Main CLI script
├── utils/
│   ├── vlm_extractor.py            # VLM extraction module
│   └── config.py                   # Configuration
├── examples/                       # Example scripts
├── scrape_output/                  # Output directory
└── archive/                        # Legacy files
```

## Environment Variables

Create a `.env` file:
```env
ubicloud_api_key=YOUR_UBICLOUD_API_KEY
```

## Examples

See the `examples/` folder for usage examples.

## License

MIT
