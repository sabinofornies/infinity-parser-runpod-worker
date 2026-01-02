# Infinity-Parser RunPod Worker

GPU-accelerated document parsing using [Infinity-Parser-7B](https://huggingface.co/infly/Infinity-Parser-7B).

## What is Infinity-Parser?

A 7B vision-language model specifically trained for scanned document parsing:
- **Languages**: English + Chinese
- **Strengths**: Tables, formulas, reading order preservation
- **Output**: Clean Markdown with HTML tables and LaTeX math

## Architecture

```
┌─────────────────┐      ┌─────────────────────────────────┐
│   Your App      │      │        RunPod GPU               │
│   (Flask)       │ ───► │   Infinity-Parser-7B            │
│                 │ ◄─── │   ~16GB VRAM (A10/RTX 4090)     │
└─────────────────┘      └─────────────────────────────────┘
```

## Deployment

### 1. Create GitHub Repository

```bash
cd runpod_infinity
git init
git add .
git commit -m "Initial commit: Infinity-Parser worker"
gh repo create infinity-parser-runpod-worker --private --source=. --push
```

### 2. Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **GitHub Repo** as source
4. Connect your GitHub account
5. Select `infinity-parser-runpod-worker` repo
6. Configure:
   - **GPU**: A10 (24GB) or RTX 4090 (24GB) recommended
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 3
7. Click **Deploy**

### 3. Configure Environment

Add to your `~/.zshrc`:

```bash
export RUNPOD_INFINITY_ENDPOINT_ID="your_endpoint_id"
```

## API Usage

### Request

```json
{
  "input": {
    "file": "<base64_encoded_pdf_or_image>",
    "file_name": "document.pdf"
  }
}
```

### Response

```json
{
  "success": true,
  "markdown": "# Document Title\n\nContent...",
  "page_count": 5,
  "file_name": "document.pdf"
}
```

## Cost Estimate

| GPU | $/hour | Est. $/page |
|-----|--------|-------------|
| A10 (24GB) | $0.28 | ~$0.003 |
| RTX 4090 (24GB) | $0.44 | ~$0.004 |

## Comparison with Marker

| Feature | Infinity-Parser | Marker |
|---------|-----------------|--------|
| Model size | 7B | Multiple models |
| Chinese | Excellent | Good |
| Tables | HTML output | Markdown |
| Formulas | LaTeX | LaTeX |
| Reading order | RL-trained | Heuristic |
| Best for | Scanned docs, Chinese | Complex PDFs |
