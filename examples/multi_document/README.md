# Multi-Document Analysis Pipeline Tutorial

This tutorial demonstrates key features of the Dachi framework through a practical multi-document analysis pipeline.

## 🎯 What You'll Learn

This tutorial showcases:

- **`async_process_map`**: Process document chunks in parallel for improved performance
- **`@signaturemethod`**: Create LLM-powered functions with templated signatures
- **`Chunk`**: Intelligently segment documents into processable pieces  
- **Streaming responses**: Real-time progress updates during processing
- **OpenAI integration**: Use Dachi's OpenAI adapter for LLM operations

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Dachi framework installed

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the Dachi framework is available in your Python path

### Running the Tutorial

From the root directory:
```bash
python run.py multi_document
```

This will launch a Streamlit web interface where you can:
1. Upload your own text documents or use the provided samples
2. Watch the parallel processing pipeline in action
3. View comprehensive analysis results
4. Export results in multiple formats

## 📚 Tutorial Features

### Document Processing Pipeline

The tutorial implements a complete document analysis workflow:

1. **Document Upload**: Upload multiple `.txt` files through the web interface
2. **Intelligent Chunking**: Use Dachi's `Chunk` processor to segment documents
3. **Parallel Analysis**: Process chunks simultaneously with `async_process_map`
4. **LLM Integration**: Analyze content using `@signaturemethod` decorators
5. **Results Synthesis**: Combine individual analyses into comprehensive insights
6. **Real-time Progress**: Stream updates as processing completes

### Key Components

- **`processor.py`**: Core Dachi framework integration
  - `DocumentAnalyzer` class with signature methods
  - Parallel processing using `async_process_map`
  - Document chunking with overlap handling
  - Streaming result generation

- **`components.py`**: Streamlit UI components
  - File upload interface
  - Progress tracking with real-time updates
  - Results visualization and export

- **`main.py`**: Main Streamlit application
  - Orchestrates the entire pipeline
  - Handles async processing in Streamlit
  - Provides interactive configuration options

## 🎓 Learning Objectives

### 1. Parallel Processing with `async_process_map`

See how Dachi enables efficient parallel processing:

```python
# Process multiple document chunks simultaneously
results = await async_process_map(
    self.analyze_chunk,
    chunks,
    max_workers=5
)
```

### 2. LLM Integration with `@signaturemethod`

Learn to create LLM-powered functions using templates:

```python
@dachi.signaturemethod
def analyze_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
    \"\"\"
    Analyze this document chunk:
    
    Document: {chunk.document_name}
    Text: {chunk.text}
    
    Return JSON with: summary, topics, sentiment, key_insights
    \"\"\"
    return {'summary': '...', 'topics': [...], ...}
```

### 3. Document Chunking

Understand intelligent text segmentation:

```python
chunker = Chunk(chunk_size=1000, overlap=100)
chunks = chunker.forward(content)
```

### 4. Streaming Results

Implement real-time progress updates:

```python
async def analyze_documents_stream(self, documents):
    yield {"step": "chunking", "status": "started"}
    # ... processing ...
    yield {"step": "chunking", "status": "completed", "results": ...}
```

## 📖 Sample Documents

The tutorial includes sample documents covering:
- AI in Healthcare
- Climate Change and Renewable Energy  
- Future of Remote Work
- Quarterly Technology Trends Report

These demonstrate the pipeline's ability to:
- Extract key themes across different domains
- Perform sentiment analysis
- Identify cross-document relationships
- Generate actionable insights

## 🛠️ Customization

### Adding New Analysis Types

Extend the `DocumentAnalyzer` class with additional signature methods:

```python
@dachi.signaturemethod
def extract_entities(self, chunk: DocumentChunk) -> Dict[str, Any]:
    \"\"\"
    Extract named entities from: {chunk.text}
    
    Return JSON with: people, organizations, locations, dates
    \"\"\"
    pass
```

### Modifying Chunk Processing

Adjust chunking parameters for different document types:

```python
# For code documents
chunker = Chunk(chunk_size=500, overlap=50)

# For academic papers  
chunker = Chunk(chunk_size=1500, overlap=200)
```

### Custom Progress Tracking

Add your own progress indicators:

```python
yield {
    "step": "custom_analysis",
    "status": "in_progress", 
    "progress": 0.75,
    "details": "Processing entity relationships..."
}
```

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Key**: Make sure your API key is valid and has sufficient credits
2. **Large Documents**: Increase chunk size for very large documents to avoid token limits
3. **Memory Usage**: Reduce `max_workers` if processing many large documents simultaneously

### Performance Tips

- Optimal chunk size is usually 800-1200 characters
- Use 3-7 parallel workers for best performance  
- Enable result caching for repeated analyses

## 📝 Next Steps

After completing this tutorial, you can:

1. **Extend the analysis**: Add more signature methods for specialized tasks
2. **Try other Dachi features**: Explore behavior trees, different adapters
3. **Build custom applications**: Use the patterns learned here in your own projects
4. **Contribute**: Share improvements or additional tutorials

## 🤝 Contributing

Found an issue or want to improve the tutorial? Please submit a pull request or open an issue in the main repository.

## 📄 License

This tutorial is part of the Dachi framework documentation and follows the same license terms.