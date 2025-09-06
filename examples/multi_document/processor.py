"""
Multi-Document Analysis Processor

This module demonstrates Dachi framework features:
- async_process_map for parallel processing
- signaturemethod for LLM integration  
- Chunk for data segmentation
- Streaming responses
"""

import asyncio
from typing import List, Dict, Any, AsyncGenerator
from dataclasses import dataclass

# Import Dachi framework components
import dachi
from dachi.proc import async_process_map, Chunk, StreamProcess, Sequential
from dachi.proc._inst import signaturemethod
from dachi.proc._msg import ToText
from dachi.adapt.xopenai import ChatCompletion, TextConv
from dachi.core import Msg


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata."""
    text: str
    document_name: str
    chunk_id: int
    

@dataclass
class AnalysisResult:
    """Results from document analysis."""
    summary: str
    topics: List[str]
    sentiment: str
    key_insights: List[str]


class DocumentAnalyzer:
    """
    Document analyzer using Dachi framework features.
    Demonstrates signaturemethod, async_process_map, and streaming.
    """
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key."""
        # Create the pipeline: ToMsg => ChatCompletion with TextConv processor
        chat_completion = ChatCompletion(
            api_key=api_key, proc=TextConv(
                from_="response"
            ), 
            kwargs={"temperature": 0.5, "model": "gpt-4o", "max_tokens": 2000, "response_format": {"type": "json_object"}}
        )
        print(f'ChatCompletion from_: {chat_completion.proc[0].from_}')
        self.engine = Sequential(
            items=[
                ToText(role="user"),
                chat_completion
            ]
        )
    
    def chunk_documents(self, documents: Dict[str, str], chunk_size: int = 1000) -> List[DocumentChunk]:
        """
        Chunk documents using custom text chunking with overlap.
        
        Args:
            documents: Dict of {filename: content}
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        for doc_name, content in documents.items():
            # Custom text chunking with overlap
            text_chunks = self._chunk_text(content, chunk_size=chunk_size, overlap=100)
            
            for i, chunk_text in enumerate(text_chunks):
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    document_name=doc_name,
                    chunk_id=i
                ))
        
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Chunk text with overlap for better context preservation.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap
            if end >= len(text):
                break
            start = end - overlap
        
        return chunks
    
    @signaturemethod(engine='engine')
    def analyze_chunk(self, chunk: DocumentChunk) -> str:
        """Analyze this document chunk and provide structured insights:

Document: {chunk.document_name}
Chunk {chunk.chunk_id}:
{chunk.text}

Provide analysis in the following format:
- summary: Brief summary of this chunk
- topics: List of 3-5 main topics covered  
- sentiment: Overall sentiment (positive/negative/neutral)
- key_insights: 2-3 important insights or findings

Return as JSON with keys: summary, topics, sentiment, key_insights"""
        
        return {}
    
    async def process_documents_parallel(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Process document chunks in parallel using async_process_map.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of analysis results
        """
        try:
            # Test with first chunk only to debug
            if chunks:
                print(f"Testing single chunk analysis...")
                test_result = self.analyze_chunk(chunks[0])
                print(f"Single chunk result: {test_result}")
            
            # Process chunks using Chunk for parallelization
            raw_results = await async_process_map(
                self.analyze_chunk,
                Chunk(data=chunks)
            )
            
            # Convert string results to dictionaries
            results = []
            for raw_result in raw_results:
                try:
                    import json
                    parsed_result = json.loads(raw_result)
                    results.append(parsed_result)
                except:
                    # Fallback if JSON parsing fails
                    results.append({
                        'summary': raw_result[:200] + "..." if len(raw_result) > 200 else raw_result,
                        'topics': ["analysis"],
                        'sentiment': "neutral", 
                        'key_insights': ["analysis provided"]
                    })
            
            return results
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @signaturemethod(engine='engine')
    def synthesize_results(self, chunk_results: List[Dict[str, Any]], document_names: List[str]) -> str:
        """Synthesize the following document analysis results into a comprehensive overview:

Documents analyzed: {document_names}

Individual chunk results:
{chunk_results}

Provide a synthesized analysis with:
- overall_summary: Comprehensive summary across all documents
- common_themes: Themes that appear across multiple documents  
- document_comparison: How documents relate to each other
- key_findings: Most important insights from the entire corpus
- recommendations: Actionable recommendations based on the analysis

Return as JSON with keys: overall_summary, common_themes, document_comparison, key_findings, recommendations"""
        
        return {}
    
    async def analyze_documents_stream(self, documents: Dict[str, str]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main processing pipeline with streaming results.
        
        This demonstrates the complete Dachi workflow:
        1. Chunk documents
        2. Process chunks in parallel  
        3. Stream intermediate results
        4. Synthesize final results
        
        Args:
            documents: Dict of {filename: content}
            
        Yields:
            Progress updates and results as they become available
        """
        
        # Step 1: Chunk documents
        yield {"step": "chunking", "status": "started", "message": "Chunking documents..."}
        
        chunks = self.chunk_documents(documents)
        
        yield {
            "step": "chunking", 
            "status": "completed", 
            "message": f"Created {len(chunks)} chunks from {len(documents)} documents",
            "chunks_count": len(chunks)
        }
        
        # Step 2: Process chunks in parallel
        yield {"step": "analysis", "status": "started", "message": "Analyzing chunks in parallel..."}
        
        chunk_results = await self.process_documents_parallel(chunks)
        
        yield {
            "step": "analysis",
            "status": "completed", 
            "message": f"Analyzed {len(chunk_results)} chunks",
            "chunk_results": chunk_results
        }
        
        # Step 3: Synthesize results
        yield {"step": "synthesis", "status": "started", "message": "Synthesizing final results..."}
        
        document_names = list(documents.keys())
        raw_final_results = self.synthesize_results(chunk_results, document_names)
        
        # Convert string result to dictionary
        try:
            import json
            print(f"Raw synthesis result: {raw_final_results}")
            final_results = json.loads(raw_final_results)
            print(f"Parsed JSON: {final_results}")
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw result was: {raw_final_results}")
            # Fallback if JSON parsing fails
            final_results = {
                'overall_summary': raw_final_results[:500] + "..." if len(raw_final_results) > 500 else raw_final_results,
                'common_themes': ["analysis provided"],
                'document_comparison': "Analysis completed",
                'key_findings': ["synthesis provided"],
                'recommendations': ["review results"]
            }
        
        yield {
            "step": "synthesis",
            "status": "completed",
            "message": "Analysis complete!",
            "final_results": final_results
        }


# Convenience function for easy use
async def analyze_documents(documents: Dict[str, str], api_key: str):
    """
    Convenience function to analyze documents with streaming results.
    
    Args:
        documents: Dict of {filename: content}
        api_key: OpenAI API key
        
    Returns:
        Async generator of results
    """
    analyzer = DocumentAnalyzer(api_key)
    async for result in analyzer.analyze_documents_stream(documents):
        yield result