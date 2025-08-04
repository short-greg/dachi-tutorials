"""
Multi-Document Analysis Processor

This module demonstrates Dachi framework features:
- async_process_map for parallel processing
- signaturemethod for LLM integration  
- Chunk for data segmentation
- Streaming responses
"""

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

# Import Dachi framework components
import dachi
from dachi.proc import async_process_map, Chunk, StreamProcess
from dachi.adapt.xopenai import OpenAIEngine
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
        self.engine = OpenAIEngine(api_key=api_key, model="gpt-4")
    
    def chunk_documents(self, documents: Dict[str, str], chunk_size: int = 1000) -> List[DocumentChunk]:
        """
        Chunk documents using Dachi's Chunk processor.
        
        Args:
            documents: Dict of {filename: content}
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        for doc_name, content in documents.items():
            # Use Dachi's Chunk processor
            chunker = Chunk(chunk_size=chunk_size, overlap=100)
            text_chunks = chunker.forward(content)
            
            for i, chunk_text in enumerate(text_chunks):
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    document_name=doc_name,
                    chunk_id=i
                ))
        
        return chunks
    
    @dachi.signaturemethod
    def analyze_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Analyze a document chunk using signaturemethod.
        
        This method will be converted to an LLM call using the template below.
        The return value will contain all variables referenced in {}.
        
        Args:
            chunk: DocumentChunk to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Template for LLM analysis
        """
        Analyze this document chunk and provide structured insights:
        
        Document: {chunk.document_name}
        Chunk {chunk.chunk_id}:
        {chunk.text}
        
        Provide analysis in the following format:
        - summary: Brief summary of this chunk
        - topics: List of 3-5 main topics covered
        - sentiment: Overall sentiment (positive/negative/neutral)
        - key_insights: 2-3 important insights or findings
        
        Return as JSON with keys: summary, topics, sentiment, key_insights
        """
        
        # This will be replaced by LLM call
        # Return structure matches template variables
        return {
            'summary': f"Analysis of {chunk.document_name} chunk {chunk.chunk_id}",
            'topics': ["topic1", "topic2", "topic3"],
            'sentiment': "neutral",
            'key_insights': ["insight1", "insight2"]
        }
    
    async def process_documents_parallel(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Process document chunks in parallel using async_process_map.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of analysis results
        """
        # Use Dachi's async_process_map for parallel processing
        results = await async_process_map(
            self.analyze_chunk,
            chunks,
            max_workers=5  # Process up to 5 chunks simultaneously
        )
        
        return results
    
    @dachi.signaturemethod  
    def synthesize_results(self, chunk_results: List[Dict[str, Any]], document_names: List[str]) -> Dict[str, Any]:
        """
        Synthesize results from all chunks using signaturemethod.
        
        This creates a comprehensive analysis from individual chunk results.
        
        Args:
            chunk_results: Results from individual chunk analysis
            document_names: Names of processed documents
            
        Returns:
            Synthesized analysis results
        """
        # Template for synthesis
        """
        Synthesize the following document analysis results into a comprehensive overview:

        Documents analyzed: {document_names}
        
        Individual chunk results:
        {chunk_results}
        
        Provide a synthesized analysis with:
        - overall_summary: Comprehensive summary across all documents
        - common_themes: Themes that appear across multiple documents  
        - document_comparison: How documents relate to each other
        - key_findings: Most important insights from the entire corpus
        - recommendations: Actionable recommendations based on the analysis
        
        Return as JSON with keys: overall_summary, common_themes, document_comparison, key_findings, recommendations
        """
        
        return {
            'overall_summary': "Synthesized summary of all documents",
            'common_themes': ["theme1", "theme2", "theme3"],
            'document_comparison': "Comparison between documents",
            'key_findings': ["finding1", "finding2", "finding3"],
            'recommendations': ["rec1", "rec2", "rec3"]
        }
    
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
        final_results = self.synthesize_results(chunk_results, document_names)
        
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