"""
Multi-Document Analysis Pipeline Tutorial

This tutorial demonstrates key Dachi framework features:
- async_process_map for parallel processing
- signaturemethod for LLM integration with templated signatures
- Chunk for document segmentation
- Streaming responses for real-time updates

Run with: python run.py multi_document
"""

import streamlit as st
import asyncio
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Import tutorial components
from components import (
    file_upload_component,
    sample_documents_component, 
    progress_tracker_component,
    update_progress,
    results_display_component,
    export_results_component
)
from processor import analyze_documents


def main():
    """Main Streamlit application."""
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key in environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("🚨 OpenAI API key not found in environment variables.")
        st.error("Please contact your administrator to set up the OPENAI_API_KEY environment variable.")
        st.info("The system cannot run without a valid OpenAI API key.")
        st.stop()
    
    st.set_page_config(
        page_title="Dachi Multi-Document Analysis",
        layout="wide"
    )
    
    st.title("Multi-Document Analysis Pipeline")
    st.markdown("""
    **Dachi Framework Tutorial: Processing Documents in Parallel**
    
    This tutorial demonstrates:
    - **async_process_map**: Process document chunks in parallel
    - **signaturemethod**: LLM integration with templated signatures  
    - **Chunk**: Intelligent document segmentation
    - **Streaming**: Real-time progress updates
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Show API key status
        st.success("OpenAI API key loaded from environment")
        
        # Processing options
        st.subheader("Processing Options")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=500,
            max_value=2000,
            value=1000,
            help="Size of text chunks for parallel processing"
        )
        
        max_workers = st.slider(
            "Max Parallel Workers",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of chunks to process simultaneously"
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Upload Documents", "Process & Analyze", "View Results"])
    
    with tab1:
        # Document upload section
        uploaded_docs = file_upload_component()
        
        st.divider()
        
        # Sample documents section
        sample_docs = sample_documents_component()
        
        # Store documents in session state
        if uploaded_docs:
            st.session_state.documents = uploaded_docs
            st.session_state.doc_source = "uploaded"
        elif sample_docs:
            st.session_state.documents = sample_docs
            st.session_state.doc_source = "sample"
    
    with tab2:
        st.header("Start Analysis")
        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("Please upload documents or load sample documents first.")
            return
        
        documents = st.session_state.documents
        st.info(f"Ready to analyze {len(documents)} documents using Dachi framework features.")
        
        # Show document summary
        for filename, content in documents.items():
            word_count = len(content.split())
            st.write(f"**{filename}**: {word_count} words")
        
        # Analysis button
        if st.button("Start Analysis", type="primary", use_container_width=True):
            # Create progress tracking containers
            progress_containers = progress_tracker_component()
            
            # Run analysis with streaming updates
            run_analysis(documents, api_key, progress_containers, chunk_size, max_workers)
    
    with tab3:
        st.header("Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Display chunk results
            if 'chunk_results' in results:
                results_display_component(
                    chunk_results=results['chunk_results'],
                    final_results=results.get('final_results')
                )
            
            # Export functionality
            if 'final_results' in results:
                st.divider()
                export_results_component(results['final_results'])
        else:
            st.info("No analysis results yet. Please run the analysis first.")


def run_analysis(documents: Dict[str, str], api_key: str, progress_containers: Dict, 
                chunk_size: int, max_workers: int):
    """
    Run the document analysis with streaming updates.
    
    Args:
        documents: Dict of {filename: content}
        api_key: OpenAI API key
        progress_containers: Progress tracking containers
        chunk_size: Size for document chunks
        max_workers: Max parallel workers
    """
    
    async def process_with_progress():
        """Async function to process documents and update progress."""
        results = {'chunk_results': [], 'final_results': None}
        
        try:
            async for update in analyze_documents(documents, api_key):
                step = update.get('step')
                status = update.get('status')
                message = update.get('message', '')
                
                if step == 'chunking':
                    progress = 1.0 if status == 'completed' else 0.3
                    update_progress(progress_containers, 'chunking', status, message, progress)
                    
                    if status == 'completed':
                        chunks_count = update.get('chunks_count', 0)
                        st.info(f"Created {chunks_count} chunks for parallel processing")
                
                elif step == 'analysis':
                    progress = 1.0 if status == 'completed' else 0.5
                    update_progress(progress_containers, 'analysis', status, message, progress)
                    
                    if status == 'completed':
                        results['chunk_results'] = update.get('chunk_results', [])
                        st.success(f"Analyzed {len(results['chunk_results'])} chunks using async_process_map")
                
                elif step == 'synthesis':
                    progress = 1.0 if status == 'completed' else 0.7
                    update_progress(progress_containers, 'synthesis', status, message, progress)
                    
                    if status == 'completed':
                        results['final_results'] = update.get('final_results', {})
                        st.success("Synthesis complete using signaturemethod!")
                        
                        # Store results in session state
                        st.session_state.analysis_results = results
                        
                        # Show completion message
                        st.success("Analysis complete! Check the 'View Results' tab to see the findings.")
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.error("Make sure your OpenAI API key is valid and you have sufficient credits.")
    
    # Run the async analysis
    asyncio.run(process_with_progress())


if __name__ == "__main__":
    main()