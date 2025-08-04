"""
UI Components for Multi-Document Analysis Tutorial

Simple Streamlit components for file upload, progress tracking,
and results display.
"""

import streamlit as st
from typing import Dict, List, Any
import json


def file_upload_component() -> Dict[str, str]:
    """
    File upload component for multiple text documents.
    
    Returns:
        Dict of {filename: content} for uploaded files
    """
    st.header("📁 Document Upload")
    st.write("Upload multiple text files to analyze with Dachi's processing pipeline.")
    
    uploaded_files = st.file_uploader(
        "Choose text files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload .txt files to demonstrate parallel document processing"
    )
    
    documents = {}
    
    if uploaded_files:
        st.write(f"**Uploaded {len(uploaded_files)} files:**")
        
        for uploaded_file in uploaded_files:
            # Read file content
            content = uploaded_file.read().decode('utf-8')
            documents[uploaded_file.name] = content
            
            # Show file info
            word_count = len(content.split())
            char_count = len(content)
            
            with st.expander(f"📄 {uploaded_file.name} ({word_count} words, {char_count} chars)"):
                st.text_area(
                    "Preview", 
                    content[:500] + "..." if len(content) > 500 else content,
                    height=100,
                    disabled=True
                )
    
    return documents


def sample_documents_component() -> Dict[str, str]:
    """
    Component to load sample documents for demo purposes.
    
    Returns:
        Dict of {filename: content} for sample documents
    """
    st.header("📚 Sample Documents")
    st.write("Or use our sample documents to see the pipeline in action.")
    
    if st.button("Load Sample Documents", type="primary"):
        import os
        from pathlib import Path
        
        sample_dir = Path(__file__).parent / "sample_documents"
        documents = {}
        
        for file_path in sample_dir.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents[file_path.name] = content
        
        if documents:
            st.success(f"Loaded {len(documents)} sample documents!")
            
            for filename, content in documents.items():
                word_count = len(content.split())
                with st.expander(f"📄 {filename} ({word_count} words)"):
                    st.text_area(
                        "Preview",
                        content[:300] + "..." if len(content) > 300 else content,
                        height=80,
                        disabled=True
                    )
        
        return documents
    
    return {}


def progress_tracker_component():
    """
    Component for tracking analysis progress with progress bars.
    
    Returns:
        Streamlit containers for progress updates
    """
    st.header("⚙️ Processing Pipeline")
    st.write("Watch Dachi's async_process_map and signaturemethod in action!")
    
    # Create containers for different steps
    chunking_container = st.container()
    analysis_container = st.container()
    synthesis_container = st.container()
    
    with chunking_container:
        st.subheader("1. Document Chunking")
        chunking_status = st.empty()
        chunking_progress = st.progress(0)
    
    with analysis_container:
        st.subheader("2. Parallel Analysis (async_process_map)")
        analysis_status = st.empty()
        analysis_progress = st.progress(0)
    
    with synthesis_container:
        st.subheader("3. Results Synthesis (signaturemethod)")
        synthesis_status = st.empty()
        synthesis_progress = st.progress(0)
    
    return {
        'chunking': {'status': chunking_status, 'progress': chunking_progress},
        'analysis': {'status': analysis_status, 'progress': analysis_progress},
        'synthesis': {'status': synthesis_status, 'progress': synthesis_progress}
    }


def update_progress(containers: Dict, step: str, status: str, message: str, progress: float = 0):
    """
    Update progress for a specific step.
    
    Args:
        containers: Progress containers from progress_tracker_component
        step: Step name (chunking, analysis, synthesis)
        status: Status (started, in_progress, completed)
        message: Status message
        progress: Progress value (0-1)
    """
    if step in containers:
        container = containers[step]
        
        # Update status message
        if status == "started":
            container['status'].info(f"🔄 {message}")
        elif status == "completed":
            container['status'].success(f"✅ {message}")
        else:
            container['status'].info(f"⏳ {message}")
        
        # Update progress bar
        container['progress'].progress(progress)


def results_display_component(chunk_results: List[Dict[str, Any]] = None, final_results: Dict[str, Any] = None):
    """
    Component for displaying analysis results.
    
    Args:
        chunk_results: Results from individual chunk analysis
        final_results: Synthesized final results
    """
    st.header("📊 Analysis Results")
    
    if chunk_results:
        st.subheader("Chunk Analysis Results")
        st.write(f"Analyzed {len(chunk_results)} document chunks in parallel:")
        
        # Group results by document
        doc_groups = {}
        for result in chunk_results:
            doc_name = result.get('document_name', 'Unknown')
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(result)
        
        # Display results by document
        for doc_name, doc_results in doc_groups.items():
            with st.expander(f"📄 {doc_name} ({len(doc_results)} chunks)"):
                for i, result in enumerate(doc_results):
                    st.write(f"**Chunk {i+1}:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Summary:**")
                        st.write(result.get('summary', 'N/A'))
                        
                        st.write("**Topics:**")
                        topics = result.get('topics', [])
                        for topic in topics:
                            st.write(f"• {topic}")
                    
                    with col2:
                        st.write("**Sentiment:**")
                        sentiment = result.get('sentiment', 'neutral')
                        if sentiment == 'positive':
                            st.success(f"😊 {sentiment.title()}")
                        elif sentiment == 'negative':
                            st.error(f"😟 {sentiment.title()}")
                        else:
                            st.info(f"😐 {sentiment.title()}")
                        
                        st.write("**Key Insights:**")
                        insights = result.get('key_insights', [])
                        for insight in insights:
                            st.write(f"• {insight}")
                    
                    if i < len(doc_results) - 1:
                        st.divider()
    
    if final_results:
        st.subheader("🎯 Final Synthesis")
        st.write("Comprehensive analysis across all documents:")
        
        # Overall Summary
        st.write("**📝 Overall Summary:**")
        st.info(final_results.get('overall_summary', 'No summary available'))
        
        # Layout in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Common Themes
            st.write("**🏷️ Common Themes:**")
            themes = final_results.get('common_themes', [])
            for theme in themes:
                st.write(f"• {theme}")
            
            # Key Findings
            st.write("**🔍 Key Findings:**")
            findings = final_results.get('key_findings', [])
            for finding in findings:
                st.write(f"• {finding}")
        
        with col2:
            # Document Comparison
            st.write("**📋 Document Comparison:**")
            st.write(final_results.get('document_comparison', 'No comparison available'))
            
            # Recommendations
            st.write("**💡 Recommendations:**")
            recommendations = final_results.get('recommendations', [])
            for rec in recommendations:
                st.write(f"• {rec}")


def export_results_component(final_results: Dict[str, Any]):
    """
    Component for exporting analysis results.
    
    Args:
        final_results: Final analysis results to export
    """
    if final_results:
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_data = json.dumps(final_results, indent=2)
            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name="document_analysis_results.json",
                mime="application/json"
            )
        
        with col2:
            # Text report export
            report = f"""
# Document Analysis Report

## Overall Summary
{final_results.get('overall_summary', 'N/A')}

## Common Themes
{chr(10).join(f"• {theme}" for theme in final_results.get('common_themes', []))}

## Key Findings
{chr(10).join(f"• {finding}" for finding in final_results.get('key_findings', []))}

## Document Comparison
{final_results.get('document_comparison', 'N/A')}

## Recommendations
{chr(10).join(f"• {rec}" for rec in final_results.get('recommendations', []))}
"""
            
            st.download_button(
                label="📝 Download as Report",
                data=report,
                file_name="document_analysis_report.txt",
                mime="text/plain"
            )