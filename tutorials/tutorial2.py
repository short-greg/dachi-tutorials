# %%
# Markdown
"""
# Tutorial 2 — From one worker to a parallel pipeline (async, streaming, and chunks)

**What you'll build:** A small "document insights" pipeline that starts with one async worker and ends as a parallel, chunked, optionally streaming pipeline. It runs offline by default. If readers want, they can plug in their own LLM caller in one place without changing the rest of the code.

**Why it matters:**

* You learn **when to use** sync, async, and streaming processes.
* You see how **chunking** and `async_process_map` boost throughput for I/O tasks.
* You practice **composing stages** with `Sequential` and **aggregating results** with `reduce`.
* You keep vendor code isolated, so APIs can change without rewrites.
* You glimpse why serialization is important (save/restore later), without getting lost in it.
"""

# %%
# Python
import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator

from dachi.core import BaseModule, Param, Attr
from dachi.proc import (
    Process, AsyncProcess, AsyncStreamProcess,
    async_process_map, 
    Chunk, Recur,
    reduce,
    Sequential
)

# Define all our classes at the top
class DocInsights(AsyncProcess):
    """Async worker that analyzes text (mocked analysis by default)"""
    
    # Pluggable caller for real LLM integration
    caller: Optional[Callable] = Attr(default=None)
    
    # Stats tracking
    total_processed: int = Attr(default=0)
    
    async def aforward(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """Analyze text and return insights dict"""
        # Handle both single and batch inputs
        texts = [text] if isinstance(text, str) else text
        
        results = []
        for t in texts:
            # Simulate async I/O work
            await asyncio.sleep(0.1)
            
            if self.caller:
                # Use real LLM if available
                response = await self.caller(t)
                # Parse response into our format
                result = {
                    "sentiment": "positive",  # Would parse from response
                    "topics": ["AI", "technology"],
                    "word_count": len(t.split())
                }
            else:
                # Mock analysis
                words = t.split()
                word_count = len(words)
                
                # Simple sentiment based on keywords
                positive_words = {"good", "great", "excellent", "amazing"}
                negative_words = {"bad", "poor", "terrible", "awful"}
                
                text_lower = t.lower()
                pos_score = sum(1 for w in positive_words if w in text_lower)
                neg_score = sum(1 for w in negative_words if w in text_lower)
                
                if pos_score > neg_score:
                    sentiment = "positive"
                elif neg_score > pos_score:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                # Mock topics
                topics = []
                if "ai" in text_lower or "artificial" in text_lower:
                    topics.append("AI")
                if "data" in text_lower:
                    topics.append("data")
                if "technology" in text_lower or "tech" in text_lower:
                    topics.append("technology")
                
                result = {
                    "sentiment": sentiment,
                    "topics": topics,
                    "word_count": word_count
                }
            
            results.append(result)
            self.total_processed += 1
        
        # Return single result or batch
        return results[0] if isinstance(text, str) else results


class DocStream(AsyncStreamProcess):
    """Streams per-chunk partial results and a final summary"""
    
    # Reuse the same analysis logic
    insights: DocInsights = Attr(default_factory=DocInsights)
    
    async def astream(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream analysis results chunk by chunk"""
        # Split into chunks for streaming
        sentences = text.split('. ')
        chunk_size = max(1, len(sentences) // 4)  # 4 chunks
        
        running_sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
        all_topics = set()
        total_words = 0
        
        for i in range(0, len(sentences), chunk_size):
            chunk = '. '.join(sentences[i:i + chunk_size])
            if not chunk:
                continue
                
            # Analyze chunk
            result = await self.insights.aforward(chunk)
            
            # Update running totals
            running_sentiment_scores[result["sentiment"]] += 1
            all_topics.update(result["topics"])
            total_words += result["word_count"]
            
            # Yield partial result
            yield {
                "type": "chunk",
                "chunk_id": i // chunk_size,
                "progress": min(100, ((i + chunk_size) / len(sentences)) * 100),
                "sentiment": result["sentiment"],
                "topics": result["topics"],
                "word_count": result["word_count"]
            }
        
        # Yield final summary
        dominant_sentiment = max(running_sentiment_scores.items(), key=lambda x: x[1])[0]
        yield {
            "type": "final",
            "sentiment": dominant_sentiment,
            "topics": list(all_topics),
            "word_count": total_words,
            "chunk_count": len(sentences) // chunk_size + (1 if len(sentences) % chunk_size else 0)
        }


class TextPrep(Process):
    """Tiny normalizer (trim, collapse whitespace)"""
    
    def forward(self, text: str) -> str:
        """Normalize text"""
        # Trim and collapse whitespace
        return ' '.join(text.split())


class ResultShape(Process):
    """Formats results for display"""
    
    format_type: str = Param(default="summary")
    
    def forward(self, result: Dict[str, Any]) -> str:
        """Format result as string"""
        if self.format_type == "summary":
            return f"Sentiment: {result['sentiment']}, Topics: {', '.join(result['topics'])}, Words: {result['word_count']}"
        elif self.format_type == "json":
            import json
            return json.dumps(result, indent=2)
        else:
            return str(result)

# %%
# Markdown
"""
## Step 1 — Hello, a single async worker

**Goal:** Learn the shape of an `AsyncProcess`.

* Run `DocInsights` on one small string.
* Show a tiny, stable output (keys like `sentiment`, `topics`, `word_count`).
* Mention: "This runs offline. To use a real LLM, set `insights.caller = your_responses_function` later."

**What to run:**
"""

# %%
# Python
# Create our async worker
insights = DocInsights()

# Analyze a single document
async def analyze_one():
    result = await insights.aforward("This is a great example of AI technology in action. The data processing is excellent.")
    return result

# Run it
result = asyncio.run(analyze_one())
print(f"Analysis result: {result}")
print(f"\nThis runs offline. To use a real LLM, set `insights.caller = your_responses_function` later.")

# %%
# Markdown
"""
## Step 2 — Concurrency with `asyncio` (quick win)

**Goal:** Feel the benefit of async before we introduce chunking.

* Call `insights` several times concurrently (e.g., `await asyncio.gather(...)`).
* Print total elapsed time to show that parallel I/O work is faster than serial calls.

**What they learn:** Async is for I/O-bound work; concurrency reduces wall-clock time.
"""

# %%
# Python
# Sample documents
docs = [
    "Artificial intelligence is transforming how we process data.",
    "The technology sector shows great promise for innovation.",
    "Machine learning algorithms are becoming more sophisticated.",
    "Data science is essential for modern business decisions.",
    "This is a terrible example of poor implementation."
]

# Serial execution
async def serial_analysis():
    start = time.time()
    results = []
    for doc in docs:
        result = await insights.aforward(doc)
        results.append(result)
    elapsed = time.time() - start
    return results, elapsed

# Concurrent execution
async def concurrent_analysis():
    start = time.time()
    tasks = [insights.aforward(doc) for doc in docs]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    return results, elapsed

# Compare
serial_results, serial_time = asyncio.run(serial_analysis())
concurrent_results, concurrent_time = asyncio.run(concurrent_analysis())

print(f"Serial execution: {serial_time:.2f}s")
print(f"Concurrent execution: {concurrent_time:.2f}s")
print(f"Speedup: {serial_time/concurrent_time:.1f}x")
print(f"\nProcessed {len(docs)} documents. Total processed so far: {insights.total_processed}")

# %%
# Markdown
"""
## Step 3 — Chunking for throughput (`Chunk`, `Recur`)

**Goal:** Move from "many single calls" to **chunked batches** and prepare for `async_process_map`.

* Show `Chunk(data=docs, n=…)` with a few `n` values (small, medium, individual).
* Use `Recur({...}, n=…)` once to pass the same config to each chunk (e.g., `include_topics=True`).

**What they learn:** Batching is a dial for cost/latency; `Recur` broadcasts constants safely.
"""

# %%
# Python
# Create chunks of different sizes
small_chunks = Chunk(data=docs, n=2)  # 2 docs per chunk
medium_chunks = Chunk(data=docs, n=3)  # 3 docs per chunk  
individual_chunks = Chunk(data=docs, n=1)  # 1 doc per chunk

print(f"Small chunks (n=2): {len(small_chunks)} chunks")
print(f"Medium chunks (n=3): {len(medium_chunks)} chunks")
print(f"Individual chunks (n=1): {len(individual_chunks)} chunks")

# Show what's in the chunks
print(f"\nFirst small chunk: {len(small_chunks[0])} documents")
print(f"First medium chunk: {len(medium_chunks[0])} documents")

# Use Recur to pass config to each chunk
config = {"include_topics": True, "verbose": False}
recurring_config = Recur(config, n=len(small_chunks))
print(f"\nRecurring config will be passed to {len(recurring_config)} chunks")

# %%
# Markdown
"""
## Step 4 — Parallel mapping with `async_process_map`

**Goal:** Apply the worker to many chunks at once and measure the impact.

* Run `async_process_map(insights, chunked_docs)` for two or three chunk sizes.
* Print a short comparison: elapsed time and a count of processed documents.

**What they learn:** `async_process_map` orchestrates parallel work, and chunk size matters.
"""

# %%
# Python
# Test different chunk sizes with async_process_map
async def test_chunk_performance():
    # Reset counter
    insights.total_processed = 0
    
    results = {}
    
    for name, chunks in [("small", small_chunks), ("medium", medium_chunks), ("individual", individual_chunks)]:
        start = time.time()
        # Process all chunks in parallel
        chunk_results = await async_process_map(insights, chunks)
        elapsed = time.time() - start
        
        # Flatten results since we get a list of lists
        all_results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, list):
                all_results.extend(chunk_result)
            else:
                all_results.append(chunk_result)
        
        results[name] = {
            "time": elapsed,
            "chunks": len(chunks),
            "docs": len(all_results)
        }
    
    return results

perf_results = asyncio.run(test_chunk_performance())

print("Chunk size comparison:")
for name, stats in perf_results.items():
    print(f"  {name}: {stats['chunks']} chunks, {stats['docs']} docs in {stats['time']:.2f}s")

# %%
# Markdown
"""
## Step 5 — Streaming results (`AsyncStreamProcess`)

**Goal:** Show incremental output for long jobs.

* Use `DocStream.astream(...)` on a longer text.
* As results arrive, print a simple progress line (`chunk_id`, `%`, `running sentiment`) and then the final object.

**What they learn:** Streaming delivers early feedback and fits long-running tasks.
"""

# %%
# Python
# Create a longer document for streaming
long_doc = """
Artificial intelligence represents one of the most transformative technologies of our time. 
The rapid advancement in machine learning has opened new possibilities. 
Data processing capabilities have grown exponentially in recent years. 
This technology brings both great opportunities and significant challenges. 
We must carefully consider the ethical implications of AI deployment. 
The benefits are clear in areas like healthcare and education. 
However, concerns about privacy and bias remain important topics. 
Overall, the future of AI looks promising with proper governance.
"""

# Stream the analysis
stream_processor = DocStream()

async def stream_analysis():
    print("Streaming analysis...")
    
    async for result in stream_processor.astream(long_doc):
        if result["type"] == "chunk":
            print(f"  Chunk {result['chunk_id']}: {result['progress']:.0f}% - Sentiment: {result['sentiment']}")
        else:  # final
            print(f"\nFinal summary:")
            print(f"  Dominant sentiment: {result['sentiment']}")
            print(f"  All topics: {', '.join(result['topics'])}")
            print(f"  Total words: {result['word_count']}")
            print(f"  Chunks processed: {result['chunk_count']}")

asyncio.run(stream_analysis())

# %%
# Markdown
"""
## Step 6 — A small pipeline with `Sequential`

**Goal:** Chain simple stages and keep data flowing.

* Build `pipeline = Sequential([TextPrep(), DocInsights(), ResultShape()])`.
* Send a few texts through the pipeline (sequentially first).
* Reuse the **same** pipeline with `async_process_map` over `Chunk(data=docs, n=…)`.

**What they learn:** Composition keeps pieces focused; the same stages work alone or in parallel.
"""

# %%
# Python
# Build a pipeline
pipeline = Sequential([
    TextPrep(),
    DocInsights(),
    ResultShape(format_type="summary")
])

# Test on single documents
print("Single document through pipeline:")
messy_doc = "  This   is    a   great   example   with   extra   spaces.  "
result = asyncio.run(pipeline.aforward(messy_doc))
print(f"  Input: '{messy_doc}'")
print(f"  Output: {result}")

# Now use the same pipeline with chunked data
print("\nChunked documents through pipeline:")
messy_docs = [
    "  AI  technology   is   amazing!  ",
    "  Data   science    provides   great   insights.  ",
    "  This  implementation   is   terrible.  "
]

async def pipeline_chunks():
    chunks = Chunk(data=messy_docs, n=2)
    results = await async_process_map(pipeline, chunks)
    return results

chunk_results = asyncio.run(pipeline_chunks())
print(f"Processed {len(messy_docs)} documents in {len(chunk_results)} chunks")
for i, result in enumerate(chunk_results):
    print(f"  Chunk {i}: {result}")

# %%
# Markdown
"""
## Step 7 — Aggregate results with `reduce` (or `async_reduce`)

**Goal:** Summarize many batch results without loading everything into memory at once.

* Create a very small aggregator (can be a one-purpose `Process` with `forward(acc, batch) -> acc`).
* Use `reduce` or `async_reduce` to compute a simple summary (counts by sentiment, top topics).
* Print the final summary.

**What they learn:** Map → Reduce is the core pattern for large jobs.
"""

# %%
# Python
class ResultAggregator(Process):
    """Simple aggregator for document insights"""
    
    def forward(self, acc: Dict[str, Any], batch: Union[Dict, List]) -> Dict[str, Any]:
        """Aggregate batch results into accumulator"""
        # Initialize accumulator if empty
        if not acc:
            acc = {
                "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
                "all_topics": set(),
                "total_words": 0,
                "doc_count": 0
            }
        
        # Handle both single results and lists
        results = [batch] if isinstance(batch, dict) else batch
        
        for result in results:
            # Skip if it's a formatted string (from ResultShape)
            if isinstance(result, str):
                continue
                
            # Update counts
            sentiment = result.get("sentiment", "neutral")
            acc["sentiment_counts"][sentiment] += 1
            acc["all_topics"].update(result.get("topics", []))
            acc["total_words"] += result.get("word_count", 0)
            acc["doc_count"] += 1
        
        return acc

# Analyze many documents
many_docs = docs * 4  # 20 documents

# Process in chunks and aggregate
async def aggregate_analysis():
    # Reset insights for fresh analysis
    fresh_insights = DocInsights()
    
    # Process documents in chunks
    chunks = Chunk(data=many_docs, n=5)
    chunk_results = await async_process_map(fresh_insights, chunks)
    
    # Flatten results 
    all_results = []
    for chunk_result in chunk_results:
        if isinstance(chunk_result, list):
            all_results.extend(chunk_result)
        else:
            all_results.append(chunk_result)
    
    # Aggregate using reduce
    aggregator = ResultAggregator()
    summary = reduce(aggregator, all_results, initial={})
    
    # Convert topics set to list for display
    summary["all_topics"] = list(summary["all_topics"])
    
    return summary

summary = asyncio.run(aggregate_analysis())

print("Aggregate summary of 20 documents:")
print(f"  Documents analyzed: {summary['doc_count']}")
print(f"  Total words: {summary['total_words']}")
print(f"  Average words/doc: {summary['total_words'] / summary['doc_count']:.1f}")
print(f"  Sentiment distribution: {dict(summary['sentiment_counts'])}")
print(f"  Topics found: {', '.join(summary['all_topics'])}")

# %%
# Markdown
"""
## Step 8 — Optional: plug in a real LLM (adapter only)

**Goal:** Keep vendor isolation while showing where to hook it in.

* Provide one small function `responses_call(prompt_or_batch) -> str|List[str]` following Tutorial 1's pattern.
* Set `insights.caller = responses_call` and run **one** short example.
* Keep the rest of the tutorial in offline mode.

**What they learn:** Real APIs live behind one function. No API-chasing across the notebook.
"""

# %%
# Python
# Example of how to plug in a real LLM (commented out by default)
"""
from openai import AsyncOpenAI

async def responses_call(text: Union[str, List[str]]) -> Union[str, List[str]]:
    client = AsyncOpenAI()
    
    if isinstance(text, str):
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Analyze sentiment and topics: {text}"}]
        )
        return response.choices[0].message.content
    else:
        # Batch processing
        tasks = [responses_call(t) for t in text]
        return await asyncio.gather(*tasks)

# To use it:
# insights.caller = responses_call
# result = await insights.aforward("Your text here")
"""

print("To use a real LLM, uncomment the code above and set:")
print("  insights.caller = responses_call")
print("\nThe rest of the tutorial runs offline by default.")

# %%
# Markdown
"""
## Step 9 — Optional: mention serialization (no deep dive)

**Goal:** Emphasize importance without turning this into a serialization lesson.

* One sentence: "Dachi's `spec()` and `state_dict()` let you save this pipeline's configuration and state."
* Optionally show **one** quick `render(pipeline.spec())` snapshot (or skip entirely if we want the leanest flow).

**What they learn:** The system can be saved and evolved later; details come in a dedicated tutorial.
"""

# %%
# Python
from dachi.core import render

print("Dachi's `spec()` and `state_dict()` let you save this pipeline's configuration and state.")
print("\nHere's a quick look at the pipeline structure:")
print(render(pipeline.spec())[:200] + "...")  # Just first 200 chars

print("\nSerialization enables saving/restoring your entire system. More details in a future tutorial!")

# %%
# Markdown
"""
## Wrap-up: What you learned

You've built a document insights pipeline that evolved from:
1. **One async worker** → understanding `AsyncProcess`
2. **Concurrent calls** → seeing async benefits for I/O
3. **Chunking** → controlling batch sizes with `Chunk`
4. **Parallel mapping** → scaling with `async_process_map`
5. **Streaming** → getting incremental results with `AsyncStreamProcess`
6. **Pipelines** → composing stages with `Sequential`
7. **Aggregation** → summarizing with `reduce`

Key takeaways:
* **Async** for I/O-bound work (API calls, file operations)
* **Chunking** to balance latency and throughput
* **Streaming** for long-running tasks with progress
* **Composition** to build complex systems from simple parts
* **Vendor isolation** - swap LLMs by changing one function

Next steps might include:
* Error handling and retries
* More complex aggregations
* Custom streaming protocols
* Production deployment patterns
"""