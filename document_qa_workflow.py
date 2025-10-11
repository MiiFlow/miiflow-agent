#!/usr/bin/env python3
"""
Production-ready Document Q&A Workflow Integration
Demonstrates how to build and integrate multimedia document processing into main application workflow.

Usage:
    python document_qa_workflow.py [PDF_PATH]

Examples:
    python document_qa_workflow.py /path/to/document.pdf
"""

import asyncio
import base64
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from miiflow_llm.agents import AgentConfig, AgentContext, create_agent
from miiflow_llm.core.agent import AgentType
from miiflow_llm.core.message import Message

# from miiflow_llm.utils.pdf_extractor import extract_pdf_text, extract_pdf_metadata


# def get_pdf_path() -> Optional[str]:
#     """Get PDF path from command line arguments or prompt user."""
#     if len(sys.argv) > 1:
#         pdf_path = sys.argv[1]
#         if Path(pdf_path).exists() and pdf_path.lower().endswith(".pdf"):
#             return pdf_path
#         else:
#             print(f" Error: File '{pdf_path}' not found or not a PDF")
#             return None

#     return None


# @dataclass
# class DocumentAnalysisRequest:
#     document_path: str
#     query: str
#     document_type: str = "pdf"
#     analysis_type: str = "comprehensive"  # comprehensive, summary, specific
#     provider: str = "openai"
#     model: Optional[str] = None


# @dataclass
# class DocumentAnalysisResponse:
#     analysis: str
#     document_metadata: Dict[str, Any]
#     processing_time: float
#     provider_used: str
#     model_used: str
#     success: bool
#     error: Optional[str] = None


# class DocumentQAAgent:
#     def __init__(
#         self,
#         provider: str = "openai",
#         model: Optional[str] = None,
#         agent_type: AgentType = AgentType.SINGLE_HOP,
#         system_prompt: Optional[str] = None,
#     ):
#         self.provider = provider
#         self.model = model or self._get_default_model(provider)
#         self.agent_type = agent_type

#         if system_prompt is None:
#             system_prompt = self._get_default_system_prompt()

#         # Create the agent
#         self.agent = create_agent(
#             AgentConfig(
#                 provider=provider,
#                 model=self.model,
#                 agent_type=agent_type,
#                 tools=[],  # No tools needed for document Q&A
#                 system_prompt=system_prompt,
#                 max_iterations=10 if agent_type == AgentType.REACT else 1,
#             )
#         )

#         # Persistent context for conversation continuity
#         self.context = AgentContext()

#         print(f"✅ DocumentQAAgent initialized: {provider.upper()} ({self.model})")

#     def _get_default_model(self, provider: str) -> str:
#         model_defaults = {
#             "openai": "gpt-4o-mini",
#             "anthropic": "claude-3-5-sonnet-20241022",
#             "groq": "llama-3.1-8b-instant",
#             "xai": "grok-beta",
#             "together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#         }
#         return model_defaults.get(provider, "gpt-4o-mini")

#     def _get_default_system_prompt(self) -> str:
#         return """You are a professional document analyst with expertise in:

# DOCUMENT ANALYSIS:
# - Comprehensive document review and summarization
# - Key insight extraction and content analysis
# - Technical document evaluation
# - Resume/CV analysis and career guidance
# - Legal document analysis
# - Research paper review

# ANALYSIS APPROACH:
# - Provide structured, thorough analysis
# - Identify key themes, strengths, and areas for improvement
# - Give actionable recommendations
# - Maintain professional tone
# - Be specific and detailed in your assessments

# When analyzing documents, always provide:
# 1. Executive summary
# 2. Key findings/insights
# 3. Specific recommendations
# 4. Areas of concern (if any)
# 5. Next steps or action items"""

#     def _load_document_as_base64(self, file_path: str) -> str:
#         path = Path(file_path)

#         if not path.exists():
#             raise FileNotFoundError(f"Document not found: {file_path}")

#         with open(path, "rb") as f:
#             file_bytes = f.read()
#             b64_encoded = base64.b64encode(file_bytes).decode("utf-8")

#             # Determine MIME type
#             if path.suffix.lower() == ".pdf":
#                 mime_type = "application/pdf"
#             else:
#                 mime_type = "application/octet-stream"

#             return f"data:{mime_type};base64,{b64_encoded}"

#     def _extract_document_metadata(self, file_path: str) -> Dict[str, Any]:
#         try:
#             path = Path(file_path)
#             basic_metadata = {
#                 "filename": path.name,
#                 "file_size": path.stat().st_size,
#                 "file_extension": path.suffix.lower(),
#             }

#             if path.suffix.lower() == ".pdf":
#                 pdf_data_uri = self._load_document_as_base64(file_path)
#                 try:
#                     pdf_metadata = extract_pdf_metadata(pdf_data_uri)
#                     basic_metadata.update(pdf_metadata)
#                 except Exception as e:
#                     basic_metadata["pdf_metadata_error"] = str(e)

#             return basic_metadata

#         except Exception as e:
#             return {"error": f"Could not extract metadata: {str(e)}"}

#     async def analyze_document(self, request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
#         import time

#         start_time = time.time()

#         try:
#             pdf_data_uri = self._load_document_as_base64(request.document_path)
#             document_metadata = self._extract_document_metadata(request.document_path)

#             print(f"Processing document: {request.document_path}")
#             print(
#                 f"Metadata: {document_metadata.get('pages', '?')} pages, {document_metadata.get('file_size', 0)/1024:.1f} KB"
#             )

#             message = Message.from_pdf(
#                 text=request.query, pdf_url=pdf_data_uri, filename=Path(request.document_path).name
#             )

#             print(f"🚀 Submitting to {self.provider.upper()} ({self.model})...")
#             result = await self.agent.run(
#                 prompt=request.query, context=self.context, message_history=[message]
#             )

#             end_time = time.time()
#             processing_time = end_time - start_time

#             response_text = result.get("response", str(result))

#             return DocumentAnalysisResponse(
#                 analysis=response_text,
#                 document_metadata=document_metadata,
#                 processing_time=processing_time,
#                 provider_used=self.provider,
#                 model_used=self.model,
#                 success=True,
#             )

#         except Exception as e:
#             end_time = time.time()
#             processing_time = end_time - start_time

#             return DocumentAnalysisResponse(
#                 analysis="",
#                 document_metadata={},
#                 processing_time=processing_time,
#                 provider_used=self.provider,
#                 model_used=self.model,
#                 success=False,
#                 error=str(e),
#             )

#     async def follow_up_question(self, query: str) -> str:
#         """Ask a follow-up question using the existing context."""
#         print(f"❓ Follow-up question: {query}")

#         result = await self.agent.run(query, context=self.context)
#         return result.get("response", str(result))

#     async def analyze_with_streaming(self, request: DocumentAnalysisRequest):
#         """Analyze document with real-time streaming (requires REACT agent)."""
#         if self.agent_type != AgentType.REACT:
#             raise ValueError("Streaming analysis requires AgentType.REACT")

#         pdf_data_uri = self._load_document_as_base64(request.document_path)
#         message = Message.from_pdf(
#             text=request.query, pdf_url=pdf_data_uri, filename=Path(request.document_path).name
#         )

#         print(f"📡 Starting streaming analysis...")

#         # Stream analysis events - pass message in message_history
#         async for event in self.agent.stream_react(
#             prompt=request.query, context=self.context, message_history=[message]
#         ):
#             yield event


# class DocumentWorkflowOrchestrator:
#     """Production workflow orchestrator for document processing."""

#     def __init__(self):
#         self.agents: Dict[str, DocumentQAAgent] = {}

#     def create_agent(
#         self,
#         agent_id: str,
#         provider: str = "openai",
#         model: Optional[str] = None,
#         agent_type: AgentType = AgentType.SINGLE_HOP,
#     ) -> DocumentQAAgent:
#         agent = DocumentQAAgent(provider=provider, model=model, agent_type=agent_type)

#         self.agents[agent_id] = agent
#         print(f"🔧 Created agent '{agent_id}': {provider.upper()}")

#         return agent

#     def get_agent(self, agent_id: str) -> Optional[DocumentQAAgent]:
#         """Get an existing agent by ID."""
#         return self.agents.get(agent_id)

#     async def process_document_workflow(
#         self, agent_id: str, document_path: str, queries: List[str], provider: str = "openai"
#     ) -> List[DocumentAnalysisResponse]:
#         agent = self.get_agent(agent_id)
#         if not agent:
#             agent = self.create_agent(agent_id, provider=provider)

#         results = []

#         # Process each query
#         for i, query in enumerate(queries, 1):
#             print(f"\n📋 Query {i}/{len(queries)}: {query[:60]}...")

#             request = DocumentAnalysisRequest(
#                 document_path=document_path, query=query, provider=provider
#             )

#             response = await agent.analyze_document(request)
#             results.append(response)

#             if response.success:
#                 print(f"✅ Completed in {response.processing_time:.2f}s")
#                 print(f"📊 Response length: {len(response.analysis)} chars")
#             else:
#                 print(f"❌ Failed: {response.error}")

#         return results


# async def test_single_query_all_providers(pdf_path: str):
#     test_query = "Provide a comprehensive summary of this document, highlighting the main topics and key information."
#     providers_config = [
#         {"provider": "openai", "model": "gpt-4o-mini", "description": "OpenAI GPT-4 (PDF→Text)"},
#         {
#             "provider": "anthropic",
#             "model": "claude-3-5-sonnet-20241022",
#             "description": "Anthropic Claude (Native PDF)",
#         },
#         {
#             "provider": "groq",
#             "model": "llama-3.1-8b-instant",
#             "description": "Groq Llama (PDF→Text)",
#         },
#         {
#             "provider": "together",
#             "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#             "description": "Together AI (PDF→Text)",
#         },
#     ]

#     results = {}

#     for config in providers_config:
#         provider = config["provider"]
#         model = config["model"]
#         description = config["description"]

#         print(f"🔄 Testing {provider.upper()}: {description}")
#         print("-" * 50)

#         try:
#             agent = DocumentQAAgent(provider=provider, model=model, agent_type=AgentType.SINGLE_HOP)

#             request = DocumentAnalysisRequest(
#                 document_path=pdf_path, query=test_query, provider=provider
#             )

#             response = await agent.analyze_document(request)

#             if response.success:
#                 results[provider] = response
#                 print(f" SUCCESS: {response.processing_time:.2f}s")
#                 print(f" Response: {len(response.analysis)} chars")
#                 print(f" Answer Preview:")
#                 print(
#                     f"   {response.analysis[:150]}..."
#                     if len(response.analysis) > 150
#                     else response.analysis
#                 )
#             else:
#                 print(f" FAILED: {response.error}")

#         except Exception as e:
#             print(f"ERROR: {str(e)}")

#         print()

#     print(" MULTI-PROVIDER COMPARISON SUMMARY")
#     print("=" * 60)

#     successful_providers = [p for p in results.keys() if results[p].success]

#     if successful_providers:
#         print(f"✅ Successful providers: {len(successful_providers)}/{len(providers_config)}")
#         print("\n⏱️  Response Times:")
#         for provider in successful_providers:
#             result = results[provider]
#             processing_method = "Native PDF" if provider == "anthropic" else "PDF→Text"
#             print(
#                 f"   {provider.upper():12} | {result.processing_time:6.2f}s | {processing_method}"
#             )
#         print("\n📏 Response Lengths:")
#         for provider in successful_providers:
#             result = results[provider]
#             print(f"   {provider.upper():12} | {len(result.analysis):5d} chars")

#     else:
#         print(" No providers succeeded")

#     return results


# # WORKFLOW EXAMPLES
# async def demonstrate_production_integration(pdf_path: str):
#     orchestrator = DocumentWorkflowOrchestrator()

#     print("\n Example 1: Multi-Provider Query Test")
#     print("-" * 40)

#     await test_single_query_all_providers(pdf_path)

#     print(f"\n\n Example 2: Complete Document Analysis Workflow")
#     print("-" * 50)

#     universal_queries = [
#         "Provide a comprehensive summary of this document, highlighting the main topics and purpose.",
#         "What are the most important points or key findings presented in this document?",
#         "Identify any action items, recommendations, or next steps mentioned in the document.",
#         "What questions might someone have after reading this document, and what additional information would be helpful?",
#     ]

#     document_results = await orchestrator.process_document_workflow(
#         agent_id="document_analyzer",
#         document_path=pdf_path,
#         queries=universal_queries,
#         provider="openai",  # Provider-agnostic - can change to any provider
#     )

#     # Display results
#     for i, result in enumerate(document_results, 1):
#         if result.success:
#             print(f"\n🤖 Analysis {i}:")
#             print("-" * 30)
#             print(result.analysis[:400] + "..." if len(result.analysis) > 400 else result.analysis)


# async def demonstrate_streaming_workflow(pdf_path: str):
#     print(f"\n🌊 STREAMING DOCUMENT ANALYSIS WORKFLOW")
#     print("-" * 50)

#     streaming_agent = DocumentQAAgent(provider="openai", agent_type=AgentType.REACT)

#     request = DocumentAnalysisRequest(
#         document_path=pdf_path,
#         query="Analyze this document and provide actionable insights or recommendations based on its content.",
#         provider="openai",
#     )

#     try:
#         async for event in streaming_agent.analyze_with_streaming(request):
#             event_type = event.event_type.value
#             print(
#                 f"📡 {event_type}: {event.data.get('thought', event.data.get('answer', ''))[:100]}..."
#             )

#     except Exception as e:
#         print(f"⚠️  Streaming error: {e}")


# async def main():
#     pdf_path = get_pdf_path()
#     if not pdf_path:
#         print("  python document_qa_workflow.py /path/to/document.pdf")
#         return

#     print(f" Using PDF: {pdf_path}")
#     await demonstrate_production_integration(pdf_path)

#     await demonstrate_streaming_workflow(pdf_path)

#     print("\n" + "=" * 70)


# if __name__ == "__main__":
#     asyncio.run(main())
