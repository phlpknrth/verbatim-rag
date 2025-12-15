"""
Extractors for identifying relevant spans in documents (RAG-agnostic).

This copy avoids importing vector-store specific types; accepts any objects
with a `.text` attribute as search results.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Dict

from .llm_client import LLMClient


class SpanExtractor(ABC):
    """Abstract base class for span extractors."""

    @abstractmethod
    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract relevant spans from search results.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        raise NotImplementedError

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Default async implementation that delegates to sync version."""
        import asyncio

        return await asyncio.to_thread(self.extract_spans, question, search_results)


class ModelSpanExtractor(SpanExtractor):
    """Extract spans using a fine-tuned QA sentence classification model."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        threshold: float = 0.5,
        extraction_mode: str = "individual",
        max_display_spans: int = 5,
    ):
        """
        Initialize the model-based span extractor.

        :param model_path: Path to the trained QA model
        :param device: Device to run the model on (auto-detects if None)
        :param threshold: Threshold for sentence classification
        :param extraction_mode: Not used for model extractor
        :param max_display_spans: Not used for model extractor
        """
        # Lazy-import heavy deps so importing this module doesn't require them
        import torch  # noqa: WPS433 (allow local import)
        from transformers import AutoTokenizer  # noqa: WPS433
        from .extractor_models.model import QAModel  # noqa: WPS433
        from .extractor_models.dataset import (  # noqa: WPS433
            QADataset,
            Sentence as DatasetSentence,
            Document as DatasetDocument,
            QASample,
        )

        self.model_path = model_path
        self.threshold = threshold
        self._torch = torch
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")

        # Cache dataset classes for reuse without re-import
        self.QADataset = QADataset
        self.DatasetSentence = DatasetSentence
        self.DatasetDocument = DatasetDocument
        self.QASample = QASample

        print(f"Loading model from {model_path}...")

        # Load the model
        self.model = QAModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load the tokenizer
        try:
            print(f"Loading tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Could not load tokenizer from {model_path}: {e}")
            # Fall back to base model
            base_model = getattr(
                self.model.config, "model_name", "answerdotai/ModernBERT-base"
            )
            print(f"Trying to load tokenizer from base model: {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            print(f"Loaded tokenizer from {base_model}")

    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans using the trained model.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        relevant_spans = {}

        for result in search_results:
            raw_text = getattr(result, "text", "")

            # Split text into sentences
            raw_sentences = self._split_into_sentences(raw_text)
            if not raw_sentences:
                relevant_spans[raw_text] = []
                continue

            # Create dataset objects for model processing
            dataset_sentences = [
                self.DatasetSentence(text=sent, relevant=False, sentence_id=f"s{i}")
                for i, sent in enumerate(raw_sentences)
            ]
            dataset_doc = self.DatasetDocument(sentences=dataset_sentences)

            qa_sample = self.QASample(
                question=question,
                documents=[dataset_doc],
                split="test",
                dataset_name="inference",
                task_type="qa",
            )

            dataset = self.QADataset([qa_sample], self.tokenizer, max_length=512)
            if len(dataset) == 0:
                relevant_spans[raw_text] = []
                continue

            encoding = dataset[0]

            input_ids = encoding["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(self.device)

            with self._torch.no_grad():
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentence_boundaries=[encoding["sentence_boundaries"]],
                )

            # Extract spans based on predictions
            spans = []
            if len(predictions) > 0 and len(predictions[0]) > 0:
                sentence_preds = self._torch.nn.functional.softmax(
                    predictions[0], dim=1
                )
                for i, pred in enumerate(sentence_preds):
                    if i < len(raw_sentences) and pred[1] > self.threshold:
                        spans.append(raw_sentences[i])

            relevant_spans[raw_text] = spans

        return relevant_spans


class LLMSpanExtractor(SpanExtractor):
    """Extract spans using an LLM with centralized client and batch processing."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str = "gpt-4o-mini",
        extraction_mode: str = "auto",
        max_display_spans: int = 5,
        batch_size: int = 5,
        prompt_template_name: str = "default",
    ):
        """
        Initialize the LLM span extractor.

        :param llm_client: LLM client for extraction (creates one if None)
        :param model: The LLM model to use (if creating new client)
        :param extraction_mode: "batch", "individual", or "auto"
        :param max_display_spans: Maximum spans to prioritize for display
        :param batch_size: Maximum documents to process in batch mode
        :param prompt_template_name: Name of the prompt template to use (default: "default")
        """
        from .prompt_templates import get_extraction_prompt
        
        self.llm_client = llm_client or LLMClient(model)
        self.extraction_mode = extraction_mode
        self.max_display_spans = max_display_spans
        self.batch_size = batch_size
        
        # Validate and load prompt template
        from .prompt_templates import list_available_templates
        
        try:
            self.prompt_template_func = get_extraction_prompt(prompt_template_name)
            self.prompt_template_name = prompt_template_name
        except ValueError as e:
            available = ", ".join(list_available_templates())
            raise ValueError(
                f"Invalid prompt_template_name '{prompt_template_name}'. "
                f"Available templates: {available}"
            ) from e

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans using LLM with mode selection.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        if not search_results:
            return {}

        # Decide on processing mode
        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )

        if should_batch:
            return self._extract_spans_batch(question, search_results)
        else:
            return self._extract_spans_individual(question, search_results)

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async version of span extraction.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        if not search_results:
            return {}

        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )

        if should_batch:
            return await self._extract_spans_batch_async(question, search_results)
        else:
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_batch(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans from multiple documents using batch processing.
        """
        print("Extracting spans (batch mode)...")

        # Limit to batch_size to avoid prompt size issues
        top_results = search_results[: self.batch_size]

        # Build document mapping for prompt template
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = getattr(result, "text", "")

        try:
            # Build prompt using the selected template
            prompt = self.prompt_template_func(question, documents_text)
            
            # Call LLM with JSON mode
            response = self.llm_client.complete(prompt, json_mode=True)
            extracted_data = json.loads(response)

            # Map back to original search results and verify spans
            verified_spans = {}

            # Process documents that were included in batch
            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                result_text = getattr(result, "text", "")
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result_text)
                    verified_spans[result_text] = verified
                else:
                    verified_spans[result_text] = []

            # Handle remaining documents (beyond batch_size) with empty spans
            for i in range(self.batch_size, len(search_results)):
                verified_spans[getattr(search_results[i], "text", "")] = []

            return verified_spans

        except Exception as e:
            print(f"Batch extraction failed, falling back to individual: {e}")
            return self._extract_spans_individual(question, search_results)

    async def _extract_spans_batch_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async batch extraction.
        """
        print("Extracting spans (async batch mode)...")

        top_results = search_results[: self.batch_size]

        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = getattr(result, "text", "")

        try:
            # Build prompt using the selected template
            prompt = self.prompt_template_func(question, documents_text)
            
            # Call LLM with JSON mode (async)
            response = await self.llm_client.complete_async(prompt, json_mode=True)
            extracted_data = json.loads(response)

            verified_spans = {}

            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                result_text = getattr(result, "text", "")
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result_text)
                    verified_spans[result_text] = verified
                else:
                    verified_spans[result_text] = []

            for i in range(self.batch_size, len(search_results)):
                verified_spans[getattr(search_results[i], "text", "")] = []

            return verified_spans

        except Exception as e:
            print(f"Async batch extraction failed, falling back to individual: {e}")
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_individual(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans from documents individually (one at a time).

        :param question: The query or question
        :param search_results: List of search results to process
        :return: Dictionary mapping result text to list of relevant spans
        """
        print("Extracting spans (individual mode)...")
        all_spans = {}

        for result in search_results:
            result_text = getattr(result, "text", "")
            try:
                # Build document dict for single document
                documents_text = {"doc_0": result_text}
                
                # Build prompt using the selected template
                prompt = self.prompt_template_func(question, documents_text)
                
                # Call LLM with JSON mode
                response = self.llm_client.complete(prompt, json_mode=True)
                extracted_data = json.loads(response)
                
                # Extract spans for doc_0
                extracted_spans = extracted_data.get("doc_0", [])
                
                verified = self._verify_spans(extracted_spans, result_text)
                all_spans[result_text] = verified
            except Exception as e:
                print(f"Individual extraction failed for document: {e}")
                all_spans[result_text] = []

        return all_spans

    async def _extract_spans_individual_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async individual extraction.
        """
        print("Extracting spans (async individual mode)...")
        all_spans = {}

        for result in search_results:
            result_text = getattr(result, "text", "")
            try:
                # Build document dict for single document
                documents_text = {"doc_0": result_text}
                
                # Build prompt using the selected template
                prompt = self.prompt_template_func(question, documents_text)
                
                # Call LLM with JSON mode (async)
                response = await self.llm_client.complete_async(prompt, json_mode=True)
                extracted_data = json.loads(response)
                
                # Extract spans for doc_0
                extracted_spans = extracted_data.get("doc_0", [])
                
                verified = self._verify_spans(extracted_spans, result_text)
                all_spans[result_text] = verified
            except Exception as e:
                print(f"Async individual extraction failed for document: {e}")
                all_spans[result_text] = []

        return all_spans

    def _verify_spans(self, spans: List[str], document_text: str) -> List[str]:
        """
        Verify that extracted spans actually exist in the document text.
        With normalization for whitespace and quote differences.
        """
        verified = []
        for span in spans:
            span_clean = span.strip()
            if not span_clean:
                continue
                
            # Try exact match first
            if span_clean in document_text:
                verified.append(span_clean)
                continue
                
            # Normalize both texts for comparison
            span_norm = self._normalize_for_matching(span_clean)
            doc_norm = self._normalize_for_matching(document_text)
            
            if span_norm in doc_norm:
                # Find the actual matching text in the original document
                original_match = self._find_matching_text(span_clean, document_text)
                if original_match:
                    verified.append(original_match)
                else:
                    verified.append(span_clean)  # Fallback
            # If direct match fails, try matching without quotes (for cases where quotes differ)
            elif self._match_without_quotes(span_norm, doc_norm):
                # Find the actual matching text in the original document
                original_match = self._find_matching_text(span_clean, document_text)
                if original_match:
                    verified.append(original_match)
                else:
                    verified.append(span_clean)  # Fallback
            else:
                # Simple warning when span is not found
                span_preview = span_clean[:100] + "..." if len(span_clean) > 100 else span_clean
                print(f"Warning: Span not found after normalization: '{span_preview}'")
        
        return verified

    def _normalize_for_matching(self, text: str) -> str:
        """Simple normalization for matching.
        
        Normalizes quotes and whitespace so that texts with different quote styles
        and spacing are recognized as equal.
        """
        import re
        
        # Normalize all quote variants to standard double quotes (keep quotes, just make them uniform)
        text = text.replace('``', '"').replace("''", '"')
        text = text.replace('"', '"').replace('"', '"')  # Left double quote
        text = text.replace('"', '"').replace('"', '"')  # Right double quote
        # Normalize single quotes to double quotes for consistency
        text = text.replace("'", '"').replace("'", '"')
        
        # Normalize hyphens with spaces: "real - life" -> "real-life"
        text = re.sub(r'\s+-\s+', '-', text)  # Remove spaces around hyphens
        
        # Normalize whitespace around parentheses
        # Remove space after opening parenthesis: "( text" -> "(text"
        text = re.sub(r'\(\s+', '(', text)
        # Remove space before closing parenthesis: "text )" -> "text)"
        text = re.sub(r'\s+\)', ')', text)
        
        # Normalize whitespace around quotes
        # Remove spaces before opening quotes
        text = re.sub(r'\s+"', '"', text)
        # Ensure space after closing quotes if followed by a letter (but don't add if already there)
        text = re.sub(r'"([a-zA-Z])', r'" \1', text)  # Add space if missing
        # Remove extra spaces after closing quotes
        text = re.sub(r'"\s+', '" ', text)  # Normalize to single space
        
        # Normalize whitespace around punctuation
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        
        return text.strip()
    
    def _match_without_quotes(self, span_norm: str, doc_norm: str) -> bool:
        """Check if span matches document when quotes are removed from both.
        
        This handles cases where the span and document have different quote styles
        but the same content.
        """
        import re
        # Remove all quotes for comparison
        span_no_quotes = re.sub(r'["\']', '', span_norm)
        doc_no_quotes = re.sub(r'["\']', '', doc_norm)
        return span_no_quotes in doc_no_quotes

    def _find_matching_text(self, span: str, document: str) -> str:
        """Find the original text that matches the span."""
        # This could be made more sophisticated, but for now just return the span
        return span


class RuleChefSpanExtractor(SpanExtractor):
    """
    Extract spans using RuleChef's learned rule-based models.
    
    RuleChef learns extraction rules from examples and corrections, providing
    more consistent and precise span extraction compared to direct LLM calls.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str = "gpt-4o-mini",
        examples: List[tuple[Dict[str, Any], Dict[str, Any]]] | None = None,
        task_name: str = "Span Extraction",
        task_description: str = "Extract relevant answer spans from documents",
        auto_learn: bool = True,
    ):
        """
        Initialize the RuleChef span extractor.

        :param llm_client: LLM client for RuleChef (creates one if None)
        :param model: The LLM model to use (if creating new client)
        :param examples: List of (input, output) tuples for training.
                         Input: {"question": str, "context": str}
                         Output: {"spans": List[Dict] with "text", "start", "end"}
        :param task_name: Name for the RuleChef task
        :param task_description: Description for the RuleChef task
        :param auto_learn: Whether to automatically learn rules from examples on init
        """
        # Try to import RuleChef
        try:
            from openai import OpenAI
            from rulechef import RuleChef, Task
        except ImportError as e:
            raise ImportError(
                "RuleChef is required for RuleChefSpanExtractor. "
                "Install it with: pip install rulechef"
            ) from e

        self.llm_client = llm_client or LLMClient(model)
        self.examples = examples or []
        self.task_name = task_name
        self.task_description = task_description
        self.auto_learn = auto_learn

        # Initialize OpenAI client for RuleChef
        # RuleChef needs OpenAI client, not our LLMClient wrapper
        api_key = self.llm_client.api_key
        if api_key == "EMPTY":
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "EMPTY":
            raise ValueError(
                "OpenAI API key required for RuleChef. "
                "Set OPENAI_API_KEY environment variable or provide LLMClient with API key."
            )

        openai_client = OpenAI(api_key=api_key)

        # Define the task
        self.task = Task(
            name=task_name,
            description=task_description,
            input_schema={"question": "str", "context": "str"},
            output_schema={"spans": "List[Span]"},
        )

        # Initialize RuleChef
        self.chef = RuleChef(self.task, openai_client)

        # Add examples and learn rules if provided
        if self.examples:
            for input_ex, output_ex in self.examples:
                self.chef.add_example(input_ex, output_ex)

        if self.auto_learn and self.examples:
            print(f"Learning rules from {len(self.examples)} examples...")
            try:
                self.chef.learn_rules()
                print("Rules learned successfully.")
            except Exception as e:
                print(f"⚠️  Warning: Rule learning encountered errors: {e}")
                print("   The extractor will still work, but rules may not be optimal.")
                print("   Consider using auto_learn=False or fewer examples.")

    def add_example(
        self, input_example: Dict[str, Any], output_example: Dict[str, Any]
    ) -> None:
        """
        Add a training example to RuleChef.

        :param input_example: Input dict with "question" and "context"
        :param output_example: Output dict with "spans" list
        """
        self.chef.add_example(input_example, output_example)
        self.examples.append((input_example, output_example))

    def learn_rules(self) -> None:
        """Learn rules from all added examples."""
        if not self.examples:
            print("Warning: No examples provided. Learning rules without examples.")
        self.chef.learn_rules()

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans using RuleChef's learned rules.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        if not search_results:
            return {}

        relevant_spans = {}

        for result in search_results:
            context = getattr(result, "text", "")
            if not context:
                relevant_spans[context] = []
                continue

            try:
                # Use RuleChef to extract spans
                # RuleChef.extract() expects a single dictionary argument matching input_schema
                # Returns a dictionary: {'spans': [{'text': ..., 'start': ..., 'end': ...}, ...]}
                input_dict = {"question": question, "context": context}
                result = self.chef.extract(input_dict)

                # Extract spans from result dictionary
                if isinstance(result, dict) and 'spans' in result:
                    spans = result['spans']
                elif isinstance(result, tuple) and len(result) >= 1:
                    spans = result[0]
                else:
                    spans = result if isinstance(result, list) else []

                # Convert RuleChef Span objects/dicts to strings
                # RuleChef returns spans as dicts with 'text', 'start', 'end' keys
                span_texts = []
                for span in spans:
                    if isinstance(span, dict):
                        span_texts.append(span.get("text", str(span)))
                    elif hasattr(span, "text"):
                        span_texts.append(span.text)
                    else:
                        span_texts.append(str(span))

                relevant_spans[context] = span_texts

            except Exception as e:
                print(f"RuleChef extraction failed for document: {e}")
                # Fallback: return empty spans on error
                relevant_spans[context] = []

        return relevant_spans

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async version of span extraction using RuleChef.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        import asyncio

        # RuleChef doesn't have native async support, so we run in a thread
        return await asyncio.to_thread(self.extract_spans, question, search_results)