"""
Prompt templates for span extraction in the Verbatim RAG system.

This module provides a registry of prompt templates that can be used
for extracting relevant spans from documents using LLMs.
"""

import json
from typing import Dict, Callable


def build_default_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Default extraction prompt - extracts exact verbatim sentences.
    
    This is the original prompt that was used in the system.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract EXACT verbatim sentences from the document that answer the question.

# Rules
1. Extract **only complete sentences** that explicitly address the question
2. Each extracted span must be a complete sentence from start to end
3. Never paraphrase, modify, or add to the original text
4. Preserve original wording, capitalization, and punctuation exactly
5. Order sentences by relevance - MOST RELEVANT FIRST
6. Extract sentence boundaries precisely (start with capital letter, end with punctuation)

# Output Format
Return a JSON object mapping document IDs to sentence arrays ordered by relevance:
{{
  "doc_0": ["most relevant sentence", "next most relevant sentence"],
  "doc_1": ["most relevant from doc 1"],
  "doc_2": []
}}

If no relevant sentences found, use empty array.

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract complete sentences from each document:"""


def build_concise_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Concise extraction prompt - shorter version for faster processing.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract exact verbatim sentences that answer the question.

Rules:
- Extract complete sentences only
- Preserve original wording exactly
- Order by relevance (most relevant first)
- Return JSON: {{"doc_0": ["sentence1", "sentence2"], "doc_1": []}}

Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract sentences:"""


def build_detailed_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Detailed extraction prompt - more explicit instructions for complex cases.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""You are an expert at extracting precise, verbatim text spans from documents.

# Task
Extract EXACT verbatim sentences from the provided documents that directly answer the question.

# Critical Rules
1. Extract **only complete sentences** that explicitly address the question
   - A complete sentence starts with a capital letter and ends with punctuation (. ! ?)
   - Do not extract partial sentences or sentence fragments
2. Each extracted span must be a complete sentence from start to end
   - Include the full sentence, including any punctuation
   - Do not truncate or modify sentences
3. Never paraphrase, modify, or add to the original text
   - Copy text exactly as it appears in the document
   - Preserve all capitalization, punctuation, and formatting
4. Preserve original wording, capitalization, and punctuation exactly
   - Do not correct grammar or spelling
   - Do not change capitalization
   - Maintain original punctuation marks
5. Order sentences by relevance - MOST RELEVANT FIRST
   - The most directly relevant sentences should appear first
   - Less relevant but still related sentences should follow
6. Extract sentence boundaries precisely
   - Start with capital letter
   - End with punctuation (. ! ?)
   - Include all words in the sentence

# Output Format
Return a JSON object mapping document IDs to sentence arrays ordered by relevance:
{{
  "doc_0": ["most relevant sentence", "next most relevant sentence"],
  "doc_1": ["most relevant from doc 1"],
  "doc_2": []
}}

Important:
- If no relevant sentences are found for a document, use an empty array []
- Each sentence should be a complete, standalone sentence
- Do not include explanations or summaries in the output

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract complete sentences from each document:"""


def build_comprehensive_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Comprehensive extraction prompt - extracts more sentences, including indirectly relevant ones.
    
    Designed to address the issue where default prompt extracts too few sentences.
    This prompt is more inclusive and extracts sentences that are relevant, even if not
    directly answering the question.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract EXACT verbatim sentences from the documents that are relevant to answering the question.

# Extraction Strategy
Be COMPREHENSIVE - extract all sentences that could be useful for answering the question, including:
- Sentences that directly answer the question
- Sentences that provide context or background information
- Sentences that address related aspects or sub-questions
- Sentences that contain relevant facts, data, or explanations

# Rules
1. Extract **complete sentences only** (start with capital letter, end with punctuation)
2. Preserve original wording, capitalization, and punctuation EXACTLY - never modify text
3. Order by relevance: MOST RELEVANT FIRST, then related/contextual sentences
4. When in doubt, include the sentence rather than exclude it
5. Extract sentences that help understand or answer the question, even if indirectly

# Output Format
Return a JSON object mapping document IDs to sentence arrays ordered by relevance:
{{
  "doc_0": ["most relevant sentence", "next most relevant", "contextual sentence"],
  "doc_1": ["relevant from doc 1", "supporting sentence"],
  "doc_2": []
}}

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract all relevant sentences from each document (be comprehensive):"""


def build_contextual_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Contextual extraction prompt - extracts answer sentences plus supporting context.
    
    This prompt explicitly asks for both direct answers and contextual sentences that
    help understand the answer. Useful when you need more complete information.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract EXACT verbatim sentences that answer the question AND provide relevant context.

# What to Extract
1. **Primary sentences**: Sentences that directly answer the question
2. **Context sentences**: Sentences that provide background, explanation, or supporting information
3. **Related sentences**: Sentences that address related aspects or help understand the answer

# Rules
1. Extract **only complete sentences** (capital letter to punctuation: . ! ?)
2. Preserve original text EXACTLY - no modifications, paraphrasing, or additions
3. Order by relevance: direct answers first, then context and supporting information
4. Include sentences that help build a complete understanding of the answer
5. Extract multiple sentences per document when relevant information is spread across sentences

# Output Format
Return a JSON object mapping document IDs to sentence arrays ordered by relevance:
{{
  "doc_0": ["direct answer sentence", "supporting context sentence", "additional relevant sentence"],
  "doc_1": ["answer from doc 1", "explanatory sentence"],
  "doc_2": []
}}

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract answer sentences and relevant context from each document:"""


def build_generous_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Generous extraction prompt - very inclusive, extracts all potentially relevant sentences.
    
    This is the most permissive prompt. Use when you want to maximize recall and
    are willing to include sentences that might be only tangentially related.
    Useful for ensuring no relevant information is missed.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract EXACT verbatim sentences that are relevant to the question.

# Extraction Philosophy
Be GENEROUS and INCLUSIVE - extract any sentence that could potentially be useful:
- Sentences that directly answer the question
- Sentences that provide context, background, or explanation
- Sentences that mention related topics, concepts, or entities from the question
- Sentences that contain relevant keywords or themes
- When uncertain, err on the side of inclusion

# Rules
1. Extract **complete sentences only** (must start with capital letter and end with . ! ?)
2. Preserve original text EXACTLY - copy verbatim, no changes whatsoever
3. Order by relevance: most directly relevant first, then less direct but still relevant
4. Extract multiple sentences per document - don't limit yourself to just one or two
5. Include sentences even if they only partially relate to the question

# Output Format
Return a JSON object mapping document IDs to sentence arrays ordered by relevance:
{{
  "doc_0": ["most relevant", "also relevant", "potentially useful", "contextual info"],
  "doc_1": ["relevant sentence", "related sentence"],
  "doc_2": []
}}

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract all potentially relevant sentences (be generous and inclusive):"""


def build_fewshot_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Few-shot extraction with positive and negative examples.
    
    Research shows that 2-5 examples can improve performance 2-3x.
    Shows the model exactly what good and bad extractions look like.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract EXACT verbatim sentences from the document that answer the question.

# Example 1: CORRECT Extraction (Real Dataset Example)
Question: "what are the main parts of the female reproductive system"

Document: "The female reproductive system is made up of the internal and external sex organs that function in reproduction of new offspring . In the human the female reproductive system is immature at birth and develops to maturity at puberty to be able to produce gametes , and to carry a fetus to full term . The internal sex organs are the uterus , Fallopian tubes , and ovaries . The uterus or womb accommodates the embryo which develops into the fetus . The uterus also produces vaginal and uterine secretions which help the transit of sperm to the Fallopian tubes . The ovaries produce the ova ( egg cells ) . The external sex organs are also known as the genitals and these are the organs of the vulva including the labia , clitoris , and vaginal opening . The vagina is connected to the uterus at the cervix ."

✓ CORRECT Output:
{{
  "doc_0": [
    "The female reproductive system is made up of the internal and external sex organs that function in reproduction of new offspring .",
    "The internal sex organs are the uterus , Fallopian tubes , and ovaries .",
    "The external sex organs are also known as the genitals and these are the organs of the vulva including the labia , clitoris , and vaginal opening .",
    "The vagina is connected to the uterus at the cervix ."
  ]
}}

Why this is correct:
- Each sentence is copied EXACTLY (including spaces before punctuation like " .")
- Only sentences that directly answer "what are the main parts" are included
- Sentences about development/maturity are NOT included (not about "parts")
- Order is logical: overview → internal → external → connection

# Example 2: INCORRECT Extractions (DO NOT do this)

❌ WRONG - Paraphrasing:
{{
  "doc_0": [
    "The female reproductive system has internal and external organs.",
    "Internal organs include uterus, tubes, and ovaries."
  ]
}}
Why wrong: Changed wording ("has" instead of "is made up of", removed "Fallopian")

❌ WRONG - Partial sentences:
{{
  "doc_0": [
    "the uterus , Fallopian tubes , and ovaries",
    "the labia , clitoris , and vaginal opening"
  ]
}}
Why wrong: Not complete sentences (no capital letter, no subject)

❌ WRONG - Removed punctuation/spacing:
{{
  "doc_0": [
    "The internal sex organs are the uterus, Fallopian tubes, and ovaries."
  ]
}}
Why wrong: Changed " ," to "," - must keep original spacing exactly

❌ WRONG - Combined sentences:
{{
  "doc_0": [
    "The internal sex organs are the uterus, Fallopian tubes, and ovaries, and the external sex organs are the genitals."
  ]
}}
Why wrong: Combined two separate sentences with "and"

❌ WRONG - Including irrelevant sentences:
{{
  "doc_0": [
    "The female reproductive system is made up of the internal and external sex organs that function in reproduction of new offspring .",
    "In the human the female reproductive system is immature at birth and develops to maturity at puberty to be able to produce gametes , and to carry a fetus to full term .",
    "The internal sex organs are the uterus , Fallopian tubes , and ovaries ."
  ]
}}
Why wrong: Second sentence is about development, not about "what are the main parts"

# Rules
1. Extract ONLY complete sentences verbatim - copy exactly as written
2. Preserve ALL spacing and punctuation (even " ." or " ,")
3. Never paraphrase, modify, combine, or add words
4. Only extract sentences that directly answer the question
5. Order by relevance: most relevant first
6. Return ONLY valid JSON - no markdown, no backticks, no explanations

# Output Format
Return pure JSON without any markdown formatting:
{{"doc_0": ["sentence1", "sentence2"], "doc_1": []}}

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract sentences (pure JSON only):"""

def build_verbatim_strict_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Ultra-strict verbatim extraction - emphasizes ZERO modifications.
    
    Research shows models tend to paraphrase even when they find the right information.
    This prompt uses extreme emphasis on character-by-character copying.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""CRITICAL INSTRUCTION: Extract sentences EXACTLY as written - character-by-character copy.

# Your ONLY Task
Find sentences that answer the question and copy them EXACTLY - not a single character changed.

# What "EXACTLY" means:
✓ Copy every word, space, punctuation mark identically
✓ Keep original capitalization (don't change "iPhone" to "Iphone")
✓ Keep original punctuation (don't change "Dr." to "Dr")
✓ Keep original spelling, even typos (don't "fix" anything)
✓ Complete sentences only (start with capital, end with . ! ?)

# What is ABSOLUTELY FORBIDDEN:
✗ Paraphrasing or rewording ("fast car" → "speedy vehicle")
✗ Adding any words ("The car" → "The fast car")
✗ Removing any words ("The fast car" → "The car")
✗ Fixing grammar or typos (copy errors exactly as they appear)
✗ Changing tense ("he runs" → "he ran")
✗ Changing singular/plural ("car" → "cars")
✗ Combining sentences into one
✗ Splitting sentences into parts
✗ Partial sentences or fragments

Think of yourself as a photocopy machine: reproduce text EXACTLY, or don't include it.

# Output Format
Pure JSON only (no markdown, no code blocks, no backticks, no explanations):
{{"doc_0": ["exact sentence 1", "exact sentence 2"], "doc_1": []}}

Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Output pure JSON with exact verbatim sentences:"""



def build_negative_example_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Uses negative examples to explicitly show what NOT to do.
    
    Research shows that showing both positive and negative examples helps
    models learn what constitutes "bad" output.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract EXACT verbatim sentences that answer the question.

# Examples of BAD Extraction (DO NOT DO THIS):

❌ BAD Example 1 - Paraphrasing:
Original: "The company was founded in 1998 by two engineers."
WRONG: "It was started in 1998 by two people."
Why wrong: Changed words (founded→started, engineers→people, company→it)

❌ BAD Example 2 - Adding words:
Original: "The car is fast."
WRONG: "The car is very fast."
Why wrong: Added the word "very"

❌ BAD Example 3 - Partial sentence:
Original: "The company was founded in 1998."
WRONG: "founded in 1998"
Why wrong: Missing subject, not a complete sentence

❌ BAD Example 4 - Combining sentences:
Original: "The car is red. It is fast."
WRONG: "The car is red and fast."
Why wrong: Combined two sentences, added "and"

❌ BAD Example 5 - Fixing typos:
Original: "The mesage was recieved yesterday."
WRONG: "The message was received yesterday."
Why wrong: Corrected spelling errors (keep original spelling)

❌ BAD Example 6 - Irrelevant content:
Question: "How do cars work?"
Original: "Cars use engines. Boats use propellers."
WRONG: {{"doc_0": ["Cars use engines.", "Boats use propellers."]}}
Why wrong: Second sentence is about boats, not cars

# What TO Do (CORRECT Approach):
✓ Copy complete sentences word-for-word, character-for-character
✓ Keep original formatting, capitalization, punctuation, spelling
✓ Extract only sentences that directly address the question
✓ Order by relevance (most relevant first)
✓ Use empty array [] if no relevant sentences exist

# Output Format (pure JSON, no markdown):
{{"doc_0": ["verbatim sentence 1", "verbatim sentence 2"], "doc_1": []}}

Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract (pure JSON only):"""


def build_twostage_extraction_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Two-stage extraction: First identify, then extract verbatim.
    
    Breaking complex tasks into stages can improve accuracy for difficult
    retrieval tasks.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Task: Extract EXACT verbatim sentences using a two-stage approach.

# Stage 1: IDENTIFY (Mental step - don't output this)
First, mentally identify which sentences are relevant to the question:
- Read each sentence carefully
- Categorize sentences:
  * PRIMARY: Sentences that directly answer the question
  * CONTEXT: Sentences that provide background, explanation, or supporting information
- Include both categories, ordered by relevance

# Stage 2: EXTRACT EXACTLY (Output this)
For each identified sentence, copy it EXACTLY:
- Every word must match the original
- Every character must match (punctuation, capitalization, spelling)
- No changes, no additions, no paraphrasing, no corrections
- Complete sentences only (capital letter → punctuation: . ! ?)

# Verification Checklist (apply before outputting):
□ Is this sentence verbatim from the document? (character-by-character match)
□ Is it a complete sentence? (starts with capital, ends with . ! ?)
□ Does it fit one of the categories: primary, context, or related?
□ Is the order correct? (primary first, then context, then related)

# Output Format
Pure JSON object only:
{{"doc_0": ["exact sentence 1", "exact sentence 2"], "doc_1": []}}

Requirements:
- NO markdown formatting (no ``` or ```json or code blocks)
- NO explanations, commentary, or metadata
- Only return the JSON object
- Empty array [] if no relevant sentences found

Question: {question}

Documents:
{json.dumps(documents, indent=2)}

JSON output:"""


def build_antimarkdown_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Extraction prompt with explicit anti-markdown instructions.
    
    LLMs commonly wrap JSON in markdown code blocks, breaking parsers.
    This prompt aggressively prevents markdown formatting.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""Extract EXACT verbatim sentences that answer the question.

# CRITICAL OUTPUT REQUIREMENT:
Your response must be PURE JSON ONLY - nothing else.
DO NOT wrap JSON in markdown code blocks.
DO NOT use ``` or ```json or any backticks.
DO NOT add any text before or after the JSON.
Just output the raw JSON object directly.

# Extraction Rules:
1. Copy complete sentences exactly as written (verbatim)
2. Never paraphrase, modify, or add to the text
3. Order by relevance (most relevant first)
4. Each sentence must be complete (capital → punctuation)

# Output Format Example:
{{"doc_0": ["sentence 1", "sentence 2"], "doc_1": []}}

NOT like this:
```json
{{"doc_0": ["sentence 1"]}}
```

NOT like this:
Here is the output:
{{"doc_0": ["sentence 1"]}}

CORRECT - Just the JSON:
{{"doc_0": ["sentence 1"]}}

Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Output (pure JSON, no markdown):"""


def build_instruction_first_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Instruction-first prompt - places critical instructions at the start.
    
    Places the most critical instruction (verbatim extraction) at the very
    beginning for maximum attention.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""COPY SENTENCES EXACTLY - DO NOT CHANGE ANY WORDS.

Extract complete sentences from documents that answer the question.
Copy each sentence character-by-character, preserving all words, punctuation, and capitalization.

Rules:
1. Verbatim extraction only - exact copy, zero modifications
2. Complete sentences (capital letter → . ! ?)
3. Most relevant sentences first
4. JSON format: {{"doc_0": ["sentence"], "doc_1": []}}
5. No markdown formatting - pure JSON only

Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract (JSON only):"""


def build_roleplay_prompt(question: str, documents: Dict[str, str]) -> str:
    """
    Role-playing prompt - assigns the model a specific expert role.
    
    Giving the model a specific role can improve task adherence and accuracy.
    
    :param question: The user's question
    :param documents: Dictionary mapping document IDs to document text
    :return: Formatted prompt string
    """
    return f"""You are a professional text extraction specialist with 10 years of experience in legal document processing, where accuracy is absolutely critical. Your specialty is extracting exact verbatim text - you never paraphrase or modify source material.

# Your Task
Extract complete sentences from documents that answer the given question.
Use your expertise to identify relevant sentences and copy them with 100% accuracy.

# Professional Standards (non-negotiable):
1. **Verbatim extraction**: Copy text exactly, character-by-character
2. **Complete sentences**: Extract only full sentences (capital → punctuation)
3. **No modifications**: Never paraphrase, add words, or correct errors
4. **Relevance ranking**: Most relevant sentences first
5. **Quality control**: Verify each sentence is an exact match before including it

# Output Format
Standard JSON format used in legal systems:
{{"doc_0": ["exact sentence 1", "exact sentence 2"], "doc_1": []}}

No markdown formatting - output pure JSON only.

# Case Details
Question: {question}

Source Documents:
{json.dumps(documents, indent=2)}

Your extraction (JSON format):"""


# Registry of available extraction prompt templates
EXTRACTION_PROMPT_REGISTRY: Dict[str, Callable[[str, Dict[str, str]], str]] = {
    "default": build_default_extraction_prompt,
    "concise": build_concise_extraction_prompt,
    "detailed": build_detailed_extraction_prompt,
    "comprehensive": build_comprehensive_extraction_prompt,
    "contextual": build_contextual_extraction_prompt,
    "generous": build_generous_extraction_prompt,
    "fewshot": build_fewshot_extraction_prompt,
    "verbatim_strict": build_verbatim_strict_prompt,
    "negative_example": build_negative_example_prompt,
    "twostage": build_twostage_extraction_prompt,
    "antimarkdown": build_antimarkdown_prompt,
    "instruction_first": build_instruction_first_prompt,
    "roleplay": build_roleplay_prompt,
}


def get_extraction_prompt(template_name: str) -> Callable[[str, Dict[str, str]], str]:
    """
    Get an extraction prompt template function by name.
    
    :param template_name: Name of the template to retrieve
    :return: Template function that takes (question, documents) and returns a prompt string
    :raises ValueError: If template_name is not found in registry
    """
    if template_name not in EXTRACTION_PROMPT_REGISTRY:
        available = ", ".join(sorted(EXTRACTION_PROMPT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown extraction prompt template: '{template_name}'. "
            f"Available templates: {available}"
        )
    return EXTRACTION_PROMPT_REGISTRY[template_name]


def list_available_templates() -> list[str]:
    """
    Get a list of all available template names.
    
    :return: List of template names
    """
    return sorted(EXTRACTION_PROMPT_REGISTRY.keys())

