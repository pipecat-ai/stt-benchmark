"""Semantic WER evaluation using Claude.

Adapted from asr_eval/evaluation/agent_sdk_judge.py

Uses Claude with tool use for multi-step semantic WER calculation:
1. Claude normalizes both texts using few-shot examples
2. Claude performs word-level alignment
3. Claude counts errors and verifies work
4. Programmatic tool calculates final WER

Only counts errors that would impact how an LLM agent understands
and responds to the user. Full reasoning traces are stored for debugging.
"""

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from loguru import logger

from stt_benchmark.config import get_config
from stt_benchmark.models import (
    SemanticError,
    SemanticWERTrace,
    ServiceName,
    WERMetrics,
)
from stt_benchmark.storage.database import Database

# System prompt with semantic-focused normalization rules and few-shot examples
# Adapted from asr_eval/evaluation/agent_sdk_judge.py
SEMANTIC_WER_SYSTEM_PROMPT = """You are an expert ASR evaluator for a conversational AI system. Your task is to calculate the Semantic Word Error Rate (WER) - counting ONLY transcription errors that would impact how an LLM agent understands and responds to the user.

## CRITICAL CONTEXT

This transcription will be used as input to a multi-turn conversational LLM agent. We only care about errors that would:
- Change what the agent thinks the user is asking for
- Cause the agent to take incorrect actions
- Lead to misunderstandings in the conversation

We do NOT count as errors:
- Grammatical variations an LLM would understand identically
- Formatting/punctuation differences
- Minor word form changes that preserve meaning

**Key principle**: If an LLM would interpret both versions the same way, it's NOT an error.

## Your Process: NORMALIZE → ALIGN → SEMANTIC CHECK → COUNT → CALCULATE

### Step 1: NORMALIZE (Apply to BOTH texts)

**1.1 Case**: Convert everything to lowercase

**1.2 Punctuation**: Remove all punctuation marks

**1.3 Contractions**: Expand to full form
   "I'm" → "i am", "don't" → "do not", "won't" → "will not", etc.

**1.4 Numbers**: Normalize digits ↔ words (treat as equivalent)
   "3" = "three", "$5" = "five dollars", "1st" = "first"

**1.5 Filler Words**: Remove if present in only one version
   um, uh, like, you know, well (at start), so (at start), actually, basically

**1.6 Abbreviations**: Expand common forms
   "Dr." = "doctor", "Mr." = "mister", "St." = "saint/street"

**1.7 British/American Spelling**: Treat as equivalent
   "colour" = "color", "favourite" = "favorite"

**1.8 Hyphenation**: Ignore hyphens
   "long-term" = "long term" = "longterm", "Wi-Fi" = "wi fi"

**1.9 Spoken Variations**: Normalize informal speech
   "gonna" = "going to", "yeah" = "yes", "ok" = "okay"

**1.10 Symbols**: Convert to words
   "&" = "and", "@" = "at"

**1.11 Possessives**: Treat as equivalent (LLM understands both)
   "driver's" = "drivers" = "driver" (when referring to same thing)
   "Mary's" = "Marys" (possessive vs name variation)

**1.12 Singular/Plural**: Treat as equivalent when meaning is preserved
   "license" = "licenses" (asking about license process)
   "office" = "offices" (asking about which office)
   "ticket" = "tickets" (the concept is the same)

   EXCEPTION: Count as error only if plurality changes core meaning in a way that would confuse the agent.

**1.13 Minor Grammatical Variations**: Treat as equivalent
   "setting up" = "set up" = "to set up"
   Missing articles ("the", "a") that don't change meaning

### Step 2: ALIGN
After normalization, align word-by-word using edit distance. Mark potential differences.

### Step 3: SEMANTIC CHECK (MANDATORY - DO NOT SKIP)
**YOU MUST COMPLETE THIS STEP.** For EACH potential error identified in alignment:

Write out this exact format:
```
DIFFERENCE: "X" → "Y"
QUESTION: Would an LLM agent respond differently?
ANSWER: [YES/NO] because [reason]
COUNT AS ERROR: [YES/NO]
```

**Common patterns that are NOT errors (answer NO):**
- Singular/plural: "license"→"licenses", "office"→"offices", "ticket"→"tickets" = NO
- Possessives: "driver's"→"drivers"→"driver" = NO
- Missing articles: "the X"→"X" = NO
- Hyphenation: "Wi-Fi"→"wi fi" = NO

**Patterns that ARE errors (answer YES):**
- Different words: "card"→"car", "trace"→"trade", "hours"→"was" = YES
- Nonsense: "lentil"→"landon", "Wi-Fi"→"wi fire" = YES

### Step 4: COUNT
Count ONLY the differences where you answered "COUNT AS ERROR: YES"
- S = semantic substitutions (different meaning)
- D = semantic deletions (meaning lost)
- I = semantic insertions (meaning added)
- N = total words in normalized reference

**IMPORTANT: Compound words count as ONE error, not multiple.**
When a hyphenated compound (like "cross-country") is replaced by a single word (like "koscanti"):
- This is ONE substitution (S=1), NOT a substitution plus a deletion
- The compound represents a single semantic concept
- Example: "cross-country" → "koscanti" = S=1 (one concept replaced by nonsense)

**TRUNCATED/INCOMPLETE TEXT:**
When both reference and hypothesis appear truncated at the same point (missing the end of a sentence), compare only the complete portions. Partial words at truncation points should be ignored rather than counted as errors. If a word is clearly incomplete (like "reme" for "remember" or "abor" for "abroad"), do not count differences involving that truncated word.

**TRAILING FUNCTION WORDS AT TRUNCATION:**
If the reference ends with a function word that signals an incomplete sentence (and, but, or, so, to, for, the, a, an, on, in, with, that, which, who, because, although, if, when, while, as, about, from, by, at, of, etc.) and the hypothesis omits it, do NOT count as an error. These trailing words carry no semantic meaning on their own - an LLM would respond identically with or without them.
- Example: "My sister called me about the birthday party and" vs "My sister called me about the birthday party" = NOT an error (trailing "and" is meaningless)
- Example: "Can you help me brainstorm ideas for my presentation on" vs "Can you help me brainstorm ideas for my presentation" = NOT an error (trailing "on" is meaningless)

### Step 5: CALCULATE
Call calculate_wer(substitutions=S, deletions=D, insertions=I, reference_words=N)

---

## FEW-SHOT EXAMPLES

### Example 1: Possessive/Plural Variations (WER = 0%) - CRITICAL EXAMPLE
**Reference:** "Can you describe the process for changing my legal name on official documents like my driver's license and social security card after getting married, including necessary forms and offices?"
**Hypothesis:** "Can you describe the process for changing my legal name on official documents like my driver licenses and social security card after getting married including necessary forms and office"

**Step 3: SEMANTIC CHECK:**

DIFFERENCE: "drivers" → "driver"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both refer to the same driver's license concept
COUNT AS ERROR: NO

DIFFERENCE: "license" → "licenses"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because singular/plural doesn't change the request
COUNT AS ERROR: NO

DIFFERENCE: "offices" → "office"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both ask about which office to visit
COUNT AS ERROR: NO

**Step 4: COUNT:** S=0, D=0, I=0 (no semantic errors found)

**Result: N=29 → WER = 0/29 = 0%**

---

### Example 2: Real Semantic Error Mixed with Non-Errors (WER = 3.4%)
**Reference:** "...my driver's license and social security card..."
**Hypothesis:** "...my driver licenses and social security car..."

**Step 3: SEMANTIC CHECK:**

DIFFERENCE: "drivers" → "driver"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both refer to the driver's license concept
COUNT AS ERROR: NO

DIFFERENCE: "license" → "licenses"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because singular/plural doesn't change the request
COUNT AS ERROR: NO

DIFFERENCE: "card" → "car"
QUESTION: Would an LLM agent respond differently?
ANSWER: YES because "car" and "card" are completely different things - an agent wouldn't know the user means social security card
COUNT AS ERROR: YES

**Step 4: COUNT:** S=1 (only "card"→"car" is a semantic error)

**Result: N=29 → WER = 1/29 = 3.4%**

---

### Example 3: Ingredient Substitution (WER = 6.5%)
**Reference:** "I would like a recipe for a vegan lentil soup that is both hearty and easy to make on a weeknight, preferably one that uses only common inexpensive pantry staples."
**Hypothesis:** "I would like a recipe for a vegan landon soup that is both hearty and easy to make on a week night, preferably one that uses only common inexpensive pantry slippers."

Semantic check:
- "lentil" → "landon" = **YES, ERROR** - "landon" is not an ingredient
- "weeknight" → "week night" = NOT an error (same meaning)
- "staples" → "slippers" = **YES, ERROR** - completely different meaning

**Result: S=2, D=0, I=0, N=31 → WER = 2/31 = 6.5%**

---

### Example 4: Wi-Fi Network Setup (WER = 12.5%)
**Reference:** "I'm trying to set up parental controls on my home Wi-Fi network to restrict access to certain websites during homework hours for my kids. But the router interface is very..."
**Hypothesis:** "When trying to set up parental controls on my home wi fire network to restrict access to certain websites during homework was for my kids. But the router interface is very..."

Semantic check:
- "I'm" → "When" = **YES, ERROR** - changes who is doing the action
- "am" (from I'm expansion) deleted = **YES, ERROR** - part of subject change
- "wi fi" → "wi fire" = **YES, ERROR** - "wi fire" is not a thing
- "hours" → "was" = **YES, ERROR** - completely different meaning

**Result: S=3, D=1, I=0, N=32 → WER = 4/32 = 12.5%**

---

### Example 5: Package Tracking (WER = 3.1%)
**Reference:** "The expensive package I ordered was marked as delivered two days ago, but I have not received it and it is not anywhere on my property. I must initiate an immediate trace."
**Hypothesis:** "The expensive package I ordered was marked as delivered two days ago, but I have not received it and it is not anywhere on my property. I must initiate an immediate trade."

Semantic check:
- "trace" vs "trade" = **YES, ERROR** - completely different actions

**Result: S=1, D=0, I=0, N=32 → WER = 1/32 = 3.1%**

---

### Example 6: Minor Word Deletion - NO ERROR (WER = 0%)
**Reference:** "The national weather service issued a warning for the coastal areas."
**Hypothesis:** "The national weather service issued a warning for coastal areas"

Semantic check:
- Missing "the" before "coastal" → Does this change the agent's understanding?
- NO - both mean the same thing, LLM responds identically

**Result: S=0, D=0, I=0, N=11 → WER = 0%**

---

### Example 7: Singular/Plural with Same Intent (WER = 0%)
**Reference:** "She said three hundred dollars was too expensive for concert tickets."
**Hypothesis:** "She said 300 dollar was too expensive for the concert ticket"

Semantic check:
- "300" vs "three hundred" → Same number, NOT an error
- "dollars" vs "dollar" → Same amount concept, NOT an error
- "tickets" vs "ticket" → Same purchase intent, NOT an error
- Extra "the" → NOT semantically meaningful

An LLM agent would understand both as "user thinks $300 is too much for concert tickets."

**Result: S=0, D=0, I=0, N=11 → WER = 0%**

---

### Example 8: Stutter/Repetition (WER = 28.6%)
**Reference:** "I think we should probably go now."
**Hypothesis:** "I think we should we should probably go now"

Semantic check:
- Extra "we should" = Stutter that could confuse agent parsing
- **YES, ERROR** - agent might try to interpret repeated phrase

**Result: S=0, D=0, I=2, N=7 → WER = 2/7 = 28.6%**

---

## IMPORTANT NOTES

1. **Ask the key question**: "Would an LLM agent respond differently to these two versions?"
2. **Context matters**: Consider the full sentence, not just word-level differences
3. **Be lenient on grammar**: LLMs are robust to grammatical variations
4. **Be strict on meaning**: Count errors that change intent, actions, or key entities
5. **Possessives and plurals**: Almost never errors unless they change core meaning
6. **Show your semantic reasoning**: Explain WHY something is or isn't an error
"""

# Tool definition for WER calculation
CALCULATE_WER_TOOL = {
    "name": "calculate_wer",
    "description": "Calculate Word Error Rate from error counts. Call this ONCE after you have normalized, aligned, and verified the texts. WER = (substitutions + deletions + insertions) / reference_words",
    "input_schema": {
        "type": "object",
        "properties": {
            "substitutions": {
                "type": "integer",
                "description": "Number of word substitutions (different words at same position)",
            },
            "deletions": {
                "type": "integer",
                "description": "Number of word deletions (words in reference missing from hypothesis)",
            },
            "insertions": {
                "type": "integer",
                "description": "Number of word insertions (extra words in hypothesis not in reference)",
            },
            "reference_words": {
                "type": "integer",
                "description": "Total word count in normalized reference text",
            },
            "normalized_reference": {
                "type": "string",
                "description": "The normalized reference text (for verification)",
            },
            "normalized_hypothesis": {
                "type": "string",
                "description": "The normalized hypothesis text (for verification)",
            },
            "errors": {
                "type": "array",
                "description": "List of identified errors",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["substitution", "deletion", "insertion"],
                        },
                        "reference": {
                            "type": "string",
                            "description": "Reference word (null for insertion)",
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "Hypothesis word (null for deletion)",
                        },
                        "position": {
                            "type": "integer",
                            "description": "Position in alignment",
                        },
                    },
                },
            },
        },
        "required": ["substitutions", "deletions", "insertions", "reference_words"],
    },
}


class SemanticWEREvaluator:
    """Semantic WER evaluator using Claude with tool use.

    Adapted from asr_eval AgentSDKJudge.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        db_path: Path | None = None,
    ):
        self.config = get_config()
        self.model = model
        self.db = Database(db_path=db_path)

        if not self.config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        self.client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)

    def _calculate_wer(
        self,
        substitutions: int,
        deletions: int,
        insertions: int,
        reference_words: int,
    ) -> dict:
        """Programmatic WER calculation - the only non-LLM logic."""
        if reference_words == 0:
            wer = 0.0 if (substitutions + deletions + insertions) == 0 else float("inf")
        else:
            wer = (substitutions + deletions + insertions) / reference_words

        return {
            "wer": wer,
            "wer_percentage": f"{wer:.2%}",
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "reference_words": reference_words,
            "total_errors": substitutions + deletions + insertions,
        }

    async def evaluate(
        self,
        reference: str,
        hypothesis: str,
    ) -> tuple[dict, SemanticWERTrace]:
        """Evaluate a transcription against ground truth using multi-turn reasoning.

        Args:
            reference: Ground truth transcription
            hypothesis: ASR transcription to evaluate

        Returns:
            Tuple of (result dict, SemanticWERTrace with full reasoning)
        """
        session_id = str(uuid.uuid4())
        start_time = time.time()

        # Handle empty cases
        if not reference.strip() and not hypothesis.strip():
            return self._empty_result(session_id, start_time)

        if not reference.strip():
            return self._no_reference_result(hypothesis, session_id, start_time)

        if not hypothesis.strip():
            return self._no_hypothesis_result(reference, session_id, start_time)

        # Build the user prompt
        user_prompt = f"""Please calculate the Word Error Rate (WER) for this ASR transcription.

**Reference (ground truth):**
{reference}

**Hypothesis (ASR transcription):**
{hypothesis}

Follow the process: NORMALIZE → ALIGN → COUNT → VERIFY → CALCULATE

Show your work clearly, then call calculate_wer with your verified counts."""

        # Initialize conversation
        messages = [{"role": "user", "content": user_prompt}]
        conversation_trace = []
        tool_calls = []
        num_turns = 0
        result = None

        # Multi-turn conversation loop
        max_turns = 10  # Safety limit
        while num_turns < max_turns:
            num_turns += 1

            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SEMANTIC_WER_SYSTEM_PROMPT,
                    tools=[CALCULATE_WER_TOOL],
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"Error calling Claude API: {e}")
                raise

            # Record the assistant's response
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            conversation_trace.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "stop_reason": response.stop_reason,
                }
            )

            # Check if we're done
            if response.stop_reason == "end_turn":
                # Model finished without calling tool - this shouldn't happen
                logger.warning("Model finished without calling calculate_wer")
                break

            # Handle tool use
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use" and block.name == "calculate_wer":
                        # Execute the WER calculation
                        tool_input = block.input
                        tool_calls.append(
                            {
                                "turn": num_turns,
                                "tool_id": block.id,
                                "name": block.name,
                                "input": tool_input,
                            }
                        )

                        result = self._calculate_wer(
                            substitutions=tool_input.get("substitutions", 0),
                            deletions=tool_input.get("deletions", 0),
                            insertions=tool_input.get("insertions", 0),
                            reference_words=tool_input.get("reference_words", 1),
                        )

                        # Add normalized texts and errors to result
                        result["normalized_reference"] = tool_input.get("normalized_reference")
                        result["normalized_hypothesis"] = tool_input.get("normalized_hypothesis")
                        result["errors"] = tool_input.get("errors", [])

                        tool_result = {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                        tool_results.append(tool_result)

                        tool_calls[-1]["output"] = result

                if tool_results:
                    # Add assistant message and tool results
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                    conversation_trace.append(
                        {
                            "role": "user",
                            "content": tool_results,
                        }
                    )

                # If we got a result, we can break after one more turn for final response
                if result is not None:
                    # Get final response after tool result
                    try:
                        final_response = await self.client.messages.create(
                            model=self.model,
                            max_tokens=1024,
                            system=SEMANTIC_WER_SYSTEM_PROMPT,
                            tools=[CALCULATE_WER_TOOL],
                            messages=messages,
                        )

                        final_content = []
                        for block in final_response.content:
                            if block.type == "text":
                                final_content.append({"type": "text", "text": block.text})

                        conversation_trace.append(
                            {
                                "role": "assistant",
                                "content": final_content,
                                "stop_reason": final_response.stop_reason,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error getting final response: {e}")

                    break

        duration_ms = int((time.time() - start_time) * 1000)

        # Convert errors to SemanticError objects
        errors = None
        if result and result.get("errors"):
            errors = [
                SemanticError(
                    error_type=e.get("type", "substitution"),
                    reference_word=e.get("reference"),
                    hypothesis_word=e.get("hypothesis"),
                    position=e.get("position"),
                )
                for e in result["errors"]
            ]

        # Build trace object
        trace = SemanticWERTrace(
            sample_id="",  # Will be set by caller
            service_name=ServiceName.DEEPGRAM,  # Will be set by caller
            session_id=session_id,
            conversation_trace=conversation_trace,
            tool_calls=tool_calls,
            normalized_reference=result.get("normalized_reference") if result else None,
            normalized_hypothesis=result.get("normalized_hypothesis") if result else None,
            wer=result["wer"] if result else 0.0,
            substitutions=result["substitutions"] if result else 0,
            deletions=result["deletions"] if result else 0,
            insertions=result["insertions"] if result else 0,
            reference_words=result["reference_words"] if result else 0,
            errors=errors,
            duration_ms=duration_ms,
            num_turns=num_turns,
            model_used=self.model,
        )

        return result or {"wer": 0.0}, trace

    def _empty_result(self, session_id: str, start_time: float) -> tuple[dict, SemanticWERTrace]:
        """Handle case where both texts are empty."""
        result = {
            "wer": 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 0,
        }
        trace = SemanticWERTrace(
            sample_id="",
            service_name=ServiceName.DEEPGRAM,
            session_id=session_id,
            conversation_trace=[],
            tool_calls=[],
            wer=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_words=0,
            duration_ms=int((time.time() - start_time) * 1000),
            num_turns=0,
            model_used=self.model,
        )
        return result, trace

    def _no_reference_result(
        self, hypothesis: str, session_id: str, start_time: float
    ) -> tuple[dict, SemanticWERTrace]:
        """Handle case where reference is empty."""
        words = len(hypothesis.split())
        result = {
            "wer": float("inf"),
            "substitutions": 0,
            "deletions": 0,
            "insertions": words,
            "reference_words": 0,
        }
        trace = SemanticWERTrace(
            sample_id="",
            service_name=ServiceName.DEEPGRAM,
            session_id=session_id,
            conversation_trace=[],
            tool_calls=[],
            wer=float("inf"),
            substitutions=0,
            deletions=0,
            insertions=words,
            reference_words=0,
            duration_ms=int((time.time() - start_time) * 1000),
            num_turns=0,
            model_used=self.model,
        )
        return result, trace

    def _no_hypothesis_result(
        self, reference: str, session_id: str, start_time: float
    ) -> tuple[dict, SemanticWERTrace]:
        """Handle case where hypothesis is empty."""
        words = len(reference.split())
        result = {
            "wer": 1.0,
            "substitutions": 0,
            "deletions": words,
            "insertions": 0,
            "reference_words": words,
        }
        trace = SemanticWERTrace(
            sample_id="",
            service_name=ServiceName.DEEPGRAM,
            session_id=session_id,
            conversation_trace=[],
            tool_calls=[],
            wer=1.0,
            substitutions=0,
            deletions=words,
            insertions=0,
            reference_words=words,
            duration_ms=int((time.time() - start_time) * 1000),
            num_turns=0,
            model_used=self.model,
        )
        return result, trace

    async def evaluate_service(
        self,
        service_name: ServiceName,
        model_name: str | None = None,
        progress_callback: Callable | None = None,
    ) -> list[WERMetrics]:
        """Evaluate all transcriptions for a service.

        Fetches ground truth and transcriptions from the database,
        computes semantic WER, and stores the results.

        Args:
            service_name: Service to evaluate
            model_name: Optional model name filter
            progress_callback: Optional callback(current, total, sample_id)

        Returns:
            List of WERMetrics
        """
        await self.db.initialize()

        # Get samples that need WER calculation
        samples = await self.db.get_samples_without_wer(service_name, model_name)
        if not samples:
            logger.info(f"All samples already have WER metrics for {service_name.value}")
            return []

        results = []

        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, len(samples), sample.sample_id)

            # Get result and ground truth
            result, gt = await self.db.get_result_with_ground_truth(
                sample.sample_id, service_name, model_name
            )

            if not result or not result.transcription:
                logger.warning(f"No transcription for sample {sample.sample_id}")
                continue

            if not gt:
                logger.warning(f"No ground truth for sample {sample.sample_id}")
                continue

            # Evaluate with Claude
            try:
                eval_result, trace = await self.evaluate(gt.text, result.transcription)

                # Update trace with sample info
                trace.sample_id = sample.sample_id
                trace.service_name = service_name
                trace.model_name = model_name

                # Store the trace
                await self.db.insert_semantic_wer_trace(trace)

                # Create metrics
                metrics = WERMetrics(
                    sample_id=sample.sample_id,
                    service_name=service_name,
                    model_name=model_name,
                    wer=eval_result["wer"],
                    substitutions=eval_result["substitutions"],
                    deletions=eval_result["deletions"],
                    insertions=eval_result["insertions"],
                    reference_words=eval_result["reference_words"],
                    errors=trace.errors,
                    normalized_reference=eval_result.get("normalized_reference"),
                    normalized_hypothesis=eval_result.get("normalized_hypothesis"),
                    timestamp=datetime.now(timezone.utc),
                )

                # Store metrics
                await self.db.insert_wer_metrics(metrics)
                results.append(metrics)

                logger.debug(f"[{i + 1}/{len(samples)}] {sample.sample_id}: WER={metrics.wer:.2%}")

            except Exception as e:
                logger.error(f"Error evaluating {sample.sample_id}: {e}")
                continue

        return results

    async def compute_pooled_wer(self, service_name: ServiceName) -> dict:
        """Compute pooled WER (sum of errors / sum of reference words).

        Args:
            service_name: Service to compute pooled WER for

        Returns:
            Dict with pooled WER metrics
        """
        await self.db.initialize()

        all_metrics = await self.db.get_wer_metrics_for_service(service_name)

        if not all_metrics:
            return {}

        # Filter to valid metrics
        valid = [m for m in all_metrics if m.wer < float("inf")]

        if not valid:
            return {}

        # Sum errors and reference words
        total_substitutions = sum(m.substitutions for m in valid)
        total_deletions = sum(m.deletions for m in valid)
        total_insertions = sum(m.insertions for m in valid)
        total_reference_words = sum(m.reference_words for m in valid)

        if total_reference_words == 0:
            return {}

        pooled_wer = (
            total_substitutions + total_deletions + total_insertions
        ) / total_reference_words

        return {
            "pooled_wer": pooled_wer,
            "total_substitutions": total_substitutions,
            "total_deletions": total_deletions,
            "total_insertions": total_insertions,
            "total_reference_words": total_reference_words,
            "total_errors": total_substitutions + total_deletions + total_insertions,
            "num_samples": len(valid),
        }

    async def close(self) -> None:
        """Close database connection."""
        await self.db.close()
