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

import asyncio
import hashlib
import inspect
import json
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
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

**1.14 Repetitions and Stutters**: Ignore words or phrases the speaker repeats or restarts
   "I, I guess" = "I guess", "best bestseller" = "bestseller", "we should we should" = "we should"

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
- Repeated words and stutters: "I I guess"→"I guess" = NO

**Patterns that ARE errors (answer YES):**
- Different words: "card"→"car", "trace"→"trade", "hours"→"was" = YES
- Nonsense: "lentil"→"landon", "Wi-Fi"→"wi fire" = YES

### Step 4: COUNT
Count ONLY the differences where you answered "COUNT AS ERROR: YES"
- S = semantic substitutions (different meaning)
- D = semantic deletions (meaning lost)
- I = semantic insertions (meaning added)

**IMPORTANT: A word that is split, merged, or a compound counts as ONE error, not multiple.**
- A hyphenated compound replaced by a single word: "cross-country" → "koscanti" = S=1
- One reference word transcribed as several words: "backyard" → "back card" = S=1, "difficulty" → "diffic ulty" = S=1
- Several reference words transcribed as one word = S=1
Each is ONE substitution, NOT a substitution plus insertions or deletions: the words represent a single concept.

**TRUNCATED/INCOMPLETE TEXT:**
When both reference and hypothesis appear truncated at the same point (missing the end of a sentence), compare only the complete portions. Partial words at truncation points should be ignored rather than counted as errors. If a word is clearly incomplete (like "reme" for "remember" or "abor" for "abroad"), do not count differences involving that truncated word.

**TRAILING FUNCTION WORDS AT TRUNCATION:**
If the reference ends with a function word that signals an incomplete sentence (and, but, or, so, to, for, the, a, an, on, in, with, that, which, who, because, although, if, when, while, as, about, from, by, at, of, etc.) and the hypothesis omits it, do NOT count as an error. These trailing words carry no semantic meaning on their own - an LLM would respond identically with or without them.
- Example: "My sister called me about the birthday party and" vs "My sister called me about the birthday party" = NOT an error (trailing "and" is meaningless)
- Example: "Can you help me brainstorm ideas for my presentation on" vs "Can you help me brainstorm ideas for my presentation" = NOT an error (trailing "on" is meaningless)

### Step 5: CALCULATE
Call calculate_wer with one entry in `errors` for each error you counted, and an empty list if there are none. The WER is computed from this list, so every counted error must appear in it exactly once. List errors word by word: a deleted phrase is one deletion entry per word, and an inserted phrase one insertion entry per word. The only entries that span several words are split, merged, or compound words, each a single substitution entry whose reference and hypothesis hold the whole span.

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
    "description": "Calculate Word Error Rate from the list of semantic errors. Call this ONCE after you have normalized, aligned, and verified the texts. List every counted error as its own entry; an empty list means no errors.",
    "input_schema": {
        "type": "object",
        "properties": {
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
                "description": "Every counted error, one entry each",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["substitution", "deletion", "insertion"],
                        },
                        "reference": {
                            "type": "string",
                            "description": "Reference word or span (null for insertion)",
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "Hypothesis word or span (null for deletion)",
                        },
                        "position": {
                            "type": "integer",
                            "description": "Position in alignment",
                        },
                    },
                    "required": ["type"],
                },
            },
        },
        "required": ["errors"],
    },
}

USER_PROMPT_TEMPLATE = """Please calculate the Word Error Rate (WER) for this ASR transcription.

The reference (ground truth) and the hypothesis (ASR transcription) are exactly the text inside their tags.

<reference>
{reference}
</reference>

<hypothesis>
{hypothesis}
</hypothesis>

Follow the process: NORMALIZE → ALIGN → COUNT → VERIFY → CALCULATE

Show your work clearly, then call calculate_wer with your verified list of errors."""

# Contraction endings that stand for a second word ("I'm" → "i am"), matching
# the prompt's normalization. "'s" is only a contraction after these words;
# elsewhere it is a possessive ("driver's") and adds no word.
_CONTRACTION_ENDINGS = ("n't", "'m", "'re", "'ve", "'ll", "'d")
_IS_CONTRACTIONS = {
    "it",
    "that",
    "what",
    "there",
    "here",
    "he",
    "she",
    "who",
    "where",
    "how",
    "when",
    "why",
    "let",
}


def count_reference_words(text: str) -> int:
    """Count the words in a reference transcription.

    WER divides by this count, so it is computed here rather than by the
    judge: every service is scored against the same count for the same
    reference. Punctuation is dropped, contractions count as the two words
    they expand to, and a hyphenated compound or a number as written
    ("2,000", "7:30") counts as one word.
    """
    count = 0
    for token in text.lower().replace("’", "'").split():
        token = token.strip('.,!?;:"()[]{}…')
        if not any(ch.isalnum() for ch in token):
            continue
        count += 1
        if token.endswith(_CONTRACTION_ENDINGS) or (
            token.endswith("'s") and token[:-2] in _IS_CONTRACTIONS
        ):
            count += 1
    return count


def error_weight(error: dict) -> int:
    """Number of errors one entry of the judge's error list counts for.

    The judge is asked to list errors word by word, except that a split,
    merged, or compound word is one substitution. An entry that spans more
    words than that still counts for each word it covers, so a deleted phrase
    listed as a single entry isn't scored as one error.
    """
    reference_words = count_reference_words(error.get("reference") or "")
    hypothesis_words = count_reference_words(error.get("hypothesis") or "")
    if error["type"] == "deletion":
        return max(reference_words, 1)
    if error["type"] == "insertion":
        return max(hypothesis_words, 1)
    if min(reference_words, hypothesis_words) <= 1:
        return 1
    return reference_words


_WORD_PUNCTUATION = ".,!?;:\"'()[]{}…"
# Words a contraction ending expands to ("it's" → "it is").
_CONTRACTION_WORDS = {
    "'s": ("is", "has"),
    "'m": ("am",),
    "'re": ("are",),
    "'ve": ("have",),
    "'ll": ("will",),
    "'d": ("would", "had"),
    "n't": ("not",),
}


def _hypothesis_vocabulary(hypothesis: str) -> set[str]:
    """Words in the hypothesis, plus the forms the judge may normalize them to."""
    words = set()
    for token in re.split(r"[\s\-]+", hypothesis.lower().replace("’", "'")):
        token = token.strip(_WORD_PUNCTUATION)
        words.add(token)
        for ending, expansions in _CONTRACTION_WORDS.items():
            if token.endswith(ending):
                words.add(token[: -len(ending)])
                words.update(expansions)
    return words


def invented_insertions(errors: list[dict], hypothesis: str) -> bool:
    """Whether the judge's insertions are mostly words the hypothesis doesn't contain.

    Inserted words come from the hypothesis, so a list of them that mostly
    isn't there means the judge misread where the hypothesis ends. Numbers are
    ignored, since the judge may write them as digits or words either way.
    """
    vocabulary = _hypothesis_vocabulary(hypothesis)
    inserted = [
        w.strip(_WORD_PUNCTUATION)
        for e in errors
        if e["type"] == "insertion"
        for w in (e.get("hypothesis") or "").lower().split()
    ]
    missing = [w for w in inserted if w and not w.isdigit() and w not in vocabulary]
    return len(missing) >= 3 and len(missing) > len(inserted) / 2


# The judge configuration. Scores from different judges aren't comparable, so
# every stored result records the judge that produced it (see
# ``SemanticWEREvaluator.judge``) and the ``wer`` command refuses to mix judges
# within a service. Changing any of these requires re-scoring every service.
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
# At "low" the model decides per request whether to think, so easy and hard
# transcripts would be judged differently; from "medium" up it thinks on
# nearly every request.
DEFAULT_JUDGE_EFFORT = "medium"
# Number of independent judgments per sample; the median one is stored.
DEFAULT_JUDGE_REPEATS = 1

# Identifies the prompts, tool definition, and error counting, so edits to any
# of them also count as a different judge.
JUDGE_FINGERPRINT = hashlib.sha256(
    (
        SEMANTIC_WER_SYSTEM_PROMPT
        + USER_PROMPT_TEMPLATE
        + json.dumps(CALCULATE_WER_TOOL, sort_keys=True)
        + inspect.getsource(count_reference_words)
        + inspect.getsource(error_weight)
        + json.dumps([_CONTRACTION_ENDINGS, sorted(_IS_CONTRACTIONS)])
    ).encode()
).hexdigest()[:8]


ERROR_TYPES = ("substitution", "deletion", "insertion")


class JudgeError(Exception):
    """The judge finished without producing a WER result."""


class SemanticWEREvaluator:
    """Semantic WER evaluator using Claude with tool use.

    Adapted from asr_eval AgentSDKJudge.

    ``effort=None`` runs the judge without thinking at temperature 0, the
    setup for models that predate adaptive thinking (such as the Claude
    Sonnet 4.5 judge), so earlier scores can be reproduced for comparison.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        effort: str | None = DEFAULT_JUDGE_EFFORT,
        repeats: int = DEFAULT_JUDGE_REPEATS,
        db_path: Path | None = None,
        max_concurrency: int = 50,
    ):
        self.config = get_config()
        self.model = model
        self.effort = effort
        self.repeats = repeats
        self.db = Database(db_path=db_path)
        self._api_semaphore = asyncio.Semaphore(max_concurrency)
        # Running token totals across every API call this evaluator makes.
        self.usage: Counter[str] = Counter()

        if not self.config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        self.client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)

    @property
    def judge(self) -> str:
        """Label identifying this judge configuration, stored with each result."""
        sampling = f"effort={self.effort}" if self.effort else "temperature=0"
        return f"{self.model} {sampling} repeats={self.repeats} rev={JUDGE_FINGERPRINT}"

    def _request(self, messages: list[dict], max_tokens: int) -> dict:
        """Build a request payload with the judge's model, thinking, and effort settings."""
        if self.effort:
            # "summarized" keeps the model's reasoning readable in the stored trace.
            sampling = {
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": self.effort},
            }
        else:
            sampling = {"temperature": 0}
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            **sampling,
            "system": [
                {
                    "type": "text",
                    "text": SEMANTIC_WER_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": [CALCULATE_WER_TOOL],
            "messages": messages,
        }

    async def warm_cache(self) -> None:
        """Send a minimal request to warm the prompt cache.

        Establishes the cache for the system prompt and tool definition
        before any real evaluation requests are sent.  This ensures all
        concurrent evaluation requests get cache hits.  ``max_tokens=0``
        writes the cache without generating output; the thinking and
        effort settings must match the real requests for them to hit it.
        """
        response = await self.client.messages.create(
            **self._request([{"role": "user", "content": "."}], max_tokens=0)
        )
        logger.info(
            f"Prompt cache warmed ({getattr(response.usage, 'cache_creation_input_tokens', 0) or 0:,} tokens written)"
        )

    def _record_usage(self, response: anthropic.types.Message) -> None:
        usage = response.usage
        self.usage["input_tokens"] += usage.input_tokens
        self.usage["output_tokens"] += usage.output_tokens
        self.usage["cache_read_input_tokens"] += usage.cache_read_input_tokens or 0
        self.usage["cache_creation_input_tokens"] += usage.cache_creation_input_tokens or 0

    async def _api_call(
        self,
        request_payload: dict,
        filename: str,
        max_retries: int = 5,
    ) -> anthropic.types.Message:
        """Make a single API call with retry.

        Retries on rate-limit (429) and server (5xx) errors with
        exponential backoff.
        """
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.messages.create(**request_payload)
                self._record_usage(response)
                return response
            except (
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
            ) as e:
                label = (
                    "rate limited" if isinstance(e, anthropic.RateLimitError) else "server error"
                )
                if attempt == max_retries:
                    raise
                wait = 15 * (2 ** (attempt - 1))
                logger.warning(
                    f"{filename}: {label}, waiting {wait}s (attempt {attempt}/{max_retries})"
                )
            await asyncio.sleep(wait)
        raise RuntimeError("unreachable")

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
        filename: str = "",
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

        user_prompt = USER_PROMPT_TEMPLATE.format(reference=reference, hypothesis=hypothesis)

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
                response = await self._api_call(self._request(messages, max_tokens=16000), filename)
            except Exception as e:
                logger.error(f"Error calling Claude API: {e}")
                raise

            # Record the assistant's response
            assistant_content = []
            for block in response.content:
                if block.type == "thinking" and block.thinking:
                    assistant_content.append({"type": "thinking", "thinking": block.thinking})
                elif block.type == "text":
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

            # Anything but a tool call (finishing without calling calculate_wer,
            # running out of tokens, or a refusal) leaves no result to store.
            if response.stop_reason != "tool_use":
                reason = response.stop_reason
                if reason == "refusal" and response.stop_details:
                    reason = f"refusal ({response.stop_details.category})"
                raise JudgeError(f"judge stopped without a result: {reason}")

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

                        # The counts come from the error list, so they always
                        # match the errors the judge actually listed.
                        errors = tool_input.get("errors")
                        if not isinstance(errors, list) or any(
                            not isinstance(e, dict) or e.get("type") not in ERROR_TYPES
                            for e in errors
                        ):
                            raise JudgeError(f"judge sent an invalid error list: {errors!r}")
                        if invented_insertions(errors, hypothesis):
                            raise JudgeError(
                                "judge listed inserted words that are not in the hypothesis"
                            )
                        counts: Counter[str] = Counter()
                        for e in errors:
                            counts[e["type"]] += error_weight(e)
                        result = self._calculate_wer(
                            substitutions=counts["substitution"],
                            deletions=counts["deletion"],
                            insertions=counts["insertion"],
                            reference_words=count_reference_words(reference),
                        )

                        # Add normalized texts and errors to result
                        result["normalized_reference"] = tool_input.get("normalized_reference")
                        result["normalized_hypothesis"] = tool_input.get("normalized_hypothesis")
                        result["errors"] = errors

                        tool_result = {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                        tool_results.append(tool_result)

                        tool_calls[-1]["output"] = result

                if tool_results:
                    conversation_trace.append(
                        {
                            "role": "user",
                            "content": tool_results,
                        }
                    )

                # The score comes from the tool call, so there is nothing
                # more to ask the judge once it has made one.
                if result is not None:
                    break

        if result is None:
            raise JudgeError(f"judge did not call calculate_wer within {max_turns} turns")

        duration_ms = int((time.time() - start_time) * 1000)

        # Convert errors to SemanticError objects
        errors = None
        if result.get("errors"):
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
            normalized_reference=result.get("normalized_reference"),
            normalized_hypothesis=result.get("normalized_hypothesis"),
            wer=result["wer"],
            substitutions=result["substitutions"],
            deletions=result["deletions"],
            insertions=result["insertions"],
            reference_words=result["reference_words"],
            errors=errors,
            duration_ms=duration_ms,
            num_turns=num_turns,
            model_used=self.model,
        )

        return result, trace

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
        words = count_reference_words(reference)
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

    async def evaluate_with_retry(
        self,
        reference: str,
        hypothesis: str,
        filename: str = "",
        max_retries: int = 3,
        timeout_secs: float = 5 * 60,
    ) -> tuple[dict, SemanticWERTrace] | None:
        """Evaluate with concurrency control, timeout, and retry logic.

        Transient API errors (429, 5xx) are handled per-call inside
        ``_api_call``.  This method retries timeouts and judge runs that
        ended without a result, and gives up on unexpected errors.
        Returns None if all attempts fail.
        """
        for attempt in range(1, max_retries + 1):
            try:
                async with self._api_semaphore:
                    return await asyncio.wait_for(
                        self.evaluate(reference, hypothesis, filename=filename),
                        timeout=timeout_secs,
                    )
            except TimeoutError:
                logger.warning(
                    f"{filename}: timed out after {timeout_secs}s (attempt {attempt}/{max_retries})"
                )
            except JudgeError as e:
                logger.warning(f"{filename}: {e} (attempt {attempt}/{max_retries})")
            except Exception as e:
                logger.error(f"{filename}: evaluation error: {e}")
                return None

        logger.error(f"{filename}: failed after {max_retries} attempts")
        return None

    async def evaluate_repeated(
        self,
        reference: str,
        hypothesis: str,
        filename: str = "",
    ) -> tuple[dict, SemanticWERTrace] | None:
        """Judge a transcription ``self.repeats`` times and return the median judgment.

        The judge doesn't give identical answers on every run, so the run
        with the median WER is kept to damp that noise.  Returns None
        unless every run succeeds, so a stored result always reflects the
        full set of runs.
        """
        pairs = await asyncio.gather(
            *(
                self.evaluate_with_retry(reference, hypothesis, filename=filename)
                for _ in range(self.repeats)
            )
        )
        if any(pair is None for pair in pairs):
            return None
        pairs.sort(key=lambda pair: pair[0]["wer"])
        return pairs[len(pairs) // 2]

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

        results: list[WERMetrics] = []
        completed = 0

        async def _eval_sample(sample):
            nonlocal completed

            # Get result and ground truth
            result, gt = await self.db.get_result_with_ground_truth(
                sample.sample_id, service_name, model_name
            )

            if not result or not result.transcription:
                logger.warning(f"No transcription for sample {sample.sample_id}")
                return

            if not gt:
                logger.warning(f"No ground truth for sample {sample.sample_id}")
                return

            # Evaluate with Claude (with retry and timeout)
            eval_pair = await self.evaluate_repeated(
                gt.text, result.transcription, filename=sample.sample_id
            )

            if eval_pair is None:
                logger.error(f"Failed to evaluate {sample.sample_id}, skipping")
                return

            eval_result, trace = eval_pair

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
                judge=self.judge,
                timestamp=datetime.now(UTC),
            )

            # Store metrics
            await self.db.insert_wer_metrics(metrics)
            results.append(metrics)

            completed += 1
            if progress_callback:
                progress_callback(completed, len(samples), sample.sample_id)

            logger.debug(f"[{completed}/{len(samples)}] {sample.sample_id}: WER={metrics.wer:.2%}")

        await asyncio.gather(*(_eval_sample(sample) for sample in samples))

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
