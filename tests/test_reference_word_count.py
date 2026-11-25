"""Tests for how semantic WER counts reference words and the judge's errors."""

import pytest

from stt_benchmark.evaluation.semantic_wer import (
    comparison_text,
    count_reference_words,
    error_weight,
    invented_insertions,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Book a table for four", 5),
        # Punctuation, including standalone marks and curly quotes, adds no words.
        ("Wait - really? Yes… “fine”.", 4),
        # Contractions count as the words they expand to.
        ("I'm sure it doesn't, and we'll see", 10),
        ("Let's go; I'd, you'll, we've, they're won't", 13),
        ("It’s here", 3),
        # "'s" after other words is a possessive.
        ("My driver's license and the neighbors' dog", 7),
        # Hyphenated compounds and numbers as written are one word each.
        ("My home Wi-Fi at 7:30 costs 2,000 dollars", 8),
        # Words in any script count.
        ("ᱢᱟᱭ ᱞᱟᱴᱟᱯ", 2),
        ("", 0),
    ],
)
def test_count_reference_words(text, expected):
    assert count_reference_words(text) == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"type": "substitution", "reference": "card", "hypothesis": "car"}, 1),
        # A split or merged word is one substitution.
        ({"type": "substitution", "reference": "backyard", "hypothesis": "back card"}, 1),
        ({"type": "substitution", "reference": "shelving unit", "hypothesis": "shopping"}, 1),
        # Wider spans count every reference word.
        (
            {"type": "substitution", "reference": "loaf of sourdough", "hypothesis": "loaful sort"},
            3,
        ),
        ({"type": "deletion", "reference": "covering elements like the header"}, 5),
        ({"type": "deletion", "reference": None}, 1),
        ({"type": "insertion", "reference": None, "hypothesis": "on dish"}, 2),
    ],
)
def test_error_weight(error, expected):
    assert error_weight(error) == expected


def _insertions(*words):
    return [{"type": "insertion", "reference": None, "hypothesis": w} for w in words]


@pytest.mark.parametrize(
    ("errors", "hypothesis", "expected"),
    [
        # Words from the prompt read as part of the hypothesis.
        (
            _insertions("follow", "the", "process", "normalize", "align", "count"),
            '"Tell me about the Globe Theatre.',
            True,
        ),
        # Normalized forms of words that are in the hypothesis.
        (_insertions("it", "is", "1", "0"), "the number ending in 780? It's 1-0.", False),
        (_insertions("do", "not", "know"), "I don't know", False),
        # A couple of missing words is tolerated.
        (_insertions("um", "uh"), "book a table", False),
    ],
)
def test_invented_insertions(errors, hypothesis, expected):
    assert invented_insertions(errors, hypothesis) is expected


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        # Case and the punctuation around words don't matter.
        ("Hello, world!", "hello world", True),
        ("It’s “fine”… really.", "it's fine really", True),
        ("Wait - really?", "wait really", True),
        # Anything inside a word does.
        ("we're late", "were late", False),
        ("at 7:30", "at 730", False),
        ("my Wi-Fi", "my wifi", False),
        ("e-mail me", "e mail me", False),
        # So does any word, in any script.
        ("send it now", "send it", False),
        ("thanks 幫", "thanks", False),
    ],
)
def test_comparison_text(first, second, same):
    assert (comparison_text(first) == comparison_text(second)) is same
