"""Tests for system/ui/lib/emoji.py - emoji detection."""

import re

from openpilot.system.ui.lib.emoji import EMOJI_REGEX, find_emoji


class TestEmojiRegex:
  """Test EMOJI_REGEX pattern."""

  def test_regex_is_compiled(self):
    """Test EMOJI_REGEX is a compiled pattern."""
    assert isinstance(EMOJI_REGEX, re.Pattern)

  def test_matches_simple_emoji(self):
    """Test regex matches simple emoji."""
    match = EMOJI_REGEX.search("\U0001f600")  # grinning face
    assert match is not None

  def test_matches_smileys(self):
    """Test regex matches smiley emojis (1F600-1F64F)."""
    emojis = ["\U0001f600", "\U0001f603", "\U0001f609", "\U0001f64f"]
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      assert match is not None, f"Failed to match {repr(emoji)}"

  def test_matches_symbols(self):
    """Test regex matches symbol emojis (1F300-1F5FF)."""
    emojis = ["\U0001f300", "\U0001f4a9", "\U0001f5ff"]
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      assert match is not None, f"Failed to match {repr(emoji)}"

  def test_matches_transport(self):
    """Test regex matches transport emojis (1F680-1F6FF)."""
    emojis = ["\U0001f680", "\U0001f697", "\U0001f6ff"]
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      assert match is not None, f"Failed to match {repr(emoji)}"

  def test_matches_flags(self):
    """Test regex matches flag emojis (1F1E0-1F1FF)."""
    emojis = ["\U0001f1e6", "\U0001f1fa", "\U0001f1f8"]  # A, U, S flag components
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      assert match is not None, f"Failed to match {repr(emoji)}"

  def test_matches_miscellaneous_symbols(self):
    """Test regex matches misc symbols (2600-26FF)."""
    emojis = ["\u2600", "\u2615", "\u26a0"]  # sun, hot beverage, warning
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      assert match is not None, f"Failed to match {repr(emoji)}"

  def test_does_not_match_plain_text(self):
    """Test regex does not match plain text."""
    match = EMOJI_REGEX.search("Hello World")
    assert match is None

  def test_does_not_match_numbers(self):
    """Test regex does not match numbers."""
    match = EMOJI_REGEX.search("12345")
    assert match is None

  def test_does_not_match_ascii_punctuation(self):
    """Test regex does not match ASCII punctuation."""
    match = EMOJI_REGEX.search("!@#$%^&*()")
    assert match is None


class TestFindEmoji:
  """Test find_emoji function."""

  def test_empty_string(self):
    """Test find_emoji on empty string."""
    result = find_emoji("")
    assert result == []

  def test_no_emoji(self):
    """Test find_emoji with no emoji."""
    result = find_emoji("Hello World")
    assert result == []

  def test_single_emoji(self):
    """Test find_emoji with single emoji."""
    result = find_emoji("Hello \U0001f600 World")

    assert len(result) == 1
    start, end, emoji = result[0]
    assert emoji == "\U0001f600"
    assert start == 6
    assert end == 7

  def test_multiple_emojis(self):
    """Test find_emoji with multiple emojis."""
    result = find_emoji("\U0001f600 and \U0001f604")

    assert len(result) == 2
    assert result[0][2] == "\U0001f600"
    assert result[1][2] == "\U0001f604"

  def test_emoji_at_start(self):
    """Test find_emoji with emoji at start."""
    result = find_emoji("\U0001f600 Hello")

    assert len(result) == 1
    assert result[0][0] == 0  # starts at position 0

  def test_emoji_at_end(self):
    """Test find_emoji with emoji at end."""
    text = "Hello \U0001f600"
    result = find_emoji(text)

    assert len(result) == 1
    assert result[0][1] == len(text)  # ends at string end

  def test_consecutive_emojis(self):
    """Test find_emoji with consecutive emojis."""
    result = find_emoji("\U0001f600\U0001f604\U0001f60a")

    # May be matched as one or multiple depending on regex
    assert len(result) > 0
    # Check that all characters are covered
    total_chars = sum(end - start for start, end, _ in result)
    assert total_chars == 3

  def test_returns_tuples(self):
    """Test find_emoji returns list of tuples."""
    result = find_emoji("\U0001f600")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert len(result[0]) == 3

  def test_positions_are_correct(self):
    """Test find_emoji returns correct positions."""
    text = "A\U0001f600B"
    result = find_emoji(text)

    assert len(result) == 1
    start, end, emoji = result[0]
    assert text[start:end] == emoji
