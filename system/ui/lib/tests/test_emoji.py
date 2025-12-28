"""Tests for system/ui/lib/emoji.py - emoji detection."""
import unittest
import re

from openpilot.system.ui.lib.emoji import EMOJI_REGEX, find_emoji


class TestEmojiRegex(unittest.TestCase):
  """Test EMOJI_REGEX pattern."""

  def test_regex_is_compiled(self):
    """Test EMOJI_REGEX is a compiled pattern."""
    self.assertIsInstance(EMOJI_REGEX, re.Pattern)

  def test_matches_simple_emoji(self):
    """Test regex matches simple emoji."""
    match = EMOJI_REGEX.search("\U0001F600")  # grinning face
    self.assertIsNotNone(match)

  def test_matches_smileys(self):
    """Test regex matches smiley emojis (1F600-1F64F)."""
    emojis = ["\U0001F600", "\U0001F603", "\U0001F609", "\U0001F64F"]
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      self.assertIsNotNone(match, f"Failed to match {repr(emoji)}")

  def test_matches_symbols(self):
    """Test regex matches symbol emojis (1F300-1F5FF)."""
    emojis = ["\U0001F300", "\U0001F4A9", "\U0001F5FF"]
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      self.assertIsNotNone(match, f"Failed to match {repr(emoji)}")

  def test_matches_transport(self):
    """Test regex matches transport emojis (1F680-1F6FF)."""
    emojis = ["\U0001F680", "\U0001F697", "\U0001F6FF"]
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      self.assertIsNotNone(match, f"Failed to match {repr(emoji)}")

  def test_matches_flags(self):
    """Test regex matches flag emojis (1F1E0-1F1FF)."""
    emojis = ["\U0001F1E6", "\U0001F1FA", "\U0001F1F8"]  # A, U, S flag components
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      self.assertIsNotNone(match, f"Failed to match {repr(emoji)}")

  def test_matches_miscellaneous_symbols(self):
    """Test regex matches misc symbols (2600-26FF)."""
    emojis = ["\u2600", "\u2615", "\u26A0"]  # sun, hot beverage, warning
    for emoji in emojis:
      match = EMOJI_REGEX.search(emoji)
      self.assertIsNotNone(match, f"Failed to match {repr(emoji)}")

  def test_does_not_match_plain_text(self):
    """Test regex does not match plain text."""
    match = EMOJI_REGEX.search("Hello World")
    self.assertIsNone(match)

  def test_does_not_match_numbers(self):
    """Test regex does not match numbers."""
    match = EMOJI_REGEX.search("12345")
    self.assertIsNone(match)

  def test_does_not_match_ascii_punctuation(self):
    """Test regex does not match ASCII punctuation."""
    match = EMOJI_REGEX.search("!@#$%^&*()")
    self.assertIsNone(match)


class TestFindEmoji(unittest.TestCase):
  """Test find_emoji function."""

  def test_empty_string(self):
    """Test find_emoji on empty string."""
    result = find_emoji("")
    self.assertEqual(result, [])

  def test_no_emoji(self):
    """Test find_emoji with no emoji."""
    result = find_emoji("Hello World")
    self.assertEqual(result, [])

  def test_single_emoji(self):
    """Test find_emoji with single emoji."""
    result = find_emoji("Hello \U0001F600 World")

    self.assertEqual(len(result), 1)
    start, end, emoji = result[0]
    self.assertEqual(emoji, "\U0001F600")
    self.assertEqual(start, 6)
    self.assertEqual(end, 7)

  def test_multiple_emojis(self):
    """Test find_emoji with multiple emojis."""
    result = find_emoji("\U0001F600 and \U0001F604")

    self.assertEqual(len(result), 2)
    self.assertEqual(result[0][2], "\U0001F600")
    self.assertEqual(result[1][2], "\U0001F604")

  def test_emoji_at_start(self):
    """Test find_emoji with emoji at start."""
    result = find_emoji("\U0001F600 Hello")

    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], 0)  # starts at position 0

  def test_emoji_at_end(self):
    """Test find_emoji with emoji at end."""
    text = "Hello \U0001F600"
    result = find_emoji(text)

    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][1], len(text))  # ends at string end

  def test_consecutive_emojis(self):
    """Test find_emoji with consecutive emojis."""
    result = find_emoji("\U0001F600\U0001F604\U0001F60A")

    # May be matched as one or multiple depending on regex
    self.assertGreater(len(result), 0)
    # Check that all characters are covered
    total_chars = sum(end - start for start, end, _ in result)
    self.assertEqual(total_chars, 3)

  def test_returns_tuples(self):
    """Test find_emoji returns list of tuples."""
    result = find_emoji("\U0001F600")

    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[0], tuple)
    self.assertEqual(len(result[0]), 3)

  def test_positions_are_correct(self):
    """Test find_emoji returns correct positions."""
    text = "A\U0001F600B"
    result = find_emoji(text)

    self.assertEqual(len(result), 1)
    start, end, emoji = result[0]
    self.assertEqual(text[start:end], emoji)


if __name__ == '__main__':
  unittest.main()
