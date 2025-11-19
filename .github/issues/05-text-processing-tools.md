# Add Text Processing Tools to MCP Server

## Overview
Implement text analysis and manipulation tools.

## Tools to Implement

### 1. text_stats
Comprehensive text statistics.
- **Input**: `text` (string, up to 100KB)
- **Output**: Object with:
  - Character count (with/without spaces)
  - Word count
  - Sentence count
  - Paragraph count
  - Average word length
  - Average sentence length
  - Reading time estimate

### 2. word_frequency
Analyze word frequency in text.
- **Input**: `text` (string), `top_n` (optional, default 10)
- **Output**: List of [word, count] pairs, sorted by frequency
- **Features**: Case-insensitive, remove punctuation, skip common words (optional)

### 3. text_transform
Transform text in various ways.
- **Input**: `text` (string), `operation` (string)
- **Operations**:
  - `uppercase` / `lowercase` / `titlecase` / `camelcase` / `snakecase`
  - `reverse` - reverse the text
  - `words_reverse` - reverse word order
  - `remove_spaces` / `remove_punctuation`
- **Output**: Transformed text

### 4. encode_decode
Encode/decode text in various formats.
- **Input**: `text` (string), `operation` (encode/decode), `format` (base64/hex/url)
- **Output**: Encoded or decoded text
- **Formats**:
  - Base64 encoding/decoding
  - Hexadecimal encoding/decoding
  - URL encoding/decoding

## Implementation Requirements
- Handle Unicode text properly
- Add input size limits to prevent memory issues
- Include regex patterns where appropriate
- Provide clear operation descriptions
- Add comprehensive docstrings with examples
- Update list_tools() and call_tool() handlers

## Acceptance Criteria
- [ ] All 4 text processing tools implemented
- [ ] Unicode text handled correctly
- [ ] Full code annotations
- [ ] Input validation with size limits
- [ ] Works correctly in Claude Desktop
- [ ] README documentation with examples

## Example Usage
```
User: "Analyze this text: 'Hello world! This is a test.'"
Tool: text_stats
Result: {
  "characters": 30,
  "words": 6,
  "sentences": 2,
  "avg_word_length": 4.2,
  "reading_time": "1 second"
}

User: "Convert 'Hello World' to base64"
Tool: encode_decode with operation=encode, format=base64
Result: "SGVsbG8gV29ybGQ="

User: "What are the most common words in this text?"
Tool: word_frequency with top_n=5
Result: [["the", 45], ["is", 32], ["and", 28], ["to", 24], ["a", 20]]
```

## References
- [Text Analysis](https://en.wikipedia.org/wiki/Text_mining)
- [Base64](https://en.wikipedia.org/wiki/Base64)
- [URL Encoding](https://en.wikipedia.org/wiki/Percent-encoding)

## Labels
enhancement, text-processing, utilities
