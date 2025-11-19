# Add Hash Generator Tool to MCP Server

## Overview
Implement cryptographic hash generation for various algorithms.

## Tool to Implement

### generate_hash
Generate cryptographic hashes of text or data.
- **Input**: 
  - `data` (string, up to 1MB)
  - `algorithm` (string: md5, sha1, sha256, sha512, blake2b)
  - `output_format` (optional: hex, base64, default=hex)
- **Output**: Hash string in specified format

## Supported Algorithms

### MD5
- 128-bit hash (32 hex characters)
- Fast but not cryptographically secure
- Good for checksums, not passwords

### SHA-1
- 160-bit hash (40 hex characters)
- Deprecated for security, but still used for git commits
- Not recommended for new applications

### SHA-256
- 256-bit hash (64 hex characters)
- Part of SHA-2 family
- Widely used, secure, good for passwords (with salt)

### SHA-512
- 512-bit hash (128 hex characters)
- More secure than SHA-256
- Slower but more resistant to collision attacks

### BLAKE2b
- Modern, fast alternative to SHA-2
- Comparable security to SHA-3
- Often faster than MD5

## Implementation Requirements
- Use Python's hashlib library
- Add input size validation (prevent huge inputs)
- Support both text strings and file content (base64 encoded)
- Show algorithm characteristics in output
- Include comprehensive docstrings with security notes
- Update list_tools() and call_tool() handlers
- Add logging for hash operations

## Acceptance Criteria
- [ ] All 5 hash algorithms implemented
- [ ] Both hex and base64 output formats work
- [ ] Full code annotations with security notes
- [ ] Input validation and size limits
- [ ] Works correctly in Claude Desktop
- [ ] README documentation with security warnings

## Example Usage
```
User: "Generate an SHA-256 hash of 'Hello World'"
Tool: generate_hash with data="Hello World", algorithm="sha256"
Result: "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"

User: "What's the MD5 hash of 'password123'?"
Tool: generate_hash with data="password123", algorithm="md5"
Result: "482c811da5d5b4bc6d497ffa98491e38"
Note: MD5 is NOT secure for passwords. Use SHA-256 with salt instead.
```

## Security Notes
- **MD5**: Broken, only use for non-security purposes
- **SHA-1**: Deprecated, avoid for new applications
- **SHA-256/512**: Secure, recommended for most uses
- **Passwords**: Always use salt + strong hash (SHA-256 minimum)
- **File Integrity**: SHA-256 or BLAKE2b recommended

## Additional Features (Optional)
- HMAC support (keyed hashing)
- File hash verification
- Hash comparison tool
- Password strength checker

## References
- [Cryptographic Hash Function](https://en.wikipedia.org/wiki/Cryptographic_hash_function)
- [SHA-2](https://en.wikipedia.org/wiki/SHA-2)
- [BLAKE2](https://en.wikipedia.org/wiki/BLAKE_(hash_function)#BLAKE2)
- [Python hashlib](https://docs.python.org/3/library/hashlib.html)

## Labels
enhancement, security, utilities
