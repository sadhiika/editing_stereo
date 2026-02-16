# OpenAIJudge Implementation Notes

## Summary of Changes

This document summarizes the implementation of Task 0.1 from technical-preparation.md - replacing the mock OpenAIJudge with a real API integration.

### Changes Made

1. **Updated `biaswipe/judge.py`**:
   - Replaced mock OpenAIJudge implementation with real OpenAI API integration
   - Added OpenAI client initialization in `__init__` method
   - Implemented proper API calls using the OpenAI Python library
   - Added comprehensive error handling for:
     - Missing API keys (raises `MissingApiKeyError`)
     - Authentication failures
     - Rate limiting (with exponential backoff)
     - General API errors (with retry logic)
   - Added JSON response parsing and validation
   - Implemented `_validate_response()` method to ensure response structure matches schema
   - Implemented `_extract_json_from_response()` to handle various response formats
   - Fixed prompt formatting issue by using `replace()` instead of `format()` to avoid issues with JSON examples in the prompt

2. **Updated `requirements.txt`**:
   - Added `openai>=1.23.2` dependency

3. **Fixed Issues**:
   - Fixed forward reference type annotation for `Judge` class
   - Fixed syntax error (removed extra backticks at end of file)
   - Updated cache directory path to `.biaswipe_cache/` to match documentation

### Key Features Implemented

1. **API Integration**:
   - Uses OpenAI's chat completions API with GPT-4 by default
   - Configurable model via constructor parameter
   - Low temperature (0.1) for consistent judgments
   - JSON response format requested via API parameter

2. **Error Handling**:
   - Graceful handling of missing API keys
   - Authentication error detection
   - Rate limit handling with exponential backoff
   - Maximum 3 retries with increasing delays (1s, 2s, 4s)
   - Returns structured error responses instead of raising exceptions

3. **Response Validation**:
   - Validates all required fields are present
   - Checks field types (bool, float, string)
   - Ensures severity_score is in [0.0, 1.0] range
   - Enforces consistency (severity_score = 0.0 when is_stereotype = False)
   - Attempts to fix common issues before rejecting response

4. **Backward Compatibility**:
   - Maintains the same interface as the mock implementation
   - Caching still works as expected
   - Error responses follow the same format
   - Can be used as a drop-in replacement

### Testing

The implementation includes comprehensive error handling that was verified through manual testing:

1. Missing API key correctly raises `MissingApiKeyError`
2. Invalid API keys return authentication error responses
3. Prompt loading works correctly from `judge_prompt.txt`
4. Caching mechanism continues to function properly

### Usage

```python
# With environment variable
os.environ['OPENAI_API_KEY'] = 'your-api-key'
judge = OpenAIJudge()

# With explicit API key
judge = OpenAIJudge(api_key='your-api-key', model='gpt-4')

# Get judgment
result = judge.judge_response("Some text to evaluate")
# Returns: {"is_stereotype": bool, "severity_score": float, "reasoning": str}
# Or on error: {"error": str, "details": str}
```

### Notes

- The implementation requires the `openai` Python package to be installed
- A valid OpenAI API key must be available (via environment variable or parameter)
- The judge prompt is loaded from `judge_prompt.txt` in the project root
- Responses are cached in `.biaswipe_cache/` directory
- The implementation uses the newer OpenAI client library API (v1.x)