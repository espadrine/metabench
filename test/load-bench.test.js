const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const { isUnambiguousModelMatch, normalizeModelName } = require('../lib/load-bench');

// Mock data for testing
function createLMArenaModel(name) {
  return { name };
}

function createOurModel(name) {
  return {
    name,
    company: 'Test Company',
    url: 'https://example.com',
    release_date: '2023-01-01',
    capabilities: { input: [], output: [] }
  };
}

const KNOWN_MODEL_MAPPINGS = {
  "gpt-4": "GPT-4",
  "gpt-4-turbo": "GPT-4 Turbo",
  "claude-3-opus": "Claude 3 Opus",
  "gemini-1.5-pro": "Gemini 1.5 Pro"
};

describe('normalizeModelName()', () => {
  test('Normalize - different case', () => {
    const result = normalizeModelName('Custom-Model');
    assert.strictEqual(result, 'custom model');
  });

  test('Normalize - spaces vs dashes', () => {
    const result = normalizeModelName('my-ai-model');
    assert.strictEqual(result, 'my ai model');
  });

  test('Normalize - special characters', () => {
    const result = normalizeModelName('model_v2.0');
    assert.strictEqual(result, 'model v2 0');
  });

  test('Normalize - identical strings', () => {
    const result = normalizeModelName('exact-match');
    assert.strictEqual(result, 'exact match');
  });

  test('Normalize - removes dates', () => {
    const result = normalizeModelName('model-20240115');
    assert.strictEqual(result, 'model');
  });

  test('Normalize - complex punctuation', () => {
    const result = normalizeModelName('ai-model_v3.1-beta');
    assert.strictEqual(result, 'ai model v3 1 beta');
  });
});

describe('isUnambiguousModelMatch() - Exact Matches', () => {
  test('Exact match - different case', () => {
    const lmarenaModel = createLMArenaModel('custom-model');
    const ourModel = createOurModel('Custom-Model');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Exact match - spaces vs dashes', () => {
    const lmarenaModel = createLMArenaModel('my-ai-model');
    const ourModel = createOurModel('My AI Model');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Exact match - special characters normalized', () => {
    const lmarenaModel = createLMArenaModel('model_v2.0');
    const ourModel = createOurModel('Model V2.0');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Exact match - identical strings', () => {
    const lmarenaModel = createLMArenaModel('exact-match');
    const ourModel = createOurModel('exact-match');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });
});

describe('isUnambiguousModelMatch() - Known Mappings', () => {
  test('Known mapping - gpt-4', () => {
    const lmarenaModel = createLMArenaModel('gpt-4');
    const ourModel = createOurModel('GPT-4');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, KNOWN_MODEL_MAPPINGS);
    assert.strictEqual(result, true);
  });

  test('Known mapping - gpt-4-turbo', () => {
    const lmarenaModel = createLMArenaModel('gpt-4-turbo');
    const ourModel = createOurModel('GPT-4 Turbo');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, KNOWN_MODEL_MAPPINGS);
    assert.strictEqual(result, true);
  });

  test('Known mapping - claude-3-opus', () => {
    const lmarenaModel = createLMArenaModel('claude-3-opus');
    const ourModel = createOurModel('Claude 3 Opus');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, KNOWN_MODEL_MAPPINGS);
    assert.strictEqual(result, true);
  });

  test('Known mapping - gemini-1.5-pro', () => {
    const lmarenaModel = createLMArenaModel('gemini-1.5-pro');
    const ourModel = createOurModel('Gemini 1.5 Pro');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, KNOWN_MODEL_MAPPINGS);
    assert.strictEqual(result, true);
  });
});

describe('isUnambiguousModelMatch() - No Data Model', () => {
  test('No data model - null', () => {
    const lmarenaModel = createLMArenaModel('test-model');
    const result = isUnambiguousModelMatch(lmarenaModel, null, {});
    assert.strictEqual(result, false);
  });

  test('No data model - undefined', () => {
    const lmarenaModel = createLMArenaModel('test-model');
    const result = isUnambiguousModelMatch(lmarenaModel, undefined, {});
    assert.strictEqual(result, false);
  });
});

describe('isUnambiguousModelMatch() - Ambiguous Cases', () => {
  test('Ambiguous - completely different names', () => {
    const lmarenaModel = createLMArenaModel('model-a');
    const ourModel = createOurModel('Completely Different Model');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, false);
  });

  test('Ambiguous - similar but not exact', () => {
    const lmarenaModel = createLMArenaModel('model-v1');
    const ourModel = createOurModel('Model V2');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, false);
  });

  test('Ambiguous - different versions', () => {
    const lmarenaModel = createLMArenaModel('model-1.0');
    const ourModel = createOurModel('Model 2.0');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, false);
  });

  test('Ambiguous - partial match', () => {
    const lmarenaModel = createLMArenaModel('ai-model');
    const ourModel = createOurModel('AI Model Pro');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, false);
  });
});

describe('isUnambiguousModelMatch() - Edge Cases', () => {
  test('Edge case - empty strings', () => {
    const lmarenaModel = createLMArenaModel('');
    const ourModel = createOurModel('');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Edge case - single character', () => {
    const lmarenaModel = createLMArenaModel('a');
    const ourModel = createOurModel('A');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Edge case - numbers only', () => {
    const lmarenaModel = createLMArenaModel('123');
    const ourModel = createOurModel('123');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Edge case - special characters only', () => {
    const lmarenaModel = createLMArenaModel('!@#');
    const ourModel = createOurModel('!@#');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });
});

describe('isUnambiguousModelMatch() - Normalization', () => {
  test('Normalization - case insensitive', () => {
    const lmarenaModel = createLMArenaModel('TEST-MODEL');
    const ourModel = createOurModel('test model');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Normalization - alphanumeric only', () => {
    const lmarenaModel = createLMArenaModel('model_v2.0-beta');
    const ourModel = createOurModel('Model V2 0 Beta');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Normalization - complex punctuation', () => {
    const lmarenaModel = createLMArenaModel('ai-model_v3.1');
    const ourModel = createOurModel('AI Model V3 1');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });
});

describe('isUnambiguousModelMatch() - Boundary Cases', () => {
  test('Boundary - very long names', () => {
    const longName = 'very-long-model-name-with-many-characters-and-numbers-123';
    const lmarenaModel = createLMArenaModel(longName);
    const ourModel = createOurModel('Very Long Model Name With Many Characters And Numbers 123');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Boundary - unicode characters', () => {
    const lmarenaModel = createLMArenaModel('model-café');
    const ourModel = createOurModel('Model Café');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });

  test('Boundary - mixed scripts', () => {
    const lmarenaModel = createLMArenaModel('model-日本語');
    const ourModel = createOurModel('Model 日本語');
    const result = isUnambiguousModelMatch(lmarenaModel, ourModel, {});
    assert.strictEqual(result, true);
  });
});