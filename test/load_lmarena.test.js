const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const { isUnambiguousMatch } = require('../bin/load_lmarena');

// Mock data for testing
function createMatch(lmarenaName, ourModelName, hasDataModel = true) {
  return {
    lmarenaModel: {
      name: lmarenaName,
      rating: 1200,
      metadata: {}
    },
    dataModel: hasDataModel ? {
      name: ourModelName,
      company: 'Test Company',
      url: 'https://example.com',
      release_date: '2023-01-01',
      capabilities: { input: [], output: [] }
    } : null,
    benchmark: {
      name: 'LMArena Text',
      score: 1200,
      source: 'https://lmarena.ai/'
    },
    needsBenchmark: true
  };
}

describe('isUnambiguousMatch() - Exact Matches', () => {
  test('Exact match - different case', () => {
    const match = createMatch('custom-model', 'Custom-Model');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Exact match - spaces vs dashes', () => {
    const match = createMatch('my-ai-model', 'My AI Model');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Exact match - special characters normalized', () => {
    const match = createMatch('model_v2.0', 'Model V2.0');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Exact match - identical strings', () => {
    const match = createMatch('exact-match', 'exact-match');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });
});

describe('isUnambiguousMatch() - Known Mappings', () => {
  test('Known mapping - gpt-4', () => {
    const match = createMatch('gpt-4', 'GPT-4');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Known mapping - gpt-4-turbo', () => {
    const match = createMatch('gpt-4-turbo', 'GPT-4 Turbo');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Known mapping - claude-3-opus', () => {
    const match = createMatch('claude-3-opus', 'Claude 3 Opus');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Known mapping - gemini-1.5-pro', () => {
    const match = createMatch('gemini-1.5-pro', 'Gemini 1.5 Pro');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });
});

describe('isUnambiguousMatch() - No Data Model', () => {
  test('No data model - null', () => {
    const match = createMatch('test-model', 'Test Model', false);
    assert.strictEqual(isUnambiguousMatch(match), false);
  });

  test('No data model - unknown model', () => {
    const match = createMatch('unknown-model', 'Unknown Model', false);
    assert.strictEqual(isUnambiguousMatch(match), false);
  });
});

describe('isUnambiguousMatch() - Ambiguous Cases', () => {
  test('Ambiguous - completely different names', () => {
    const match = createMatch('model-a', 'Completely Different Model');
    assert.strictEqual(isUnambiguousMatch(match), false);
  });

  test('Ambiguous - similar but not exact', () => {
    const match = createMatch('model-v1', 'Model V2');
    assert.strictEqual(isUnambiguousMatch(match), false);
  });

  test('Ambiguous - different versions', () => {
    const match = createMatch('model-1.0', 'Model 2.0');
    assert.strictEqual(isUnambiguousMatch(match), false);
  });

  test('Ambiguous - partial match', () => {
    const match = createMatch('ai-model', 'AI Model Pro');
    assert.strictEqual(isUnambiguousMatch(match), false);
  });
});

describe('isUnambiguousMatch() - Edge Cases', () => {
  test('Edge case - empty strings', () => {
    const match = createMatch('', '');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Edge case - single character', () => {
    const match = createMatch('a', 'A');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Edge case - numbers only', () => {
    const match = createMatch('123', '123');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Edge case - special characters only', () => {
    const match = createMatch('!@#', '!@#');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });
});

describe('isUnambiguousMatch() - Normalization', () => {
  test('Normalization - case insensitive', () => {
    const match = createMatch('TEST-MODEL', 'test model');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Normalization - alphanumeric only', () => {
    const match = createMatch('model_v2.0-beta', 'Model V2 0 Beta');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Normalization - complex punctuation', () => {
    const match = createMatch('ai-model_v3.1', 'AI Model V3 1');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });
});

describe('isUnambiguousMatch() - Boundary Cases', () => {
  test('Boundary - very long names', () => {
    const longName = 'very-long-model-name-with-many-characters-and-numbers-123';
    const match = createMatch(longName, 'Very Long Model Name With Many Characters And Numbers 123');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Boundary - unicode characters', () => {
    const match = createMatch('model-café', 'Model Café');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Boundary - mixed scripts', () => {
    const match = createMatch('model-日本語', 'Model 日本語');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });
});