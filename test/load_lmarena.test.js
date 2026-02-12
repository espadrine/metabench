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

describe('isUnambiguousMatch() - LMArena Specific Tests', () => {
  test('Returns false when no dataModel', () => {
    const match = createMatch('test-model', 'Test Model', false);
    assert.strictEqual(isUnambiguousMatch(match), false);
  });

  test('Returns false for ambiguous matches', () => {
    const match = createMatch('model-a', 'Completely Different Model');
    assert.strictEqual(isUnambiguousMatch(match), false);
  });

  test('Returns true for exact matches', () => {
    const match = createMatch('exact-model', 'exact-model');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Returns true for known mappings', () => {
    const match = createMatch('gpt-4', 'GPT-4');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });

  test('Returns true for normalized matches', () => {
    const match = createMatch('model_v2.0', 'Model V2.0');
    assert.strictEqual(isUnambiguousMatch(match), true);
  });
});