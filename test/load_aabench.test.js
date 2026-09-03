const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

describe('modelNameFromAA', () => {
  test('all keys are in data/aabench-data.json at .models[*].name', () => {
    const aabenchPath = path.join(__dirname, '../data/aabench-data.json');
    const aabench = JSON.parse(fs.readFileSync(aabenchPath, 'utf8'));
    const aaNames = new Set(aabench.models.map((d) => d.name));

    // Load modelNameFromAA from bin/load_aabench.js
    const { modelNameFromAA } = require('../bin/load_aabench');

    const missing = Object.keys(modelNameFromAA).filter((k) => !aaNames.has(k));

    assert.deepStrictEqual(
      missing,
      [],
      `modelNameFromAA keys missing in data/aabench-data.json: ${missing.join(', ')}`
    );
  });
});
