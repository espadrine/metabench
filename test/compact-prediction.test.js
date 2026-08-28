'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { encodePredictions, decodePredictions, round3, buildBenchmarks, buildSources, buildCompanies, buildUrls, buildCapabilities } = require('../lib/compact-prediction');

test('round3 keeps 3 decimals', () => {
  assert.strictEqual(round3(1.23456), 1.235);
  assert.strictEqual(round3(1.2344), 1.234);
  assert.strictEqual(round3(32), 32);
  assert.strictEqual(round3(0.5444722215), 0.544);
});

test('helpers build sorted intern tables', () => {
  const models = [
    { name: 'm2', company: 'B', url: 'u2', release_date: '2024-01-01', capabilities: { input: ['text'], output: ['text'] }, benchmarks: [{ name: 'Z', score: 1, source: 's2', stdDev: 0, isGuessed: false }] },
    { name: 'm1', company: 'A', url: 'u1', release_date: '2023-01-01', capabilities: { input: ['image'], output: ['tool'] }, benchmarks: [{ name: 'A', score: 2, source: 's1', stdDev: 0, isGuessed: true }] },
  ];
  assert.deepStrictEqual(buildBenchmarks(models), ['A', 'Z']);
  assert.deepStrictEqual(buildSources(models).sources, ['s1', 's2']);
  assert.deepStrictEqual(buildCompanies(models).companies, ['A', 'B']);
  assert.deepStrictEqual(buildUrls(models).urls, ['u1', 'u2']);
  assert.deepStrictEqual(buildCapabilities(models).capabilities, ['image', 'text', 'tool']);
});

test('encode/decode minimal JSON roundtrip', () => {
  const input = {
    models: [
      {
        name: 'Alpha',
        company: 'OpenAI',
        url: 'https://example.com/alpha',
        release_date: '2024-06-01',
        capabilities: { input: ['text'], output: ['text', 'tool'] },
        benchmarks: [
          { name: 'MMLU', score: 87.12345, source: 'https://example.com', stdDev: 0.123456, isGuessed: false },
          { name: 'GSM8K', score: 92, source: 'https://other.com', stdDev: 0, isGuessed: true },
        ],
      },
      {
        name: 'Beta',
        company: 'Anthropic',
        url: 'https://example.com/beta',
        release_date: '2024-07-01',
        capabilities: { input: ['text', 'image'], output: ['text'] },
        benchmarks: [
          { name: 'MMLU', score: 80, source: 'https://example.com', stdDev: 1.23456, isGuessed: false },
          { name: 'GSM8K', score: 88.88888, source: '', stdDev: 0, isGuessed: false },
        ],
      },
    ],
  };
  const compact = encodePredictions(input);
  // compact shape
  assert.strictEqual(compact.version, 1);
  assert.deepStrictEqual(compact.benchmarks, ['GSM8K', 'MMLU']);
  assert.deepStrictEqual(compact.companies, ['Anthropic', 'OpenAI']);
  assert.deepStrictEqual(compact.capabilities, ['image', 'text', 'tool']);
  // models are lists
  assert.ok(Array.isArray(compact.models[0]));
  assert.strictEqual(compact.models[0].length, 6);
  // tuples are 3-decimal rounded
  const alphaTupleMMLU = compact.models[0][5][1]; // Alpha is second after sorting? Actually models order preserved, Alpha first: index 0 -> tuples aligned to ['GSM8K','MMLU'], so MMLU at 1
  assert.deepStrictEqual(alphaTupleMMLU, [87.123, 0.123, 1, 0]); // 87.12345 -> 87.123, 0.123456 -> 0.123

  const decoded = decodePredictions(compact);
  assert.strictEqual(decoded.models.length, 2);
  // check Alpha roundtripped with rounding
  const a = decoded.models.find(m => m.name === 'Alpha');
  assert.deepStrictEqual(a.capabilities, { input: ['text'], output: ['text', 'tool'] });
  const mmlu = a.benchmarks.find(b => b.name === 'MMLU');
  assert.strictEqual(mmlu.score, 87.123);
  assert.strictEqual(mmlu.stdDev, 0.123);
  assert.strictEqual(mmlu.source, 'https://example.com');
  assert.strictEqual(mmlu.isGuessed, false);
  // check Beta bitmap with image
  const b = decoded.models.find(m => m.name === 'Beta');
  assert.deepStrictEqual(b.capabilities, { input: ['image', 'text'], output: ['text'] });
});

test('encode/decode empty models', () => {
  const compact = encodePredictions({ models: [] });
  assert.deepStrictEqual(compact.benchmarks, []);
  assert.deepStrictEqual(compact.models, []);
  const decoded = decodePredictions(compact);
  assert.deepStrictEqual(decoded, { models: [] });
});
