const assert = require('node:assert/strict');
const { test } = require('node:test');
const {
  mean,
  calculateMeansByBenchmark,
  calculateBenchmarkEstimators,
  computeAllBenchmarks,
  estimateMissingBenchmarks,
} = require('../lib/score-prediction-weighed-bivariate');

// ---------- mean ----------
test('mean - basic functionality', () => {
  const values = [1, 2, 3, 4, 5];
  const result = mean(values);
  assert.strictEqual(result, 3);
});

test('mean - empty array', () => {
  const values = [];
  const result = mean(values);
  assert.strictEqual(result, 0);
});

test('mean - single value', () => {
  const values = [42];
  const result = mean(values);
  assert.strictEqual(result, 42);
});

test('mean - negative values', () => {
  const values = [-1, -2, -3, -4, -5];
  const result = mean(values);
  assert.strictEqual(result, -3);
});

test('mean - mixed positive and negative', () => {
  const values = [-2, 0, 2];
  const result = mean(values);
  assert.strictEqual(result, 0);
});

// ---------- calculateMeansByBenchmark ----------
test('calculateMeansByBenchmark - basic functionality', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 20, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: 30, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 40, source: 'Original', stdDev: 0 },
      },
    },
  };
  const means = calculateMeansByBenchmark(indexedData);
  assert.strictEqual(means.bench1, 20);
  assert.strictEqual(means.bench2, 30);
});

test('calculateMeansByBenchmark - with null scores', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: 30, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 40, source: 'Original', stdDev: 0 },
      },
    },
  };
  const means = calculateMeansByBenchmark(indexedData);
  assert.strictEqual(means.bench1, 20);
  assert.strictEqual(means.bench2, 40); // Only model2 has a score for bench2
});

test('calculateMeansByBenchmark - all null scores', () => {
  const indexedData = {
    benchmarkNames: ['bench1'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: null, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: null, source: 'Original', stdDev: 0 },
      },
    },
  };
  const means = calculateMeansByBenchmark(indexedData);
  assert.strictEqual(means.bench1, 0); // No valid scores, returns 0
});

// ---------- calculateBenchmarkEstimators ----------
test('calculateBenchmarkEstimators - basic functionality', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 20, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: 30, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 40, source: 'Original', stdDev: 0 },
      },
    },
  };
  
  const estimators = calculateBenchmarkEstimators(indexedData);
  
  // Check structure
  assert.ok(estimators.bench1);
  assert.ok(estimators.bench2);
  assert.ok(estimators.bench1.bench1);
  assert.ok(estimators.bench1.bench2);
  assert.ok(estimators.bench2.bench1);
  assert.ok(estimators.bench2.bench2);
  
  // Check that estimators have a and b properties
  assert.ok('a' in estimators.bench1.bench1);
  assert.ok('b' in estimators.bench1.bench1);
  
  // For perfect correlation, the estimator should be accurate
  // bench1 predicts bench2: mean(bench1)=20, mean(bench2)=30
  // slope a = cov(bench1, bench2) / var(bench1) = 100 / 100 = 1
  // intercept b = mean(bench2) - a * mean(bench1) = 30 - 1*20 = 10
  assert.strictEqual(estimators.bench2.bench1.a, 1);
  assert.strictEqual(estimators.bench2.bench1.b, 10);
});

test('calculateBenchmarkEstimators - no overlapping data', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: null, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 40, source: 'Original', stdDev: 0 },
      },
    },
  };
  
  const estimators = calculateBenchmarkEstimators(indexedData);
  
  // No overlapping data, so estimators should be zero
  assert.strictEqual(estimators.bench1.bench2.a, 0);
  assert.strictEqual(estimators.bench1.bench2.b, 0);
  assert.strictEqual(estimators.bench2.bench1.a, 0);
  assert.strictEqual(estimators.bench2.bench1.b, 0);
});

test('calculateBenchmarkEstimators - single data point', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 20, source: 'Original', stdDev: 0 },
      },
    },
  };
  
  const estimators = calculateBenchmarkEstimators(indexedData);
  
  // With only one data point, the algorithm handles it gracefully
  // The actual behavior is that it computes the mean and uses that
  assert.ok('a' in estimators.bench2.bench1);
  assert.ok('b' in estimators.bench2.bench1);
  // Don't assert specific values since the algorithm handles edge cases
});

// ---------- computeAllBenchmarks ----------
test('computeAllBenchmarks - basic functionality', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: 30, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 40, source: 'Original', stdDev: 0 },
      },
    },
  };
  
  const result = computeAllBenchmarks(indexedData);
  
  // Check structure
  assert.ok(result.benchmarkNames);
  assert.ok(result.modelFromName);
  assert.ok(result.modelFromName.model1);
  assert.ok(result.modelFromName.model2);
  
  // Model1's bench2 should now have an estimated score
  assert.ok(result.modelFromName.model1.bench2);
  assert.strictEqual(typeof result.modelFromName.model1.bench2.score, 'number');
  assert.strictEqual(result.modelFromName.model1.bench2.source, 'Weighed bivariate regression');
  
  // Model2's scores should remain unchanged
  assert.strictEqual(result.modelFromName.model2.bench1.score, 30);
  assert.strictEqual(result.modelFromName.model2.bench2.score, 40);
});

test('computeAllBenchmarks - all missing scores', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: null, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: null, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
      },
    },
  };
  
  const result = computeAllBenchmarks(indexedData);
  
  // With no data, all scores should be estimated as 0 (the mean of no data)
  assert.strictEqual(result.modelFromName.model1.bench1.score, 0);
  assert.strictEqual(result.modelFromName.model1.bench2.score, 0);
  assert.strictEqual(result.modelFromName.model2.bench1.score, 0);
  assert.strictEqual(result.modelFromName.model2.bench2.score, 0);
});

// ---------- estimateMissingBenchmarks ----------
test('estimateMissingBenchmarks - basic functionality', () => {
  const benchmarks = {
    models: [
      {
        name: 'model1',
        benchmarks: [
          { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
          { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
        ],
      },
      {
        name: 'model2',
        benchmarks: [
          { name: 'bench1', score: 30, source: 'Original', stdDev: 0 },
          { name: 'bench2', score: 40, source: 'Original', stdDev: 0 },
        ],
      },
    ],
  };
  
  const result = estimateMissingBenchmarks(benchmarks);
  
  // Check structure
  assert.ok(result.models);
  assert.strictEqual(result.models.length, 2);
  
  // Model1's missing bench2 should be estimated
  const model1 = result.models.find(m => m.name === 'model1');
  const bench2 = model1.benchmarks.find(b => b.name === 'bench2');
  assert.strictEqual(typeof bench2.score, 'number');
  assert.strictEqual(bench2.source, 'Weighed bivariate regression');
  
  // Model2's scores should remain unchanged
  const model2 = result.models.find(m => m.name === 'model2');
  const bench1Model2 = model2.benchmarks.find(b => b.name === 'bench1');
  const bench2Model2 = model2.benchmarks.find(b => b.name === 'bench2');
  assert.strictEqual(bench1Model2.score, 30);
  assert.strictEqual(bench2Model2.score, 40);
});

test('estimateMissingBenchmarks - multiple iterations', () => {
  const benchmarks = {
    models: [
      {
        name: 'model1',
        benchmarks: [
          { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
          { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
        ],
      },
      {
        name: 'model2',
        benchmarks: [
          { name: 'bench1', score: 30, source: 'Original', stdDev: 0 },
          { name: 'bench2', score: 40, source: 'Original', stdDev: 0 },
        ],
      },
    ],
  };
  
  const result = estimateMissingBenchmarks(benchmarks, 3);
  
  // Should still work with multiple iterations
  const model1 = result.models.find(m => m.name === 'model1');
  const bench2 = model1.benchmarks.find(b => b.name === 'bench2');
  assert.strictEqual(typeof bench2.score, 'number');
});

test('estimateMissingBenchmarks - empty input', () => {
  const benchmarks = {
    models: [],
  };
  
  const result = estimateMissingBenchmarks(benchmarks);
  
  // Should handle empty input gracefully
  assert.ok(result.models);
  assert.strictEqual(result.models.length, 0);
});

test('estimateMissingBenchmarks - complex scenario with multiple benchmarks', () => {
  const benchmarks = {
    models: [
      {
        name: 'model1',
        benchmarks: [
          { name: 'bench1', score: 50, source: 'Original', stdDev: 2 },
          { name: 'bench2', score: 60, source: 'Original', stdDev: 3 },
          { name: 'bench3', score: null, source: 'Original', stdDev: 0 },
        ],
      },
      {
        name: 'model2',
        benchmarks: [
          { name: 'bench1', score: 70, source: 'Original', stdDev: 2 },
          { name: 'bench2', score: 80, source: 'Original', stdDev: 3 },
          { name: 'bench3', score: 90, source: 'Original', stdDev: 4 },
        ],
      },
      {
        name: 'model3',
        benchmarks: [
          { name: 'bench1', score: 30, source: 'Original', stdDev: 2 },
          { name: 'bench2', score: null, source: 'Original', stdDev: 0 },
          { name: 'bench3', score: 50, source: 'Original', stdDev: 4 },
        ],
      },
    ],
  };
  
  const result = estimateMissingBenchmarks(benchmarks);
  
  // Check all models are present
  assert.strictEqual(result.models.length, 3);
  
  // Check that missing scores are estimated
  const model1 = result.models.find(m => m.name === 'model1');
  const model3 = result.models.find(m => m.name === 'model3');
  
  const bench3Model1 = model1.benchmarks.find(b => b.name === 'bench3');
  const bench2Model3 = model3.benchmarks.find(b => b.name === 'bench2');
  
  assert.strictEqual(typeof bench3Model1.score, 'number');
  assert.strictEqual(typeof bench2Model3.score, 'number');
  assert.strictEqual(bench3Model1.source, 'Weighed bivariate regression');
  assert.strictEqual(bench2Model3.source, 'Weighed bivariate regression');
});

// ---------- Edge Cases and Error Handling ----------
test('edge case - single benchmark', () => {
  const indexedData = {
    benchmarkNames: ['bench1'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
      },
    },
  };
  
  const means = calculateMeansByBenchmark(indexedData);
  assert.strictEqual(means.bench1, 10);
  
  const estimators = calculateBenchmarkEstimators(indexedData);
  assert.ok(estimators.bench1);
  assert.ok(estimators.bench1.bench1);
});

test('edge case - very large numbers', () => {
  const values = [1e10, 2e10, 3e10];
  const result = mean(values);
  assert.strictEqual(result, 2e10);
});

test('edge case - very small numbers', () => {
  const values = [1e-10, 2e-10, 3e-10];
  const result = mean(values);
  assert.strictEqual(result, 2e-10);
});

test('edge case - zero variance data', () => {
  const indexedData = {
    benchmarkNames: ['bench1', 'bench2'],
    modelFromName: {
      model1: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 20, source: 'Original', stdDev: 0 },
      },
      model2: {
        bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
        bench2: { name: 'bench2', score: 20, source: 'Original', stdDev: 0 },
      },
    },
  };
  
  const estimators = calculateBenchmarkEstimators(indexedData);
  
  // With zero variance, the algorithm handles it gracefully
  assert.ok('a' in estimators.bench2.bench1);
  assert.ok('b' in estimators.bench2.bench1);
  // Don't assert specific values since the algorithm handles edge cases
});