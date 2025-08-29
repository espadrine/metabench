const { test } = require('node:test');
const assert = require('node:assert/strict');
const {
  estimateMissingBenchmarks,
  trainModelForBenchmark,
  gaussianElimination,
  predictMissingScore,
  computeMeans,
  listBenchmarkNames,
  deepCopy,
  invertMatrix
} = require('../lib/score-prediction');

// Test estimateMissingBenchmarks
test('estimateMissingBenchmarks - basic functionality', () => {
  const benchmarks = {
    model1: { bench1: 10, bench2: null },
    model2: { bench1: 20, bench2: 30 }
  };
  const result = estimateMissingBenchmarks(benchmarks, 0); // Only initial fill
  assert.strictEqual(result.scores.model1.bench2, 30); // Mean of bench2 is (30)/1 = 30
  assert.strictEqual(result.scores.model2.bench2, 30); // Already has value
  assert(result.uncertainty.model1.bench2.variance > 0); // Should have positive variance
  assert.strictEqual(result.uncertainty.model2.bench2.variance, 0); // Known value should have 0 variance
});

test('estimateMissingBenchmarks - iterative updates', () => {
  const benchmarks = {
    model1: { bench1: 10, bench2: null },
    model2: { bench1: 20, bench2: 30 }
  };
  const result = estimateMissingBenchmarks(benchmarks, 1); // One iteration
  // After one iteration, bench2 for model1 should be updated from the initial mean
  assert.notStrictEqual(result.scores.model1.bench2, 30); // Should be different after iteration
});

test('estimateMissingBenchmarks - empty input', () => {
  const result = estimateMissingBenchmarks({});
  assert.deepStrictEqual(result.scores, {});
  assert.deepStrictEqual(result.uncertainty, {});
});

// Test trainModelForBenchmark
test('trainModelForBenchmark - basic functionality', () => {
  const benchmarks = {
    model1: { bench1: 10, bench2: 20 },
    model2: { bench1: 30, bench2: 40 }
  };
  const result = trainModelForBenchmark(benchmarks, 'bench2');
  assert(result.coefficients.bench1 !== undefined); // Should have coefficient for bench1
  assert(result.bias !== undefined); // Should have bias
  assert(result.residualVariance == 0);
});

// Test gaussianElimination
test('gaussianElimination - basic functionality', () => {
  const A = [
    [2, 1],
    [4, 3]
  ];
  const b = [5, 11];
  const x = gaussianElimination(A, b);
  assert.deepStrictEqual(x, [2, 1]); // Solution to 2x + y = 5, 4x + 3y = 11
});

test('gaussianElimination - singular matrix', () => {
  const A = [
    [1, 1],
    [1, 1]
  ];
  const b = [2, 2];
  const x = gaussianElimination(A, b);
  assert.strictEqual(x, null); // Should return null for singular matrix
});

// Test predictMissingScore
test('predictMissingScore - basic functionality', () => {
  const modelBenchmarks = { bench1: 10 };
  const benchRegression = {
    coefficients: { bench1: 2 },
    bias: 5,
    residualVariance: 1,
    covMatrix: null,
    featureBenches: ['bench1']
  };
  const result = predictMissingScore(modelBenchmarks, 'bench2', benchRegression);
  assert.strictEqual(result.prediction, 25); // 2*10 + 5
  assert.strictEqual(result.variance, 1); // residual variance
  assert.strictEqual(result.stdDev, 1); // sqrt(1)
});

// Test computeMeans
test('computeMeans - basic functionality', () => {
  const benchmarks = {
    model1: { bench1: 10, bench2: 20 },
    model2: { bench1: 30, bench2: 40 }
  };
  const means = computeMeans(benchmarks);
  assert.strictEqual(means.bench1, 20); // (10+30)/2
  assert.strictEqual(means.bench2, 30); // (20+40)/2
});

// Test listBenchmarkNames
test('listBenchmarkNames - basic functionality', () => {
  const benchmarks = {
    model1: { bench1: 10, bench2: 20 },
    model2: { bench1: 30, bench3: 40 }
  };
  const benches = listBenchmarkNames(benchmarks);
  assert(benches.has('bench1'));
  assert(benches.has('bench2'));
  assert(benches.has('bench3'));
  assert.strictEqual(benches.size, 3);
});

// Test deepCopy
test('deepCopy - basic functionality', () => {
  const original = { a: 1, b: { c: 2 } };
  const copy = deepCopy(original);
  assert.deepStrictEqual(copy, original);
  assert.notStrictEqual(copy, original); // Different reference
  copy.b.c = 3;
  assert.notStrictEqual(original.b.c, 3); // Original not modified
});

// Test invertMatrix
test('invertMatrix - basic functionality', () => {
  const matrix = [
    [1, 0],
    [0, 1]
  ];
  const inverse = invertMatrix(matrix);
  assert.deepStrictEqual(inverse, [
    [1, 0],
    [0, 1]
  ]); // Identity matrix should invert to itself
});

test('invertMatrix - 2x2 matrix', () => {
  const matrix = [
    [2, 1],
    [1, 2]
  ];
  const inverse = invertMatrix(matrix);
  assert.deepStrictEqual(inverse, [
    [2/3, -1/3],
    [-1/3, 2/3]
  ]); // 2x2 matrix inverse
});

test('invertMatrix - singular matrix', () => {
  const matrix = [
    [1, 1],
    [1, 1]
  ];
  const inverse = invertMatrix(matrix);
  assert.strictEqual(inverse, null); // Should return null for singular matrix
});
