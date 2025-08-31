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
  invertMatrix,
} = require('../lib/score-prediction');

// ---------- estimateMissingBenchmarks ----------
test('estimateMissingBenchmarks - basic functionality', () => {
  const data = {
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
          { name: 'bench1', score: 20, source: 'Original', stdDev: 0 },
          { name: 'bench2', score: 30, source: 'Original', stdDev: 0 },
        ],
      },
    ],
  };
  const result = estimateMissingBenchmarks(data, 0); // No iterations
  const model1Bench2 = result.models[0].benchmarks[1];
  const model2Bench2 = result.models[1].benchmarks[1];
  assert.strictEqual(model1Bench2.score, 30);
  assert.strictEqual(model2Bench2.score, 30);
  assert.ok(model1Bench2.stdDev > 0, 'variance should be positive for imputed value');
  assert.strictEqual(model2Bench2.stdDev, 0, 'known value should have zero variance');
});

test('estimateMissingBenchmarks - iterative updates', () => {
  const data = {
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
          { name: 'bench1', score: 20, source: 'Original', stdDev: 0 },
          { name: 'bench2', score: 30, source: 'Original', stdDev: 0 },
        ],
      },
    ],
  };
  const result = estimateMissingBenchmarks(data, 1); // One iteration
  const model1Bench2 = result.models[0].benchmarks[1];
  assert.strictEqual(model1Bench2.score, 30, 'score remains the mean after one iteration');
  assert.strictEqual(model1Bench2.stdDev, 30, 'variance remains from initial imputation');
});

test('estimateMissingBenchmarks - empty input', () => {
  const result = estimateMissingBenchmarks({ models: [] }, 0);
  assert.deepStrictEqual(result, { models: [] });
});

// ---------- trainModelForBenchmark ----------
test('trainModelForBenchmark - basic functionality', () => {
  const benchmarks = {
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
  const result = trainModelForBenchmark(benchmarks, 'bench2');
  assert.ok(result.coefficients.bench1 !== undefined, 'coefficient for bench1 should exist');
  assert.ok(result.bias !== undefined, 'bias should be defined');
  assert.strictEqual(result.residualVariance, 0, 'residual variance should be zero for perfect fit');
});

// ---------- gaussianElimination ----------
test('gaussianElimination - basic functionality', () => {
  const A = [
    [2, 1],
    [4, 3],
  ];
  const b = [5, 11];
  const x = gaussianElimination(A, b);
  assert.deepStrictEqual(x, [2, 1]);
});

test('gaussianElimination - singular matrix', () => {
  const A = [
    [1, 1],
    [1, 1],
  ];
  const b = [2, 2];
  const x = gaussianElimination(A, b);
  assert.strictEqual(x, null);
});

// ---------- predictMissingScore ----------
test('predictMissingScore - basic functionality', () => {
  const modelBenchmarks = {
    bench1: { name: 'bench1', score: 10, source: 'Original', stdDev: 0 },
  };
  const benchRegression = {
    coefficients: { bench1: 2 },
    bias: 5,
    residualVariance: 1,
    covMatrix: null,
    featureBenches: ['bench1'],
  };
  const result = predictMissingScore(modelBenchmarks, 'bench2', benchRegression);
  assert.strictEqual(result.prediction, 25, 'prediction should be 2*10 + 5');
  assert.strictEqual(result.variance, 1, 'variance should equal residual variance when covMatrix is null');
  assert.strictEqual(result.stdDev, 1, 'stdDev should be sqrt of variance');
});

// ---------- computeMeans ----------
test('computeMeans - basic functionality', () => {
  const benchmarks = {
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
  const means = computeMeans(benchmarks);
  assert.strictEqual(means.bench1, 20);
  assert.strictEqual(means.bench2, 30);
});

// ---------- listBenchmarkNames ----------
test('listBenchmarkNames - basic functionality', () => {
  const benchmarks = {
    model1: { bench1: 10, bench2: 20 },
    model2: { bench1: 30, bench3: 40 },
  };
  const benches = listBenchmarkNames(benchmarks);
  assert(benches.has('bench1'));
  assert(benches.has('bench2'));
  assert(benches.has('bench3'));
  assert.strictEqual(benches.size, 3);
});

// ---------- deepCopy ----------
test('deepCopy - basic functionality', () => {
  const original = { a: 1, b: { c: 2 } };
  const copy = deepCopy(original);
  assert.deepStrictEqual(copy, original);
  assert.notStrictEqual(copy, original);
  copy.b.c = 3;
  assert.notStrictEqual(original.b.c, 3);
});

// ---------- invertMatrix ----------
test('invertMatrix - basic functionality', () => {
  const matrix = [
    [1, 0],
    [0, 1],
  ];
  const inverse = invertMatrix(matrix);
  assert.deepStrictEqual(inverse, [
    [1, 0],
    [0, 1],
  ]);
});

test('invertMatrix - 2x2 matrix', () => {
  const matrix = [
    [2, 1],
    [1, 2],
  ];
  const inverse = invertMatrix(matrix);
  assert.deepStrictEqual(inverse, [
    [2 / 3, -1 / 3],
    [-1 / 3, 2 / 3],
  ]);
});

test('invertMatrix - singular matrix', () => {
  const matrix = [
    [1, 1],
    [1, 1],
  ];
  const inverse = invertMatrix(matrix);
  assert.strictEqual(inverse, null);
});

