// Tests for pure leaderboard logic using Node's built-in test runner
const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  mean,
  calculateBenchmarkMean,
  calculateMeansByBenchmark,
  calculateBenchmarkEstimators,
  computeAllBenchmarks,
} = require('../lib/leaderboard');

test('mean: empty and simple arrays', () => {
  assert.equal(mean([]), 0);
  assert.equal(mean([10]), 10);
  assert.equal(mean([1, 2, 3]), 2);
});

test('calculateBenchmarkMean: ignores nulls, averages numbers', () => {
  const data = {
    M1: { A: 10, B: null },
    M2: { A: 20, B: null },
    M3: { A: null, B: 5 },
  };
  assert.equal(calculateBenchmarkMean(data, 'A'), 15);
  assert.equal(calculateBenchmarkMean(data, 'B'), 5);
});

test('calculateMeansByBenchmark: returns map of benchmark to mean', () => {
  const data = {
    M1: { X: 2, Y: null },
    M2: { X: 4, Y: 6 },
  };
  const means = calculateMeansByBenchmark(data);
  assert.deepEqual(means, { X: 3, Y: 6 });
});

test('calculateBenchmarkEstimators: fits linear relation with paired data', () => {
  // Construct data where Y = 2*X + 1 for common models, plus some missing values
  const data = {
    A: { X: 0, Y: 1 },
    B: { X: 1, Y: 3 },
    C: { X: 2, Y: 5 },
    D: { X: 3, Y: 7 },
    // Missing pairs should be ignored
    E: { X: 10, Y: null },
    F: { X: null, Y: 123 },
  };
  const est = calculateBenchmarkEstimators(data);
  assert.ok(est.X && est.X.Y, 'estimators should include X->Y');
  assert.ok(est.Y && est.Y.X, 'estimators should include Y->X');
  const xy = est.X.Y;
  assert.ok(Math.abs(xy.a - 2) < 1e-12, `expected slope ~2, got ${xy.a}`);
  assert.ok(Math.abs(xy.b - 1) < 1e-12, `expected intercept ~1, got ${xy.b}`);
  // Inverse regression won't be exact reciprocal; just ensure it's finite
  const yx = est.Y.X;
  assert.ok(Number.isFinite(yx.a));
  assert.ok(Number.isFinite(yx.b));
});

test('calculateBenchmarkEstimators: constant X yields a=0 and b=meanY', () => {
  const data = {
    A: { X: 5, Y: 10 },
    B: { X: 5, Y: 20 },
    C: { X: 5, Y: 30 },
  };
  const est = calculateBenchmarkEstimators(data);
  const xy = est.X.Y;
  assert.equal(xy.a, 0);
  assert.equal(xy.b, 20); // meanY
});

test('computeAllBenchmarks: imputes missing values correctly', () => {
  const data = {
    A: { X: 0, Y: 1 },
    B: { X: 1, Y: 3 },
    C: { X: 2, Y: 5 },
    D: { X: 3, Y: 7 },
    E: { X: 10, Y: null }, // missing Y
    F: { X: null, Y: 123 }, // missing X
  };

  const estimators = calculateBenchmarkEstimators(data);
  const meanX = calculateMeansByBenchmark(data).X;
  const meanY = calculateMeansByBenchmark(data).Y;
  const aXtoY = estimators.X.Y.a;
  const aYtoX = estimators.Y.X.a;
  const expectedY = meanY + (aXtoY + aYtoX) * (data.E.X - meanX) / (1 + aYtoX * aYtoX);
  const expectedX = meanX + (aYtoX + aXtoY) * (data.F.Y - meanY) / (1 + aXtoY * aXtoY);

  const imputed = computeAllBenchmarks(data);
  // Original data should remain unchanged
  assert.deepEqual(data.E.Y, null);
  assert.deepEqual(data.F.X, null);

  // Imputed values should match expectations
  assert.ok(Math.abs(imputed.E.Y - expectedY) < 1e-12, 'imputed Y matches expectation');
  assert.ok(Math.abs(imputed.F.X - expectedX) < 1e-12, 'imputed X matches expectation');

  // No other values should change
  assert.equal(imputed.A.X, data.A.X);
  assert.equal(imputed.A.Y, data.A.Y);
});
