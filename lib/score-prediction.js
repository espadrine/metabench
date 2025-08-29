// We have a set of models m1, m2, … and benchmarks b1, b2, …
// Some of the scores s(mi, bj) are known, but some are not.
// We wish to guess the missing scores, using a weighed sum of the model's scores
// on other benchmarks: s'(mi, bj) = Σi,k≠j a(bk, bj) * s(mi, bj) + b(bj)
// To minimize the prediction error, we want to minimize
// L = Σik (s(mi, bj) - s'(mi, bj))².
// We can compute the derivatives:
// dL/da(bk, bj) = -2 Σi (s(mi, bj) - s'(mi, bj)) * s(mi, bk)
// dL/db(bj) = -2 Σi (s(mi, bj) - s'(mi, bj))
// The minimal loss is found when the derivatives are zero.
// That is the solution to the equations (S^T×S)×β(bj) = S^T×s(bj)
// where S is the matrix of known scores (with rows for models
// and columns for benchmarks, plus one column of 1s for the bias b(bj)),
// s(bj) is the vector of known scores for predicted benchmark bj,
// and β(bj) is the vector of coefficients a(bk, bj) and bias b(bj).
//
// We can approximate the solution by plugging in guessed scores
// in place of missing scores. We start with benchmark means as guesses,
// and iterate a few times by using the current guesses to compute new guesses.


// Estimate missing benchmark scores using an iterative approach, with uncertainty estimation.
// 1. Start by filling all missing values with the mean of their respective benchmarks.
// 2. For a fixed number of iterations (e.g., 3), do:
//  a. Train models for each benchmark using the current filled-in data.
//  b. Predict missing values and their uncertainties using these models.
//  c. Update the missing values with the new predictions.
// 3. Return both the final filled-in data and uncertainty information.
//
// Parameters:
// - benchmarks: the benchmark scores. {model: {bench: score (number|null)}}
// - numIterations: number of iterations to perform (default: 3)
// Returns: { scores: estimated benchmark scores ({model: {bench: score (number)}}),
//           uncertainty: uncertainty information ({model: {bench: {variance: number, stdDev: number}}}) }
function estimateMissingBenchmarks(benchmarks, numIterations = 1000) {
  // Create a copy of the input data to not modify it directly
  let result = deepCopy(benchmarks);

  // Compute means for each benchmark across all models where the score is known.
  const means = computeMeans(benchmarks);

  // First, fill all missing values with the mean of their respective benchmarks.
  const allBenches = listBenchmarkNames(benchmarks);

  // Initialize missing values with means and set their uncertainties
  let uncertainty = {}; // To store uncertainties
  for (const model in result) {
    if (!(model in uncertainty)) {
      uncertainty[model] = {};
    }
    for (const bench of allBenches) {
      const originalValue = benchmarks[model]?.[bench];
      if (originalValue === null || originalValue === undefined || !(bench in benchmarks[model])) {
        // Missing value - initialize with mean and high uncertainty
        result[model][bench] = means[bench] || 0;
        uncertainty[model][bench] = {
          variance: (means[bench] || 0) * (means[bench] || 0), // Initial uncertainty based on mean
          stdDev: means[bench] || 0
        };
      } else {
        // Observed value - set uncertainty to 0
        uncertainty[model][bench] = {
          variance: 0,
          stdDev: 0
        };
      }
    }
  }

  // Perform iterative updates
  for (let iter = 0; iter < numIterations; iter++) {
    // Train models for each benchmark using the current filled-in data
    const benchRegression = {};
    for (const bench of allBenches) {
      benchRegression[bench] = trainModelForBenchmark(result, bench);
    }

    // Create a copy of the current result to update
    const newResult = deepCopy(result);
    const newUncertainty = deepCopy(uncertainty);

    // Update missing values using the trained models
    for (const model in result) {
      for (const bench of allBenches) {
        // Only update if the value was originally missing (null or undefined)
        const originalModel = benchmarks[model] || {};
        if (!(bench in originalModel) || originalModel[bench] === null || originalModel[bench] === undefined) {
          if (benchRegression[bench]) {
            const predictionInfo = predictMissingScore(result[model], bench, benchRegression[bench]);
            newResult[model][bench] = predictionInfo.prediction;
            newUncertainty[model][bench] = {
              variance: predictionInfo.variance,
              stdDev: predictionInfo.stdDev
            };
          } else {
            newResult[model][bench] = means[bench] || 0;
            newUncertainty[model][bench] = {
              variance: (means[bench] || 0) * 2, // High uncertainty for mean-based estimates
              stdDev: Math.sqrt((means[bench] || 0) * 2)
            };
          }
        }
      }
    }

    // Update result and uncertainty for the next iteration
    result = newResult;
    uncertainty = newUncertainty;
  }

  // Return both the filled data and the uncertainty information
  return { scores: result, uncertainty };
}

// For a given benchmark, collect all models with known scores
// for that benchmark and the other benchmarks.
// Then, set up and solve the multivariate regression equations
// to find the regression coefficients and bias.
// Parameters:
// - benchmarks: {model: {bench: score (number)}}
//   They should all have a score: we try to predict all of them.
//   Fill missing values with your best estimate, eg. the mean.
// - bench: the name of the benchmark to model (string)
// Returns:
// - coefficients: {otherBench: coefficient}
// - bias: number
// - residualVariance: number (estimate of error variance)
// - covMatrix: 2D array (covariance matrix of coefficients) or null
// - featureBenches: array of benchmark names used as features
function trainModelForBenchmark(benchmarks, bench) {
  const allBenches = listBenchmarkNames(benchmarks);
  const featureBenches = Array.from(allBenches).filter(b => b !== bench);
  if (featureBenches.length === 0) {
    return { coefficients: {}, bias: 0, residualVariance: 0, covMatrix: null, featureBenches: [] };
  }
  const X = [];
  const y = [];
  for (const modelName in benchmarks) {
    const modelBenchmarks = benchmarks[modelName];
    if (modelBenchmarks[bench] === null || modelBenchmarks[bench] === undefined) {
      continue;
    }
    const row = [];
    for (const b of featureBenches) {
      const otherScore = modelBenchmarks[b];
      row.push(otherScore);
    }
    const predictedScore = modelBenchmarks[bench];
    X.push(row);
    y.push(predictedScore);
  }
  if (X.length === 0) {
    return { coefficients: {}, bias: 0, residualVariance: 0, covMatrix: null, featureBenches: featureBenches };
  }
  const X_with_bias = X.map(row => [...row, 1]);
  const numFeaturesWithBias = featureBenches.length + 1;
  const XtX = new Array(numFeaturesWithBias).fill().map(() => new Array(numFeaturesWithBias).fill(0));
  const Xty = new Array(numFeaturesWithBias).fill(0);
  for (let i = 0; i < X_with_bias.length; i++) {
    const xi = X_with_bias[i];
    const yi = y[i];
    for (let j = 0; j < numFeaturesWithBias; j++) {
      Xty[j] += xi[j] * yi;
      for (let k = 0; k <= j; k++) {
        XtX[j][k] += xi[j] * xi[k];
        if (j !== k) {
          XtX[k][j] += xi[j] * xi[k];
        }
      }
    }
  }
  const XtX_copy = XtX.map(row => [...row]);
  const beta = gaussianElimination(XtX_copy, Xty.map(v => v));
  if (!beta) {
    const defaultCoefficients = {};
    featureBenches.forEach(b => {
      defaultCoefficients[b] = 0;
    });
    return { coefficients: defaultCoefficients, bias: 0, residualVariance: 0, covMatrix: null, featureBenches: featureBenches };
  }
  const coefficients = {};
  for (let i = 0; i < featureBenches.length; i++) {
    coefficients[featureBenches[i]] = beta[i];
  }
  const bias = beta[featureBenches.length];

  // Compute residuals and residual variance.
  let sumSquaredResiduals = 0;
  for (let i = 0; i < X_with_bias.length; i++) {
    const xi = X_with_bias[i];
    let pred = bias;
    for (let j = 0; j < featureBenches.length; j++) {
      pred += beta[j] * xi[j];
    }
    const residual = y[i] - pred;
    sumSquaredResiduals += residual * residual;
  }
  const denominator = X.length - numFeaturesWithBias;
  const residualVariance = denominator > 0
    ? Math.max(0, sumSquaredResiduals / denominator)
    : 0;

  // Compute the inverse of XtX to get (X^T X)^{-1}
  const XtX_inverse = invertMatrix(XtX);

  let covMatrix = null;
  if (XtX_inverse) {
    // The covariance matrix is sigma^2 * (X^T X)^{-1}
    covMatrix = XtX_inverse.map(row =>
      row.map(val => val * residualVariance)
    );
  }

  return {
    coefficients,
    bias,
    residualVariance,
    covMatrix,
    featureBenches
  };
}

// Solve x in Ax = b.
// Parameters:
// - A: 2D array (matrix) of size n x n
// - b: array of size n
function gaussianElimination(A, b) {
  const n = A.length;
  for (let i = 0; i < n; i++) {
    // Partial pivoting
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
        maxRow = k;
      }
    }
    // Swap rows
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [b[i], b[maxRow]] = [b[maxRow], b[i]];
    // If diagonal element is zero, matrix is singular
    if (Math.abs(A[i][i]) < 1e-10) {
      return null;
    }
    // Eliminate
    for (let k = i + 1; k < n; k++) {
      const factor = A[k][i] / A[i][i];
      for (let j = i; j < n; j++) {
        A[k][j] -= factor * A[i][j];
      }
      b[k] -= factor * b[i];
    }
  }
  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = b[i];
    for (let j = i + 1; j < n; j++) {
      x[i] -= A[i][j] * x[j];
    }
    x[i] /= A[i][i];
  }
  return x;
}

// Use the trained model to predict a missing score and its uncertainty
// for a given model and benchmark.
// Parameters:
// - modelBenchmarks: {bench: score (number|null)} for a single model
// - bench: the benchmark being predicted (string)
// - benchRegression: {coefficients: {otherBench: coefficient}, bias: number,
//                   residualVariance: number, covMatrix: 2D array|null, featureBenches: array}
function predictMissingScore(modelBenchmarks, bench, benchRegression) {
  const { coefficients, bias, residualVariance, covMatrix, featureBenches } = benchRegression;

  // Prepare the feature vector x, including the bias term (1 at the end)
  const x = [];
  for (const b of featureBenches) {
    const score = modelBenchmarks[b];
    x.push(score);
  }
  x.push(1); // bias term

  // Compute the prediction
  let sum = bias;
  for (let i = 0; i < featureBenches.length; i++) {
    const b = featureBenches[i];
    sum += coefficients[b] * x[i];
  }
  const prediction = sum;

  // Compute the variance of the prediction
  let predictionVariance = residualVariance; // irreducible error
  if (covMatrix) {
    // Add the reducible error: x^T Var(beta) x
    let reducibleError = 0;
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x.length; j++) {
        reducibleError += x[i] * x[j] * covMatrix[i][j];
      }
    }
    predictionVariance += reducibleError;
  } else {
    // If we couldn't compute the covariance matrix, just use residual variance
    predictionVariance = residualVariance;
  }

  // Ensure predictionVariance is not negative (can happen due to numerical issues)
  predictionVariance = Math.max(0, predictionVariance);

  return {
    prediction,
    variance: predictionVariance,
    stdDev: Math.sqrt(predictionVariance)
  };
}

// Calculate the mean score for each benchmark
// across all models where the score is known.
// This helps in imputing missing values.
function computeMeans(benchmarks) {
  const benchMeans = {};
  const benchCounts = {};
  const allBenches = listBenchmarkNames(benchmarks);
  // Initialize
  for (const bench of allBenches) {
    benchMeans[bench] = 0;
    benchCounts[bench] = 0;
  }
  // Compute sums and counts
  for (const model in benchmarks) {
    for (const bench in benchmarks[model]) {
      const score = benchmarks[model][bench];
      if (score !== null && score !== undefined) {
        benchMeans[bench] += score;
        benchCounts[bench] += 1;
      }
    }
  }
  // Compute means
  for (const bench in benchMeans) {
    if (benchCounts[bench] > 0) {
      benchMeans[bench] /= benchCounts[bench];
    } else {
      benchMeans[bench] = 0;
    }
  }
  return benchMeans;
}

// Take benchmark data and return benchmark names.
// - benchmarks: {model: {bench: score (number|null)}}
// Returns: set of benchmark names (String)
function listBenchmarkNames(benchmarks) {
  const allBenches = new Set();
  for (const model in benchmarks) {
    for (const bench in benchmarks[model]) {
      allBenches.add(bench);
    }
  }
  return allBenches;
}

// Function to create a deep copy of the data
function deepCopy(data) {
  return JSON.parse(JSON.stringify(data));
}

// Invert a square matrix using Gaussian elimination
function invertMatrix(matrix) {
  const n = matrix.length;
  if (n === 0) return null;
  const inverse = new Array(n).fill().map(() => new Array(n).fill(0));
  const identity = new Array(n).fill().map((_, i) => new Array(n).fill(0).map((_, j) => (i === j ? 1 : 0)));

  for (let col = 0; col < n; col++) {
    // Make a deep copy of matrix for each column to solve
    const A = matrix.map(row => [...row]);
    const b = identity.map(row => row[col]);
    const x = gaussianElimination(A, b);
    if (!x) return null;
    for (let i = 0; i < n; i++) {
      inverse[i][col] = x[i];
    }
  }
  return inverse;
}

module.exports = {
  estimateMissingBenchmarks,
  trainModelForBenchmark,
  gaussianElimination,
  predictMissingScore,
  computeMeans,
  listBenchmarkNames,
  deepCopy,
  invertMatrix
};
