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
// 1. Start by filling all missing values with an initial flawed estimate.
// 2. For a fixed number of iterations (e.g., 3), do:
//  a. Train models for each benchmark using the current filled-in data.
//  b. Predict missing values and their uncertainties using these models.
//  c. Update the missing values with the new predictions.
// 3. Return both the final filled-in data and uncertainty information.
//
// Parameters:
// - benchmarks: the benchmark scores. {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
// - numIterations: number of iterations to perform (default: 1000)
// Returns: {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
function estimateMissingBenchmarks(benchmarks, numIterations = 1000) {
  benchmarks = indexBenchmarkData(benchmarks);

  // Create a copy of the input data to not modify it directly
  let result = deepCopy(benchmarks);

  // Compute weighed bivariate estimates for estimation initialization.
  const weighedBivariate = require('./score-prediction-weighed-bivariate.js');
  const {modelFromName: initialEstimations} = weighedBivariate.computeAllBenchmarks(deepCopy(benchmarks));

  // Initialize missing values with the weighed bivariate estimates.
  for (const modelName in benchmarks.modelFromName) {
    for (const bench in benchmarks.modelFromName[modelName]) {
      const eval = benchmarks.modelFromName[modelName][bench];
      if (eval.score == null) {
        eval.score = initialEstimations[modelName][bench].score;
        eval.stdDev = eval.score;  // Arbitrary high variance.
      }
    }
  }

  // Perform iterative updates
  for (let iter = 0; iter < numIterations; iter++) {
    const benchRegression = {};
    for (const bench of benchmarks.benchmarkNames) {
      benchRegression[bench] = trainModelForBenchmark(benchmarks, bench);
    }
    const newBenchmarks = deepCopy(benchmarks);
    for (const model in newBenchmarks.modelFromName) {
      const modelBenchmarks = newBenchmarks.modelFromName[model];
      for (const bench of benchmarks.benchmarkNames) {
        const eval = modelBenchmarks[bench];
        if (eval.source === 'Multivariate regression') {
            // Predict scores using the full previous information.
            const predictionInfo = predictMissingScore(benchmarks.modelFromName[model], bench, benchRegression[bench]);
            eval.score = predictionInfo.prediction;
            eval.stdDev = predictionInfo.stdDev;
        }
      }
    }
    benchmarks = newBenchmarks;
  }

  return unindexBenchmarkData(benchmarks);
}

// For a given benchmark, collect all models with known scores
// for that benchmark and the other benchmarks.
// Then, set up and solve the multivariate regression equations
// to find the regression coefficients and bias.
// Parameters:
// - benchmarks: {benchmarkNames, modelFromName: {<name>: {<benchmark name>: {name, score: number, source, stdDev}}}}
//   It has the format of the output of indexBenchmarkData(),
//   except all evals should have a score: we try to predict all of them.
//   Fill missing values with your best estimate, eg. the mean.
// - bench: the name of the benchmark to model (string)
// Returns:
// - coefficients: {otherBench: coefficient}
// - bias: number
// - residualVariance: number (estimate of error variance)
// - covMatrix: 2D array (covariance matrix of coefficients) or null
// - featureBenches: array of benchmark names used as features
function trainModelForBenchmark(benchmarks, bench) {
  const featureBenches = benchmarks.benchmarkNames.filter(b => b !== bench);
  if (featureBenches.length === 0) {
    return { coefficients: {}, bias: 0, residualVariance: 0, covMatrix: null, featureBenches: [] };
  }
  const X = [];
  const y = [];
  for (const model in benchmarks.modelFromName) {
    const modelBenchmarks = benchmarks.modelFromName[model];
    const row = [];
    for (const b of featureBenches) {
      const otherScore = modelBenchmarks[b].score;
      row.push(otherScore);
    }
    const predictedScore = modelBenchmarks[bench].score;
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
    // Singular matrix: fallback to mean prediction for each model.
    const meanY = y.reduce((a, b) => a + b, 0) / y.length;
    const defaultCoefficients = {};
    featureBenches.forEach(b => { defaultCoefficients[b] = 0; });
    return { coefficients: defaultCoefficients, bias: meanY, residualVariance: 0, covMatrix: null, featureBenches };
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
    if (Math.abs(A[i][i]) < 1e-50) {
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
// - modelBenchmarks: {<benchmark name>: {name, score: number|null, source, stdDev}} for a single model.
// - bench: the benchmark being predicted (string)
// - benchRegression: {coefficients: {otherBench: coefficient}, bias: number,
//                   residualVariance: number, covMatrix: 2D array|null, featureBenches: array}
function predictMissingScore(modelBenchmarks, bench, benchRegression) {
  const { coefficients, bias, residualVariance, covMatrix, featureBenches } = benchRegression;

  // Prepare the feature vector x, including the bias term (1 at the end)
  const x = [];
  for (const b of featureBenches) {
    const score = modelBenchmarks[b].score;
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

// Add indexes for quick access to the benchmark data structure.
// - benchmarks: the benchmark scores. {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
// Returns: {
//   benchmarkNames: Set of String,
//   modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}}
function indexBenchmarkData(benchmarks) {
  benchmarks = deepCopy(benchmarks);
  const benchmarkNames = new Set();
  const modelNames = new Set();
  for (const model of benchmarks.models) {
    modelNames.add(model.name);
    for (const benchmark of model.benchmarks) {
      benchmarkNames.add(benchmark.name);
    }
  }

  const modelFromName = {};
  for (const model of modelNames) {
    modelFromName[model] = {};
    for (const bench of benchmarkNames) {
      // We could have multiple scores for a given model and benchmark (from multiple runs and sources).
      // For now, we put each in a list.
      modelFromName[model][bench] = [];
    }
  }

  for (const model of benchmarks.models) {
    for (const benchmark of model.benchmarks) {
      modelFromName[model.name][benchmark.name].push(benchmark);
    }
  }

  // We now merge the scores for a given benchmark and score.
  for (const model of modelNames) {
    for (const bench of benchmarkNames) {
      const entries = modelFromName[model][bench];
      if (entries.length > 1) {
        // Merge the entries by averaging their scores and stdDevs.
        let sumScores = 0;
        for (const entry of entries) {
          sumScores += entry.score;
        }
        const avgScore = sumScores / entries.length;

        let sumErrors = 0;
        for (const entry of entries) {
          const error = (entry.score - avgScore);
          sumErrors += error * error;
        }
        const stdDev = Math.sqrt(sumErrors / (entries.length - 1));

        const source = 'Multiple: ' + entries.map(e => `Score ${e.score} at ${e.source}`).join('; ');

        modelFromName[model][bench] = {
          name: bench,
          score: avgScore,
          source: source,
          stdDev: stdDev,
        };
      } else if (entries.length === 1) {
        const { name, score, source, stdDev } = entries[0];
        modelFromName[model][bench] = {
          name,
          score,
          source: source || 'Original',
          stdDev: stdDev || 0,
        };
      } else {
        modelFromName[model][bench] = { name: bench, score: null, source: 'Multivariate regression', stdDev: 0 };
      }
    }
  }

  return { benchmarkNames: Array.from(benchmarkNames), modelFromName };
}

function unindexBenchmarkData(benchmarks) {
  const models = [];
  for (const modelName in benchmarks.modelFromName) {
    const modelBenchmarks = [];
    for (const benchName of benchmarks.benchmarkNames) {
      modelBenchmarks.push(benchmarks.modelFromName[modelName][benchName]);
    }
    models.push({
      name: modelName,
      benchmarks: modelBenchmarks,
    });
  }
  return { models };
}

// Function to create a deep copy of the data
function deepCopy(data) {
  return JSON.parse(JSON.stringify(data));
}

module.exports = {
  estimateMissingBenchmarks,
  trainModelForBenchmark,
  gaussianElimination,
  predictMissingScore,
  listBenchmarkNames,
  deepCopy,
  invertMatrix
};
