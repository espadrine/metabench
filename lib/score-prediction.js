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


// Estimate missing benchmark scores using an iterative approach.
// 1. Start by filling all missing values with the mean of their respective benchmarks.
// 2. For a fixed number of iterations (e.g., 3), do:
//  a. Train models for each benchmark using the current filled-in data.
//  b. Predict missing values using these models.
//  c. Update the missing values with the new predictions.
// 3. Return the final filled-in data.
//
// Parameters:
// - benchmarks: the benchmark scores. {model: {bench: score (number|null)}}
// - numIterations: number of iterations to perform (default: 3)
// Returns: estimated benchmark scores ({model: {bench: score (number)}})
function estimateMissingBenchmarks(benchmarks, numIterations = 1000) {
  // Create a copy of the input data to not modify it directly
  let result = deepCopy(benchmarks);

  // Compute means for each benchmark across all models where the score is known.
  const means = computeMeans(benchmarks);

  // First, fill all missing values with the mean of their respective benchmarks.
  const allBenches = listBenchmarkNames(benchmarks);

  // Initialize missing values with means
  for (const model in result) {
    for (const bench of allBenches) {
      if (!(bench in result[model]) || result[model][bench] === null || result[model][bench] === undefined) {
        result[model][bench] = means[bench] || 0;
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

    // Update missing values using the trained models
    for (const model in result) {
      for (const bench of allBenches) {
        // Only update if the value was originally missing (null or undefined)
        const originalModel = benchmarks[model] || {};
        if (!(bench in originalModel) || originalModel[bench] === null || originalModel[bench] === undefined) {
          if (benchRegression[bench]) {
            newResult[model][bench] = predictMissingScore(result[model], bench, benchRegression[bench]);
          } else {
            newResult[model][bench] = means[bench] || 0;
          }
        }
      }
    }

    // Update result for the next iteration
    result = newResult;
  }

  return result;
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
function trainModelForBenchmark(benchmarks, bench) {
  const allBenches = listBenchmarkNames(benchmarks);
  const featureBenches = Array.from(allBenches).filter(b => b !== bench);
  if (featureBenches.length === 0) {
    return { coefficients: {}, bias: 0 };
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
    return { coefficients: {}, bias: 0 };
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
  const beta = gaussianElimination(XtX, Xty);
  if (!beta) {
    const defaultCoefficients = {};
    featureBenches.forEach(b => {
      defaultCoefficients[b] = 0;
    });
    return { coefficients: defaultCoefficients, bias: 0 };
  }
  const coefficients = {};
  for (let i = 0; i < featureBenches.length; i++) {
    coefficients[featureBenches[i]] = beta[i];
  }
  const bias = beta[featureBenches.length];
  return { coefficients, bias };
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

// Use the trained model to predict a missing predictedScore
// for a given model and benchmark.
// Parameters:
// - modelBenchmarks: {bench: score (number|null)} for a single model
// - bench: the benchmark being predicted (string)
// - benchRegression: {coefficients: {otherBench: coefficient}, bias: number}
function predictMissingScore(modelBenchmarks, bench, benchRegression) {
  const { coefficients, bias } = benchRegression;
  let sum = bias;
  for (const b in coefficients) {
    const score = modelBenchmarks[b];
    const coeff = coefficients[b];
    sum += coeff * score;
  }
  return sum;
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

module.exports = {
  estimateMissingBenchmarks,
};
