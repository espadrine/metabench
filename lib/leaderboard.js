// Pure leaderboard logic (exported for tests and reuse)
const { Variable, Constant } = require('./autograd');

function mean(values) {
  if (!Array.isArray(values) || values.length === 0) return 0;
  const sum = values.reduce((a, b) => a + b, 0);
  return sum / values.length;
}

function calculateBenchmarkMean(data, benchmark) {
  // Preserve existing behavior: ignore only nulls
  const values = Object.values(data)
    .map(m => m && m[benchmark])
    .filter(v => v !== null);
  return mean(values);
}

function calculateMeansByBenchmark(data) {
  const benchmarks = new Set();
  for (const model of Object.values(data)) {
    if (model && typeof model === 'object') {
      Object.keys(model).forEach(b => benchmarks.add(b));
    }
  }
  const result = {};
  for (const b of benchmarks) {
    result[b] = calculateBenchmarkMean(data, b);
  }
  return result;
}

// Compute predictors between benchmark scores.
// data is of the form { model_name: { benchmark_name: score, … }, … }
// Return an object { known_benchmark: { unknown_benchmark: { a: number, b: number } } },
// where the predicted score for unknown_benchmark is a*known_score + b.
function calculateBenchmarkEstimators(data) {
  // Determine all benchmark names present across models
  const benchmarks = new Set();
  for (const model of Object.values(data)) {
    if (model && typeof model === 'object') {
      Object.keys(model).forEach(b => benchmarks.add(b));
    }
  }

  // The rough formula for a and b are:
  // a = cov(X, Y) / var(X)
  // b = mean(Y) - a * mean(X)
  //
  // A more efficient formula (converting benchmark bj to bk) is:
  // a[bj,bk] = Σ((s[bj] - mean[bj]) * (s[bk] - mean[bk])) / Σ((s[bj] - mean[bj])²)
  // b[bj,bk] = mean[bk] - a[bj,bk] * mean[bj]

  const benchmarkList = Array.from(benchmarks);
  const result = {};
  for (const bj of benchmarkList) {
    result[bj] = {};
    for (const bk of benchmarkList) {
      // Collect paired numeric (score of benchmark j, score of benchmark k)
      // where both bj and bk exist
      const pairs = [];
      for (const model of Object.values(data)) {
        if (!model) continue;
        const sbj = model[bj];
        const sbk = model[bk];
        if (sbj != null && sbk != null) {
          pairs.push([sbj, sbk]);
        }
      }

      if (pairs.length === 0) {
        result[bj][bk] = { a: 0, b: 0 };
        continue;
      }

      const sbjs = pairs.map(p => p[0]);
      const sbks = pairs.map(p => p[1]);
      const meanSbj = mean(sbjs);
      const meanSbk = mean(sbks);
      let num = 0;
      let den = 0;
      for (const [sbj, sbk] of pairs) {
        const dj = sbj - meanSbj;
        const dk = sbk - meanSbk;
        num += dj * dk;
        den += dj * dj;
      }
      const a = den !== 0 ? num / den : 0;
      const b = meanSbk - a * meanSbj;
      result[bj][bk] = { a, b };
    }
  }

  return result;
}

// Return a new data object where all missing benchmark values are
// estimated using the closed‑form solution derived from the loss
// minimisation.  The algorithm uses the pairwise regressors
// (`a_{bj,bk}` and `a_{bk,bj}`) computed by
// `calculateBenchmarkEstimators` and the marginal means of each
// benchmark.
//
// Parameters:
// - data: the benchmark scores. {model: {bench: score (number|null)}}
// Returns: estimated benchmark scores ({model: {bench: score (number)}})
function computeAllBenchmarks(data) {
  // Pre‑compute the necessary statistics once.
  const estimators = calculateBenchmarkEstimators(data);
  const means = calculateMeansByBenchmark(data);
  const estimated = {};

  // The score for benchmark k is:
  // s[bk] = mean[bk] + Σj ((a[bj,bk] + a[bk,bj])*(s[bj] - mean[bj])) / Σj (1 + a[bk,bj]^2)

  for (const [modelName, model] of Object.entries(data)) {
    const newModel = {};
    // Determine benchmarks present on this model.
    for (const bench of Object.keys(model)) {
      const score = model[bench];
      if (score != null) {
        newModel[bench] = score;
      } else {
        // Missing value – compute estimate.
        const meanK = means[bench] ?? 0;
        let num = 0;
        let den = 0;
        for (const [bj, sj] of Object.entries(model)) {
          if (bj === bench) continue;
          if (sj == null) continue;
          const aJK = estimators[bj] && estimators[bj][bench] ? estimators[bj][bench].a : 0;
          const aKJ = estimators[bench] && estimators[bench][bj] ? estimators[bench][bj].a : 0;
          const meanJ = means[bj] ?? 0;
          num += (aJK + aKJ) * (sj - meanJ);
          den += 1 + aKJ * aKJ;
        }
        const estimated = den !== 0 ? meanK + num / den : meanK;
        newModel[bench] = estimated;
      }
    }
    estimated[modelName] = newModel;
  }

  return estimated;
}

// Estimate missing benchmark scores using stochastic gradient descent.
//
// The function builds a computational graph where known scores are
// `Constant`s and missing scores are `Variable`s.  For every model and
// every pair of benchmarks `(bj, bk)`, a prediction `a * s[bk] + b` is
// generated using the coefficients from `calculateBenchmarkEstimators`.
// The squared error `(pred - s[bj])^2` is accumulated into a global
// loss variable.  After the loss is built, gradients are computed by
// calling `loss.computeGradients()`.  The function then performs a
// specified number of SGD iterations, logging the loss at each step.
//
// Parameters:
// - data: the benchmark scores. {model: {bench: score (number|null)}}
// - iterations: number of SGD iterations (numnber, default 100)
// Returns: estimated benchmark scores ({model: {bench: score (number)}})
function estimateMissingBenchmarks(data, iterations = 10000000) {
  // Compute initial estimates for all benchmarks using the closed‑form
  // solution.  These estimates will be used to seed unknown scores.
  const initScoreEstimates = computeAllBenchmarks(data);

  // Build benchmark variables: known scores are Constants, unknown
  // scores are Variables initialized to the closed‑form estimate.
  const benchmarks = {};
  for (const [modelName, model] of Object.entries(data)) {
    benchmarks[modelName] = {};
    for (const bench of Object.keys(model)) {
      const known = model[bench];
      if (known != null) {
        benchmarks[modelName][bench] = new Constant(known);
      } else {
        const initVal = initScoreEstimates[modelName][bench];
        benchmarks[modelName][bench] = new Variable(initVal);
      }
    }
  }

  // Build estimator variables from the closed‑form coefficients.
  const initScoreEstimators = calculateBenchmarkEstimators(data);
  const scoreEstimators = {};
  for (const bj of Object.keys(initScoreEstimators)) {
    scoreEstimators[bj] = {};
    for (const bk of Object.keys(initScoreEstimators[bj])) {
      const { a, b } = initScoreEstimators[bj][bk];
      scoreEstimators[bj][bk] = { a: new Variable(a), b: new Variable(b) };
    }
  }

  // Gather all Variable nodes for SGD.  The estimators are Variable
  // objects and will be updated during training.
  const allVars = varListFromGraph(benchmarks, scoreEstimators);

  const [maxStepSize, minStepSize] = [0.1, 1e-6];
  for (let i = 0; i < iterations; i++) {
    const loss = squaredPredictionErrorLoss(benchmarks, scoreEstimators);
    loss.computeGradients();
    const progress = i / iterations; // From 0 to 1
    const stepSize = maxStepSize * (1 - progress) + minStepSize * progress;
    gradientDescent(allVars, stepSize);
    const { a, b } = scoreEstimators.Aider.Codeforces;
    console.error(`Iteration ${i + 1}/${iterations} loss: ${loss.value} a: ${a.value} b: ${b.value} da: ${a.gradient} db: ${b.gradient}`);
  }

  // Extract the final estimated scores from the benchmark nodes.
  const estimated = {};
  for (const [m, model] of Object.entries(benchmarks)) {
    estimated[m] = {};
    for (const [bench, node] of Object.entries(model)) {
      estimated[m][bench] = node.value;
    }
  }
  return estimated;
}

// Helper that builds the loss computation graph from the graph of
// benchmark values and the variable estimators.
function squaredPredictionErrorLoss(benchmarks, scoreEstimators) {
  let loss = new Constant(0);
  for (const [m, model] of Object.entries(benchmarks)) {
    for (const [bj, scoreBj] of Object.entries(model)) {
      for (const [bk, scoreBk] of Object.entries(model)) {
        if (bj === bk) continue;
        // Only include predictions of known scores.
        //if (scoreBj instanceof Variable) continue;
        const { a, b } = scoreEstimators[bk][bj];
        const pred = a.multiply(scoreBk).add(b);
        const error = pred.subtract(scoreBj);
        const squaredError = error.power(2);
        loss = loss.add(squaredError);
      }
    }
  }
  return loss;
}

// Perform a simple SGD update on all Variable nodes in the benchmarks and
// estimator variables.
//
// Parameters:
// - variables: Variable nodes to update (Array)
// - stepSize: size of the step of the descent (Number)
function gradientDescent(variables, stepSize) {
  // Compute the norm of the gradient.
  let norm = 0;
  for (const node of variables) {
    norm += node.gradient * node.gradient;
  }
  norm = Math.sqrt(norm);

  lr = stepSize / norm;
  for (const node of variables) {
    node.value -= lr * node.gradient;
  }
}

// Build a data structure of the form { model: { benchmark: score } }
// where score is a Constant for known values and Variable for missing ones.
function buildBenchmarkVariables(data) {
  const benchmarks = {};
  for (const [m, model] of Object.entries(data)) {
    benchmarks[m] = {};
    for (const [bench, score] of Object.entries(model)) {
      benchmarks[m][bench] = score != null ? new Constant(score) : new Variable(0);
    }
  }
  return benchmarks;
}

function buildBenchmarkEstimatorVariables(data) {
  // Determine all benchmark names present across models
  const benchmarks = new Set();
  for (const model of Object.values(data)) {
    if (model && typeof model === 'object') {
      Object.keys(model).forEach(b => benchmarks.add(b));
    }
  }

  const scoreEstimators = {};
  for (const bj of benchmarks) {
    scoreEstimators[bj] = {};
    for (const bk of benchmarks) {
      scoreEstimators[bj][bk] = { a: new Variable(0), b: new Variable(0) };
    }
  }

  return scoreEstimators;
}

function varListFromGraph(benchmarks, scoreEstimators) {
  // Collect all Variable nodes in the benchmarks and estimators
  const variables = [];
  for (const model of Object.values(benchmarks)) {
    for (const node of Object.values(model)) {
      if (node instanceof Variable) {
        variables.push(node);
      }
    }
  }
  for (const estimators of Object.values(scoreEstimators)) {
    for (const { a, b } of Object.values(estimators)) {
      if (a instanceof Variable) variables.push(a);
      if (b instanceof Variable) variables.push(b);
    }
  }
  return variables;
}

module.exports = {
  mean,
  calculateBenchmarkMean,
  calculateMeansByBenchmark,
  calculateBenchmarkEstimators,
  computeAllBenchmarks,
  estimateMissingBenchmarks,
};
