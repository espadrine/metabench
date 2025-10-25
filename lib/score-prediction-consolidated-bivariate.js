// Pure leaderboard logic (exported for tests and reuse)
const { Variable, Constant } = require('./autograd');

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
// - benchmarks: the benchmark scores. {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
// - iterations: number of SGD iterations (number, default 1000)
// Returns: estimated benchmark scores ({models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]})
function estimateMissingBenchmarks(benchmarks, iterations = 10000) {
  // Index the input data for easier processing
  const indexedData = indexBenchmarkData(benchmarks);

  // Compute initial estimates for all benchmarks using the closed‑form
  // solution.  These estimates will be used to seed unknown scores.
  const scoreEstimates = computeAllBenchmarks(indexedData);
  for (const [modelName, benchmarks] of Object.entries(scoreEstimates.modelFromName)) {
    for (const [benchName, bench] of Object.entries(benchmarks)) {
      returnBenchData = indexedData.modelFromName[modelName][benchName];
      returnBenchData.score = bench.score;
      returnBenchData.stdDev = bench.stdDev;
    }
  }

  // Convert back to the original format
  return unindexBenchmarkData(indexedData);
}

// Return a new indexed data object where all missing benchmark values are
// estimated using the closed‑form solution derived from the loss
// minimisation.  The algorithm uses the pairwise regressors
// (`a_{bj,bk}` and `a_{bk,bj}`) computed by
// `calculateBenchmarkEstimators` and the marginal means of each
// benchmark.
//
// Parameters:
// - indexedData: {benchmarkNames, modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}}}
// Returns: estimated indexed data ({benchmarkNames, modelFromName: {<name>: {<benchmark name>: {name, score: number, source, stdDev}}}})
function computeAllBenchmarks(indexedData) {
  // Pre‑compute the necessary statistics once.
  const estimators = calculateBenchmarkEstimators(indexedData);
  const means = calculateMeansByBenchmark(indexedData);
  const estimated = { benchmarkNames: indexedData.benchmarkNames, modelFromName: {} };

  // The score for benchmark k is:
  // s[bk] = mean[bk] + Σj ((a[bj,bk] + a[bk,bj])*(s[bj] - mean[bj])) / Σj (1 + a[bk,bj]^2)

  for (const [modelName, model] of Object.entries(indexedData.modelFromName)) {
    const newModel = {};
    // Determine benchmarks present on this model.
    for (const benchName of indexedData.benchmarkNames) {
      const bench = model[benchName];
      if (bench && bench.score != null) {
        // Copy existing score with its properties
        newModel[benchName] = { ...bench };
      } else {
        // Missing value – compute estimate.
        const meanK = means[benchName] ?? 0;
        let num = 0;
        let den = 0;
        for (const [bj, benchBj] of Object.entries(model)) {
          if (bj === benchName) continue;
          if (!benchBj || benchBj.score == null) continue;
          const sj = benchBj.score;
          const aJK = estimators[bj] && estimators[bj][benchName] ? estimators[bj][benchName].a : 0;
          const aKJ = estimators[benchName] && estimators[benchName][bj] ? estimators[benchName][bj].a : 0;
          const meanJ = means[bj] ?? 0;
          num += (aJK + aKJ) * (sj - meanJ);
          den += 1 + aKJ * aKJ;
        }
        const estimatedValue = den !== 0 ? meanK + num / den : meanK;
        newModel[benchName] = { name: benchName, score: estimatedValue, source: 'Bivariate regression', stdDev: 0 };
      }
    }
    estimated.modelFromName[modelName] = newModel;
  }

  return estimated;
}

// Compute predictors between benchmark scores.
// Parameters:
// - indexedData: {benchmarkNames, modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}}}
// Return an object { known_benchmark: { unknown_benchmark: { a: number, b: number } } },
// where the predicted score for unknown_benchmark is a*known_score + b.
function calculateBenchmarkEstimators(indexedData) {
  // The rough formula for a and b are:
  // a = cov(X, Y) / var(X)
  // b = mean(Y) - a * mean(X)
  //
  // A more efficient formula (converting benchmark bj to bk) is:
  // a[bj,bk] = Σ((s[bj] - mean[bj]) * (s[bk] - mean[bk])) / Σ((s[bj] - mean[bj])²)
  // b[bj,bk] = mean[bk] - a[bj,bk] * mean[bj]

  const benchmarkList = indexedData.benchmarkNames;
  const result = {};
  for (const bj of benchmarkList) {
    result[bj] = {};
    for (const bk of benchmarkList) {
      // Collect paired numeric (score of benchmark j, score of benchmark k)
      // where both bj and bk exist
      const pairs = [];
      for (const model of Object.values(indexedData.modelFromName)) {
        const sbj = model[bj] ? model[bj].score : null;
        const sbk = model[bk] ? model[bk].score : null;
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

// Calculate means for each benchmark across all models where the score is known
// Parameters:
// - indexedData: {benchmarkNames, modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}}}
function calculateMeansByBenchmark(indexedData) {
  const result = {};
  for (const bench of indexedData.benchmarkNames) {
    const values = [];
    for (const model of Object.values(indexedData.modelFromName)) {
      if (model[bench] && model[bench].score != null) {
        values.push(model[bench].score);
      }
    }
    result[bench] = mean(values);
  }
  return result;
}

function mean(values) {
  if (!Array.isArray(values) || values.length === 0) return 0;
  const sum = values.reduce((a, b) => a + b, 0);
  return sum / values.length;
}

// Add indexes for quick access to the benchmark data structure.
// - benchmarks: the benchmark scores. {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
// Returns: {
//   benchmarkNames: array of String,
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
        modelFromName[model][bench] = { name: bench, score: null, source: 'Bivariate regression', stdDev: 0 };
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
  mean,
  calculateMeansByBenchmark,
  calculateBenchmarkEstimators,
  computeAllBenchmarks,
  estimateMissingBenchmarks,
};
