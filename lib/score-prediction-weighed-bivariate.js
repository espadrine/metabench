// Estimate missing benchmark scores using weighed bivariate regression.
// It computes the regression coefficients only between pairs of benchmarks, and
// averages the predictions of other benchmarks for its own score estimation.
//
// Parameters:
// - benchmarks: the benchmark scores. {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
// - iterations: number of SGD iterations (number, default 1)
// Returns: estimated benchmark scores ({models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]})
function estimateMissingBenchmarks(benchmarks, iterations = 1) {
  // Index the input data for easier processing
  const indexedData = indexBenchmarkData(benchmarks);

  // Compute initial estimates for all benchmarks using the closed‑form
  // solution.  These estimates will be used to seed unknown scores.
  let scoreEstimates = indexedData;
  for (let i = 0; i < iterations; i++) {
    scoreEstimates = computeAllBenchmarks(scoreEstimates);
  }
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
  // {estimators: {<predictor benchmark>: {<predicted benchmark>: {a, b}}}}
  calculateBenchmarkEstimators(indexedData);
  // Bivariate regression scores are added to the indexData as {estimations}:
  // {<model name>: {<predicted benchmark>: {<predictor benchmark>: {score, variance}}}},
  const biScores = predictBivariateScores(indexedData);
  const means = calculateMeansByBenchmark(indexedData);
  const estimated = { benchmarkNames: indexedData.benchmarkNames, modelFromName: {} };

  for (const [modelName, model] of Object.entries(indexedData.modelFromName)) {
    const newModel = {};
    for (const bk of indexedData.benchmarkNames) {
      const bench = model[bk];
      const meanK = means[bk] ?? 0;
      if (bench != null && bench.score != null && bench.source !== 'Weighed bivariate regression') {
        // Copy existing score with its properties
        newModel[bk] = { ...bench };
      } else {
        // Missing value – compute estimate.
        let num = 0, den = 0, numVar = 0;
        for (const [bj, benchBj] of Object.entries(model)) {
          // Only predict from real values.
          if (benchBj.score == null) { continue; }
          if (biScores[modelName][bk][bj] == null) { continue; }
          const sk = biScores[modelName][bk][bj].score;
          if (sk == null) { continue; }
          const weight = 1;
          num += sk * weight;
          den += weight;
          // The overall variance is Σ var(estimator) ÷ N².
          numVar += weight * weight * indexedData.estimations[modelName][bk][bj].variance;
        }

        let estimatedValue, estimatedVariance;
        if (den === 0) {
          estimatedValue = meanK;
          estimatedVariance = estimatedValue * estimatedValue;
        } else {
          estimatedValue = num / den;
          estimatedVariance = numVar / (den * den);
        }
        newModel[bk] = {
          name: bk,
          score: estimatedValue,
          stdDev: Math.sqrt(estimatedVariance),
          source: 'Weighed bivariate regression',
        };
      }
    }
    estimated.modelFromName[modelName] = newModel;
  }

  return estimated;
}

// Compute predictors between benchmark scores.
// Parameters:
// - indexedData: {benchmarkNames, modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}}}
// Return {<predicted benchmark>: {<predictor benchmark>: {a: number, b: number}}},
// where the predicted score for unknown_benchmark is a*known_score + b.
// Those estimators are also added to the indexedData as {estimators}.
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
  for (const bk of benchmarkList) {
    result[bk] = {};
    for (const bj of benchmarkList) {
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
        result[bk][bj] = { a: 0, b: 0 };
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
      result[bk][bj] = { a, b };
    }
  }

  indexedData.estimators = result;

  return result;
}

// Compute the estimation for the model scores using only
// bivariate regression, obtained from calculateBenchmarkEstimators().
// We even compute the scores when we already have it,
// which will be useful when estimating the variance.
// Parameters:
// - indexedData: {
//     benchmarkNames: list of strings,
//     modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}},
//     estimators: {<predicted benchmark>: {<predictor benchmark>: {a: number, b: number}}},
//   }
// Returns {<model name>: {<predicted benchmark>: {<predictor benchmark>: {score, variance}}}},
// which it also adds to the indexedData as {estimations}
function predictBivariateScores(indexedData) {
  predictBivariateScoreEstimates(indexedData);
  predictBivariateScoreVariances(indexedData);
  return indexedData.estimations;
}

// Parameters:
// - indexedData: {
//     benchmarkNames: list of strings,
//     modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}},
//     estimators: {<predicted benchmark>: {<predictor benchmark>: {a: number, b: number}}},
//   }
// Returns {<model name>: {<predicted benchmark>: {<predictor benchmark>: {score}}}},
// which it also adds to indexedData as {estimations}.
function predictBivariateScoreEstimates(indexedData) {
  const predictions = {};
  const {estimators} = indexedData;
  for (const [modelName, model] of Object.entries(indexedData.modelFromName)) {
    predictions[modelName] = {};
    for (const bk of indexedData.benchmarkNames) {
      predictions[modelName][bk] = {};
      // Make one prediction for each predictor benchmark that has a real score.
      for (const [bj, benchBj] of Object.entries(model)) {
        if (bj === bk) { continue; }
        if (benchBj == null || benchBj.score == null) { continue; }

        // The score for benchmark k is:
        // s[bk] = Σj s[bj] × a[bj,bk] + b[bj,bk] ÷ N.
        const sj = benchBj.score;
        const aJK = estimators[bk][bj].a;
        const bJK = estimators[bk][bj].b;
        const sk = sj * aJK + bJK;
        predictions[modelName][bk][bj] = { score: sk };
      }
    }
  }
  indexedData.estimations = predictions;
  return predictions;
}

// Parameters:
// - indexedData: {
//     benchmarkNames: list of strings,
//     modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}},
//     estimators: {<predicted benchmark>: {<predictor benchmark>: {a: number, b: number}}},
//     estimations: {<model name>: {<predicted benchmark>: {<predictor benchmark>: {score}}}},
//   }
// Returns {<model name>: {<predicted benchmark>: {<predictor benchmark>: {score, variance}}}},
// which it also sets in indexedData in {estimations}.
function predictBivariateScoreVariances(indexedData) {
  // Compute the mean squared error between known benchmark scores and estimated ones.
  // {<predicted benchmark>: {<predictor benchmark>: {mse}}}
  const {estimators} = indexedData;
  for (const bk of indexedData.benchmarkNames) {
    for (const bj of indexedData.benchmarkNames) {
      const est = estimators[bk][bj];
      // The mean squared error is Σ (a×sj + b - sk)² ÷ (N-2)
      let count = 0, se = 0;
      for (const model of Object.values(indexedData.modelFromName)) {
        if (model[bk] == null || model[bj] == null) { continue; }
        if (model[bk].score == null || model[bj].score == null) { continue; }
        // We now have two defined scores for the two benchmarks on the same model.
        const estimatedScore = model[bj].score * est.a + est.b;
        const scoreDiff = model[bk].score - estimatedScore;
        se += scoreDiff * scoreDiff;
        count += 1;
      }
      // The normal formula is N-2, but if we don't have enough, we make do.
      est.mse = se / Math.max(count - 2, 1);
    }
  }
  for (const model of Object.values(indexedData.estimations)) {
    for (const bk of indexedData.benchmarkNames) {
      for (const bj of indexedData.benchmarkNames) {
        if (model[bk][bj] == null) { continue; }
        model[bk][bj].variance = estimators[bk][bj].mse;
      }
    }
  }
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
