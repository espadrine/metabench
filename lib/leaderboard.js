// Pure leaderboard logic (exported for tests and reuse)

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

/**
 * Return a new data object where all missing benchmark values are
 * imputed using the closed‑form solution derived from the loss
 * minimisation.  The algorithm uses the pairwise regressors
 * (`a_{bj,bk}` and `a_{bk,bj}`) computed by
 * `calculateBenchmarkEstimators` and the marginal means of each
 * benchmark.
 *
 * @param {{[model:string]: {[bench:string]: number|null}}} data
 * @returns {{[model:string]: {[bench:string]: number}}}
 */
function computeAllBenchmarks(data) {
  // Pre‑compute the necessary statistics once.
  const estimators = calculateBenchmarkEstimators(data);
  const means = calculateMeansByBenchmark(data);
  const imputed = {};

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
    imputed[modelName] = newModel;
  }

  return imputed;
}

module.exports = {
  mean,
  calculateBenchmarkMean,
  calculateMeansByBenchmark,
  calculateBenchmarkEstimators,
  computeAllBenchmarks,
};
