// Simple mean-based benchmark prediction algorithm

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
        modelFromName[model][bench] = { name: bench, score: null, source: 'Mean prediction', stdDev: 0 };
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

// Calculate means for each benchmark across all models where the score is known
// Parameters:
// - indexedData: {benchmarkNames, modelFromName: {<name>: {<benchmark name>: {name, score: number|null, source, stdDev}}}}
function calculateBenchmarkMeans(indexedData) {
  const means = {};
  for (const benchName of indexedData.benchmarkNames) {
    let sum = 0;
    let count = 0;

    for (const model of Object.values(indexedData.modelFromName)) {
      const benchmark = model[benchName];
      if (benchmark && benchmark.score != null) {
        sum += benchmark.score;
        count += 1;
      }
    }

    means[benchName] = count > 0 ? sum / count : 0;
  }
  return means;
}

// Estimate missing benchmark scores by replacing each missing score with the mean of that benchmark
//
// Parameters:
// - benchmarks: the benchmark scores. {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
// Returns: estimated benchmark scores ({models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]})
function estimateMissingBenchmarks(benchmarks) {
  // Index the input data for easier processing
  const indexedData = indexBenchmarkData(benchmarks);

  // Calculate the mean for each benchmark
  const means = calculateBenchmarkMeans(indexedData);

  // Update missing scores with the mean for their respective benchmarks
  for (const modelName in indexedData.modelFromName) {
    for (const benchName of indexedData.benchmarkNames) {
      const benchmark = indexedData.modelFromName[modelName][benchName];
      if (benchmark && benchmark.score == null) {
        // Replace missing score with the mean for this benchmark
        indexedData.modelFromName[modelName][benchName] = {
          name: benchName,
          score: means[benchName],
          source: 'Mean prediction',
          stdDev: Math.abs(means[benchName]) // Use absolute mean as a basic uncertainty estimate
        };
      }
    }
  }

  // Convert back to the original format
  return unindexBenchmarkData(indexedData);
}

module.exports = {
  estimateMissingBenchmarks
};
