const fs = require('fs');
const path = require('path');
const { estimateMissingBenchmarks } = require('../lib/score-prediction.js');

// Load the benchmark data from data/models.json.
// @param {string} filePath - Path to the models.json file
// @returns {object} The parsed benchmark data
function loadScoresSync(filePath = path.join(__dirname, '..', 'data', 'models.json')) {
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Calculate means and standard deviations for each benchmark
// @param {object} benchmarks - The benchmark data structure
// @returns {object} An object with benchmark means and stdDevs
function calculateBenchmarkStats(benchmarks) {
  const benchStats = {};

  // First pass: calculate means
  const benchSums = {};
  const benchCounts = {};

  for (const model of benchmarks.models) {
    for (const benchmark of model.benchmarks) {
      if (benchmark.score !== undefined && benchmark.score !== null && typeof benchmark.score === 'number') {
        const benchName = benchmark.name;

        if (!benchSums[benchName]) {
          benchSums[benchName] = 0;
          benchCounts[benchName] = 0;
        }

        benchSums[benchName] += benchmark.score;
        benchCounts[benchName] += 1;
      }
    }
  }

  // Calculate means
  const benchMeans = {};
  for (const benchName in benchSums) {
    benchMeans[benchName] = benchSums[benchName] / benchCounts[benchName];
  }

  // Second pass: calculate standard deviations
  const benchSquaredDiffs = {};
  for (const model of benchmarks.models) {
    for (const benchmark of model.benchmarks) {
      if (benchmark.score !== undefined && benchmark.score !== null && typeof benchmark.score === 'number') {
        const benchName = benchmark.name;
        const diff = benchmark.score - benchMeans[benchName];

        if (!benchSquaredDiffs[benchName]) {
          benchSquaredDiffs[benchName] = 0;
        }

        benchSquaredDiffs[benchName] += diff * diff;
      }
    }
  }

  // Calculate standard deviations
  const benchStdDevs = {};
  for (const benchName in benchSquaredDiffs) {
    const variance = benchSquaredDiffs[benchName] / benchCounts[benchName];
    benchStdDevs[benchName] = Math.sqrt(variance);

    // Handle case where std dev is 0 (all scores for this benchmark are identical)
    if (benchStdDevs[benchName] === 0) {
      benchStdDevs[benchName] = 1; // Prevent division by zero, meaning normalized score will be 0
    }
  }

  return { means: benchMeans, stdDevs: benchStdDevs };
}

// Normalize a score using the formula: (score - mean) / stdDev
// @param {number} score - The raw score to normalize
// @param {string} benchmarkName - The name of the benchmark
// @param {object} stats - Object with means and stdDevs for each benchmark
// @returns {number} The normalized score
function normalizeScore(score, benchmarkName, stats) {
  const mean = stats.means[benchmarkName] || 0;
  const stdDev = stats.stdDevs[benchmarkName] || 1;
  return (score - mean) / stdDev;
}

// Compute Mean Squared Error (MSE) for the score prediction algorithm using normalized scores
// by systematically removing whole benchmark objects and measuring prediction accuracy.
// @param {object} benchmarks - The benchmark data structure
// @param {number} numTests - Number of deterministic tests to run
// @param {boolean} verbose - Whether to show detailed output
// @returns {number} The computed MSE on normalized scores
function computeMSE(benchmarks, numTests = 50, verbose = false) {
  // Create a deep copy of the original data to avoid modifications
  const originalBenchmarks = JSON.parse(JSON.stringify(benchmarks));

  // Calculate benchmark statistics (means and standard deviations) for normalization
  const stats = calculateBenchmarkStats(originalBenchmarks);

  // Find all known scores that we can use for testing
  const knownScores = [];
  for (const model of originalBenchmarks.models) {
    for (const benchmark of model.benchmarks) {
      if (benchmark.score !== undefined && benchmark.score !== null && typeof benchmark.score === 'number') {
        const normalizedTrueScore = normalizeScore(benchmark.score, benchmark.name, stats);
        knownScores.push({
          modelName: model.name,
          benchmarkName: benchmark.name,
          trueScore: benchmark.score,
          normalizedTrueScore: normalizedTrueScore
        });
      }
    }
  }

  // If we have fewer known scores than requested tests, adjust accordingly
  const actualNumTests = Math.min(numTests, knownScores.length);

  // Select deterministic subset of known scores for testing
  // Sort by model name then benchmark name for deterministic selection
  knownScores.sort((a, b) => {
    if (a.modelName !== b.modelName) {
      return a.modelName.localeCompare(b.modelName);
    }
    return a.benchmarkName.localeCompare(b.benchmarkName);
  });

  const testScores = knownScores.slice(0, actualNumTests);

  // Array to store the squared errors
  const squaredErrors = [];

  if (verbose) {
    console.error(`Computing MSE using ${actualNumTests} deterministic test cases with normalized scores...`);
    console.error(`Benchmark statistics:`);
    for (const benchName in stats.means) {
      console.error(`  ${benchName}: mean=${stats.means[benchName].toFixed(4)}, std=${stats.stdDevs[benchName].toFixed(4)}`);
    }
    console.error('');
  }

  // For each test score, remove the entire benchmark object, predict it, and calculate error
  for (let i = 0; i < testScores.length; i++) {
    const testScore = testScores[i];

    if (verbose) {
      console.error(`Test ${i + 1}/${actualNumTests}: ${testScore.modelName} - ${testScore.benchmarkName} (true score: ${testScore.trueScore})`);
    }

    // Create a copy of the original data and remove the entire benchmark object
    const testData = JSON.parse(JSON.stringify(originalBenchmarks));

    // Find and remove the entire benchmark object from the test data
    for (const model of testData.models) {
      if (model.name === testScore.modelName) {
        for (const [index, benchmark] of model.benchmarks.entries()) {
          if (benchmark.name === testScore.benchmarkName) {
            model.benchmarks.splice(index, 1); // Remove the entire benchmark object
            break;
          }
        }
        break;
      }
    }

    // Use estimateMissingBenchmarks to predict the removed score
    const predictedBenchmarks = estimateMissingBenchmarks(testData);

    // Find the predicted score for our test case
    let predictedScore = null;
    for (const model of predictedBenchmarks.models) {
      if (model.name === testScore.modelName) {
        for (const benchmark of model.benchmarks) {
          if (benchmark.name === testScore.benchmarkName) {
            predictedScore = benchmark.score;
            break;
          }
        }
        break;
      }
    }

    if (predictedScore !== null && predictedScore !== undefined) {
      // Normalize both the predicted and true score using the original data stats
      const normalizedPredictedScore = normalizeScore(predictedScore, testScore.benchmarkName, stats);
      const normalizedTrueScore = testScore.normalizedTrueScore;

      const squaredError = Math.pow(normalizedPredictedScore - normalizedTrueScore, 2);
      squaredErrors.push(squaredError);

      if (verbose) {
        console.error(`  Predicted: ${predictedScore.toFixed(4)} (normalized: ${normalizedPredictedScore.toFixed(4)}), True normalized: ${normalizedTrueScore.toFixed(4)}, Error: ${Math.abs(normalizedPredictedScore - normalizedTrueScore).toFixed(4)}, Squared Error: ${squaredError.toFixed(4)}`);
      }
    } else {
      console.warn(`  Warning: Could not predict score for ${testScore.modelName} - ${testScore.benchmarkName}`);
    }
  }

  // Calculate and return the Mean Squared Error
  if (squaredErrors.length === 0) {
    throw new Error('No predictions could be made for MSE calculation');
  }

  const mse = squaredErrors.reduce((sum, squaredError) => sum + squaredError, 0) / squaredErrors.length;

  return mse;
}

// Main function to run the MSE benchmark with normalized scores
// @param {number} numTests - Number of deterministic tests to run
// @param {boolean} verbose - Whether to show detailed output
function main(numTests = 50, verbose = false) {
  try {
    if (verbose) {
      console.error('Loading benchmark data...');
    }
    const benchmarks = loadScoresSync();

    if (verbose) {
      console.error(`\nStarting MSE computation with normalized scores using ${numTests} tests...`);
    }
    const mse = computeMSE(benchmarks, numTests, verbose);

    console.log(`Number of tests performed:\t${numTests}`);
    console.log(`Mean Squared Error (on normalized scores):\t${mse.toFixed(6)}`);
    console.log(`Root Mean Squared Error:\t${Math.sqrt(mse).toFixed(6)}`);

    return mse;
  } catch (error) {
    console.error('Error during MSE computation:', error);
    process.exit(1);
  }
}

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  let numTests = 50;
  let verbose = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--verbose' || args[i] === '-v') {
      verbose = true;
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log('Usage: node benchmark/score-prediction.js [numTests] [--verbose|-v]');
      console.log('  numTests: Number of tests to run (default: 50)');
      console.log('  --verbose, -v: Show detailed output (default: false)');
      console.log('  --help, -h: Show this help message');
      process.exit(0);
    } else if (!isNaN(parseInt(args[i]))) {
      numTests = parseInt(args[i], 10);
    } else {
      console.error(`Unknown argument: ${args[i]}`);
      console.log('Use --help for usage information');
      process.exit(1);
    }
  }

  if (isNaN(numTests) || numTests <= 0) {
    console.error('numTests must be a positive integer');
    process.exit(1);
  }

  return { numTests, verbose };
}

// Run the benchmark if this file is executed directly
if (require.main === module) {
  const { numTests, verbose } = parseArgs();
  main(numTests, verbose);
}

module.exports = {
  computeMSE,
  loadScoresSync,
  calculateBenchmarkStats,
  normalizeScore
};
