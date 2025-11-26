const fs = require('fs');
const path = require('path');
const { estimateMissingBenchmarks } = require('../lib/score-prediction-weighed-bivariate.js');

// Load the benchmark data from aggregated company model files.
// @returns {object} The parsed benchmark data
function loadScoresSync() {
  const { loadModels } = require('../lib/load-models');
  return loadModels();
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
// @returns {object} {mse, duration} The computed MSE on normalized scores, and the duration in seconds.
function computeMSE(benchmarks, numTests = 100, verbose = false) {
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
  shuffle(knownScores, 12345); // Fixed seed for reproducibility

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
  const duration = logProgress('Running tests', testScores.length, i => {
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
  });

  // Calculate and return the Mean Squared Error
  if (squaredErrors.length === 0) {
    throw new Error('No predictions could be made for MSE calculation');
  }

  const mse = squaredErrors.reduce((sum, squaredError) => sum + squaredError, 0) / squaredErrors.length;

  return {mse, duration};
}

function logProgress(msg, total, exec) {
  const eraseLine = '\u001b[2K\r';
  let avgDuration = 0;  // in milliseconds
  for (let i = 0; i < total; i++) {
    const remainingSecs = avgDuration / 1000 * (total-i);
    process.stderr.write(eraseLine);
    process.stderr.write(`${msg} ${i + 1}/${total} ETA ${(remainingSecs/60).toFixed(3)} min\r`);
    const startTime = Date.now();
    exec(i);
    const duration = Date.now() - startTime;
    if (i === 0) { avgDuration = duration; }
    // Exponential moving average with α=1/32
    avgDuration = (avgDuration * 31 + duration) / 32;
  }
  process.stderr.write(eraseLine);
  return avgDuration / 1000;  // return duration in seconds
}


class SFC32 {
  seed(seed) {
    this.state = Uint32Array.from([0, seed, 0, 1]);
    for (let i = 0; i < 12; i++) { this.random32(); }
    return this;
  }
  random32() {
    const s = this.state;
    const r = s[0] + s[1] + s[3];
    s[3]++;
    s[0] = s[1] ^ s[1] >>> 9;
    s[1] = s[2] + (s[2] << 3);
    s[2] = (s[2] << 21 | s[2] >>> 11) + r;
    return r;
  }
  random01() {
    return this.random32() / 0x100000000;
  }
}

function shuffle(array, seed) {
  const rng = new SFC32().seed(seed);
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng.random01() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// Main function to run the MSE benchmark with normalized scores
// @param {number} numTests - Number of deterministic tests to run
// @param {boolean} verbose - Whether to show detailed output
function main(numTests = 100, verbose = false) {
  try {
    if (verbose) {
      console.error('Loading benchmark data...');
    }
    const benchmarks = loadScoresSync();

    if (verbose) {
      console.error(`\nStarting MSE computation with normalized scores using ${numTests} tests...`);
    }
    const {mse, duration} = computeMSE(benchmarks, numTests, verbose);

    console.log(`Number of tests performed:\t${numTests}`);
    console.log(`Mean Squared Error (on normalized scores):\t${mse.toFixed(6)}`);
    console.log(`Standard Error:\t${Math.sqrt(mse).toFixed(6)}`);
    console.log(`Duration:\t${duration.toFixed(3)} seconds`);

    return mse;
  } catch (error) {
    console.error('Error during MSE computation:', error);
    process.exit(1);
  }
}

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  let numTests = 100;
  let verbose = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--verbose' || args[i] === '-v') {
      verbose = true;
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log('Usage: node benchmark/score-prediction.js [numTests] [--verbose|-v]');
      console.log('  numTests: Number of tests to run (default: 100)');
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
