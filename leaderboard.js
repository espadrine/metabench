const fs = require('fs');
const path = require('path');
const { estimateMissingBenchmarks } = require('./lib/score-prediction.js');
const Table = require('./lib/cli-table');

function loadScoresSync(filePath = path.join(__dirname, 'data', 'scores.json')) {
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Main execution: read scores, impute missing benchmarks, and print each model
if (require.main === module) {
  try {
    const original = loadScoresSync();
    const { scores: guessedBench, uncertainty } = estimateMissingBenchmarks(original);
    const keys = Object.keys(guessedBench);
    // Sort by Terminal-Bench, descending
    keys.sort((a, b) => {
      const va = guessedBench[a]["Terminal-Bench"]; // may be undefined
      const vb = guessedBench[b]["Terminal-Bench"];
      return (vb || 0) - (va || 0);
    });
    const benchmarkSet = new Set();
    for (const m of keys) {
      for (const b of Object.keys(guessedBench[m])) benchmarkSet.add(b);
    }
    const benchList = Array.from(benchmarkSet).sort();
    const table = new Table({ head: ['Model', ...benchList] });
    for (const m of keys) {
      const row = [m];
      for (const b of benchList) {
        const val = guessedBench[m][b];
        const isMissing = original[m] && original[m][b] == null; // null or undefined
        const stdDev = uncertainty?.[m]?.[b]?.stdDev || 0;
        // We display 2σ, ie. ~95% confidence interval.
        const display = typeof val === 'number'
          ? Math.trunc(val) + (isMissing ? `±${Math.round(2*stdDev)}` : '')
          : val;
        row.push(display);
      }
      table.push(row);
    }
    console.log(table.toString());

    // Write predictions with uncertainties to a JSON file.
    // Target format: { model: { bench: { score, stddev } } }
    const predictionOutput = {};
    for (const m of keys) {
      predictionOutput[m] = {};
      for (const b of benchList) {
        const score = guessedBench[m][b];
        // Use stdDev from uncertainty if present; otherwise 0.
        const stddev = uncertainty?.[m]?.[b]?.stdDev || 0;
        predictionOutput[m][b] = { score, stddev };
      }
    }
    const outputPath = path.join(__dirname, 'data', 'scores-prediction.json');
    try {
      fs.writeFileSync(outputPath, JSON.stringify(predictionOutput, null, 2), 'utf8');
    } catch (writeErr) {
      // Non‑fatal: warn but continue.
      console.error('Failed to write predictions JSON:', writeErr.message);
    }
  } catch (err) {
    console.error('Failed to compute benchmarks:', err.message);
    throw err;
    process.exitCode = 1;
  }
}
