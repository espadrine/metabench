const fs = require('fs');
const path = require('path');
const { computeAllBenchmarks, estimateMissingBenchmarks } = require('./lib/leaderboard');
const Table = require('./cli-table');

function loadScoresSync(filePath = path.join(__dirname, 'scores.json')) {
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Main execution: read scores, impute missing benchmarks, and print each model
if (require.main === module) {
  try {
    const original = loadScoresSync();
    //const guessedBench = computeAllBenchmarks(original);
    const guessedBench = estimateMissingBenchmarks(original);
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
        const display = typeof val === 'number'
          ? Math.trunc(val) + (isMissing ? '?' : '')
          : val;
        row.push(display);
      }
      table.push(row);
    }
    console.log(table.toString());
  } catch (err) {
    console.error('Failed to compute benchmarks:', err.message);
    process.exitCode = 1;
  }
}
