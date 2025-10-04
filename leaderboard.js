const fs = require('fs');
const path = require('path');
const { estimateMissingBenchmarks } = require('./lib/score-prediction-weighed-bivariate.js');
const Table = require('./lib/cli-table');

// Load the benchmark data from data/models.json.
// File format:
// { models: [ { name: string, benchmarks: [ { name: string, score: number }, … ] }, … ] }
function loadScoresSync(filePath = path.join(__dirname, 'data', 'models.json')) {
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Write the predicted scores (with uncertainties) to a JSON file.
// This is extracted into a function to keep the main logic focused.
function writePredictionsOutput(predictionOutput, filePath) {
  fs.writeFileSync(filePath, JSON.stringify(predictionOutput, null, 2), 'utf8');
}

// benchmarks: {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
function printTable(benchmarks) {
  const benchmarkSet = new Set();
  for (const model of benchmarks.models) {
    for (const eval of model.benchmarks) {
      benchmarkSet.add(eval.name);
    }
  }
  const benchmarkNames = Array.from(benchmarkSet).sort();

  // Sort by Terminal-Bench, descending
  benchmarks.models.sort((m1, m2) => {
    const s1 = m1.benchmarks.find(b => b.name === 'Terminal-Bench')?.score || 0;
    const s2 = m2.benchmarks.find(b => b.name === 'Terminal-Bench')?.score || 0;
    return (s2 || 0) - (s1 || 0);
  });
  const table = new Table({ head: ['Model', ...benchmarkNames] });
  for (const model of benchmarks.models) {
    const row = [model.name];
    for (const benchName of benchmarkNames) {
      const eval = model.benchmarks.find(b => b.name === benchName);
      const { score, stdDev } = eval;
      // We display 2σ, ie. ~95% confidence interval.
      const display = typeof score === 'number'
        ? Math.trunc(score) + (stdDev !== 0 ? `±${Math.round(2*stdDev)}` : '')
        : score;
      row.push(display);
    }
    table.push(row);
  }
  console.log(table.toString());
}

// Main execution: read scores, impute missing benchmarks, and print each model
if (require.main === module) {
  const raw = loadScoresSync();
  const benchmarks = estimateMissingBenchmarks(raw);
  //printTable(benchmarks);

  const outputPath = path.join(__dirname, 'data', 'models-prediction.json');
  writePredictionsOutput(benchmarks, outputPath);
}
