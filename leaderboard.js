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

// Load benchmark capability data from data/benchmarks.json
function loadBenchmarkCapabilities() {
  const filePath = path.join(__dirname, 'data', 'benchmarks.json');
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Check if a model has the required capabilities for a benchmark
function getMissingCapabilities(modelCapabilities, benchmarkCapabilities) {
  const missing = {
    input: [],
    output: []
  };

  // If no capability requirements specified, return empty object (no missing capabilities)
  if (!modelCapabilities || !benchmarkCapabilities) {
    return missing;
  }

  // Check input capabilities
  const requiredInputs = benchmarkCapabilities.input || [];
  const modelInputs = modelCapabilities.input || [];

  for (const requiredInput of requiredInputs) {
    if (!modelInputs.includes(requiredInput)) {
      missing.input.push(requiredInput);
    }
  }

  // Check output capabilities
  const requiredOutputs = benchmarkCapabilities.output || [];
  const modelOutputs = modelCapabilities.output || [];

  for (const requiredOutput of requiredOutputs) {
    if (!modelOutputs.includes(requiredOutput)) {
      missing.output.push(requiredOutput);
    }
  }

  return missing;
}

// Replace predicted scores with random scores when models lack required capabilities
function adjustScoresByCapabilities(benchmarks) {
  const benchmarkData = loadBenchmarkCapabilities();
  const benchmarkMap = new Map();

  // Create a map for quick lookup of benchmark requirements
  benchmarkData.benchmarks.forEach(benchmarkInfo => {
    benchmarks.models.forEach(model => {
      const missingCapabilities = getMissingCapabilities(model.capabilities, benchmarkInfo.capabilities);
      const hasMissingCapabilities = missingCapabilities.input.length > 0 || missingCapabilities.output.length > 0;
      if (hasMissingCapabilities) {
        // Model lacks required capabilities, replace with random score
        const missingInputs = missingCapabilities.input.map(cap => `input:${cap}`);
        const missingOutputs = missingCapabilities.output.map(cap => `output:${cap}`);
        const allMissing = [...missingInputs, ...missingOutputs];
        model.benchmarks.push({
          name: benchmarkInfo.name,
          score: benchmarkInfo.random_score,
          source: `Missing capability: ${allMissing.join(', ')}`,
        });
      }
    });
  });

  return benchmarks;
}

// Add capabilities back to models in prediction output
function addCapabilitiesToPrediction(benchmarks, rawScores) {
  const modelCapabilitiesMap = new Map();
  
  // Create a map for model capabilities from original data
  rawScores.models.forEach(model => {
    modelCapabilitiesMap.set(model.name, model.capabilities);
  });
  
  benchmarks.models.forEach(model => {
    // Add capabilities from original data
    const modelCapabilities = modelCapabilitiesMap.get(model.name);
    if (modelCapabilities) {
      model.capabilities = modelCapabilities;
    }
  });
  
  return benchmarks;
}

// Calculate "Cost of 1K responses" benchmark
// Formula: (ArtificialAnalysis Consumed Tokens (Millions) / 7.3) * (119 / 1000)
function addCostOf1KResponses(benchmarks) {
  // Mistral Small 3.2 tokens consumed by Artificial Analysis benchmarks, in millions
  const BASELINE_AA_TOKEN_CONSUMPTION = 7.3;
  // Tokens from sample question "What is the unit of cross-entropy?" given to Mistral Small 3.2
  const BASELINE_TOKENS_PER_INPUT = 11;
  const BASELINE_TOKENS_PER_OUTPUT = 119;
  const RESPONSES_PER_K = 1000; // 1K responses

  benchmarks.models.forEach(model => {
    const inputCostPerMillionTokens = model.benchmarks.find(b =>
      b.name === 'Input cost'
    );
    const outputCostPerMillionTokens = model.benchmarks.find(b =>
      b.name === 'Output cost'
    );
    // Find the ArtificialAnalysis Consumed Tokens (Millions) benchmark
    const aaTokenConsumption = model.benchmarks.find(b =>
      b.name === 'ArtificialAnalysis Consumed Tokens (Millions)'
    );

    if (aaTokenConsumption && typeof aaTokenConsumption.score === 'number') {
      const costPerOutputToken = outputCostPerMillionTokens.score / 1e6;
      const costPerInputToken = inputCostPerMillionTokens.score / 1e6;
      const tokensPerResponse = aaTokenConsumption.score / BASELINE_AA_TOKEN_CONSUMPTION * BASELINE_TOKENS_PER_OUTPUT;
      // Calculate expected cost per responses.
      const costPerResponse = costPerInputToken * BASELINE_TOKENS_PER_INPUT + costPerOutputToken * tokensPerResponse;

      // Add the new benchmark
      model.benchmarks.push({
        name: 'Cost of 1K responses',
        score: costPerResponse * 1e3,
        source: 'Calculated from ArtificialAnalysis Consumed Tokens',
        stdDev: 0
      });
    }
  });

  return benchmarks;
}

// Main execution: read scores, impute missing benchmarks, and print each model
if (require.main === module) {
  const rawScores = loadScoresSync();
  let benchmarks = adjustScoresByCapabilities(rawScores);
  benchmarks = estimateMissingBenchmarks(benchmarks);
  benchmarks = addCapabilitiesToPrediction(benchmarks, rawScores);
  benchmarks = addCostOf1KResponses(benchmarks);
  //printTable(benchmarks);

  const outputPath = path.join(__dirname, 'data', 'models-prediction.json');
  writePredictionsOutput(benchmarks, outputPath);
}
