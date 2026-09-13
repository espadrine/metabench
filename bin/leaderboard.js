const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { estimateMissingBenchmarks } = require('../lib/score-prediction-weighed-bivariate.js');
const { encodePredictions } = require('../lib/compact-prediction.js');
const Table = require('../lib/cli-table');

// Load the benchmark data from aggregated company model files.
// File format:
// { models: [ { name: string, benchmarks: [ { name: string, score: number }, … ] }, … ] }
function loadScoresSync() {
  const { loadModels } = require('../lib/load-models');
  return loadModels();
}

// Write the predicted scores (with uncertainties) to a compact gzipped JSON file.
// Schema: see lib/compact-prediction.js. Floating numbers rounded to 3 decimals,
// CompactModel is list [name, companyIdx, urlIdx, release_date, capBitmap, BenchTuple[]],
// capabilities is flat list + bitmap after release_date.
function writePredictionsOutput(predictionOutput, filePath) {
  const compact = encodePredictions(predictionOutput);
  const json = JSON.stringify(compact);
  // Preserve Unicode (e.g. τ³) — ensure_ascii=False equivalent: write raw utf8, not escaped.
  // JSON.stringify already preserves Unicode when not escaped.
  const gz = zlib.gzipSync(Buffer.from(json, 'utf8'), { level: 9 });
  fs.writeFileSync(filePath, gz);
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
  const filePath = path.join(__dirname, '..', 'data', 'benchmarks.json');
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

// Mistral Small 3.2 tokens output by AA Intelligence Index Output Tokens per Task
const MISTRAL_AA_TOKEN_OUTPUT = 7146;
// Tokens from sample question "What is the unit of cross-entropy?" given to Mistral Small 3.2
const BASELINE_TOKENS_PER_INPUT = 11;
const BASELINE_TOKENS_PER_OUTPUT = 119;

// Calculate "Cost of 1K chats" benchmark
// Formula: (AA Output Tokens per Task / Mistral's) * (Mistral baseline output) * output price,
// plus the baseline question input tokens * input price.
function addCostOf1KChats(benchmarks) {
  const RESPONSES_PER_K = 1000; // 1K responses

  benchmarks.models.forEach(model => {
    const inputCostPerMillionTokens = model.benchmarks.find(b =>
      b.name === 'Input cost'
    );
    const outputCostPerMillionTokens = model.benchmarks.find(b =>
      b.name === 'Output cost'
    );
    // Find the AA Output Tokens per Task benchmark
    const aaOutputTokensPerTask = model.benchmarks.find(b =>
      b.name === 'AA Output Tokens per Task'
    );

    if (aaOutputTokensPerTask && typeof aaOutputTokensPerTask.score === 'number') {
      const costPerOutputToken = outputCostPerMillionTokens.score / 1e6;
      const costPerInputToken = inputCostPerMillionTokens.score / 1e6;
      const tokensPerResponse = aaOutputTokensPerTask.score / MISTRAL_AA_TOKEN_OUTPUT * BASELINE_TOKENS_PER_OUTPUT;
      // Calculate expected cost per responses.
      const costPerResponse = costPerInputToken * BASELINE_TOKENS_PER_INPUT +
                              costPerOutputToken * tokensPerResponse;

      // Add the new benchmark
      model.benchmarks.push({
        name: 'Cost of 1K chats',
        score: costPerResponse * 1e3,
        source: 'Calculated from AA Output Tokens per Task',
        stdDev: 0
      });
    }
  });

  return benchmarks;
}

// Calculate "Cost of agent" benchmark
// Unlike "Cost of 1K chats" (11 input tokens to 119 output tokens),
// this uses a 10:1 input-to-output token mix:
// Formula: 10 * input price per token + 1 * output price per token.
function addCostOfAgent(benchmarks) {
  const INPUT_TOKENS_PER_MIX = 10;
  const OUTPUT_TOKENS_PER_MIX = 1;

  benchmarks.models.forEach(model => {
    const inputCostPerMillionTokens = model.benchmarks.find(b =>
      b.name === 'Input cost'
    );
    const outputCostPerMillionTokens = model.benchmarks.find(b =>
      b.name === 'Output cost'
    );
    // Find the AA Output Tokens per Task benchmark
    const aaOutputTokensPerTask = model.benchmarks.find(b =>
      b.name === 'AA Output Tokens per Task'
    );

    if (inputCostPerMillionTokens && outputCostPerMillionTokens && aaOutputTokensPerTask &&
        typeof inputCostPerMillionTokens.score === 'number' &&
        typeof outputCostPerMillionTokens.score === 'number' &&
        typeof aaOutputTokensPerTask.score === 'number') {
      const costPerOutputToken = outputCostPerMillionTokens.score / 1e6;
      const costPerInputToken = inputCostPerMillionTokens.score / 1e6;
      const tokensPerResponse = aaOutputTokensPerTask.score / MISTRAL_AA_TOKEN_OUTPUT * OUTPUT_TOKENS_PER_MIX;
      const costPerMix = costPerInputToken * INPUT_TOKENS_PER_MIX +
                         costPerOutputToken * tokensPerResponse;

      // Add the new benchmark
      model.benchmarks.push({
        name: 'Cost of agent',
        score: costPerMix * 1e5,
        source: 'Calculated from Input cost and Output cost',
        stdDev: 0
      });
    }
  });

  return benchmarks;
}

// Calculate "Completion Latency" benchmark
function addCompletionLatency(benchmarks) {
  benchmarks.models.forEach(model => {
    const aaOutputTokensPerTask = model.benchmarks.find(b =>
      b.name === 'AA Output Tokens per Task'
    );
    const outputSpeed = model.benchmarks.find(b => b.name === 'Output speed');
    const activeParams = model.benchmarks.find(b => b.name === 'Active parameters');
    if (aaOutputTokensPerTask && typeof aaOutputTokensPerTask.score === 'number') {
      // How many times more tokens does it consume over the Mistral baseline?
      const consumptionMultiplier = aaOutputTokensPerTask.score / MISTRAL_AA_TOKEN_OUTPUT;
      // How many output tokens would it consume for the baseline question?
      const consumedTokens = BASELINE_TOKENS_PER_OUTPUT * consumptionMultiplier;

      let tokensPerSecond = -1;
      if (outputSpeed && typeof outputSpeed.score === 'number') {
        tokensPerSecond = outputSpeed.score;
      } else if (activeParams && typeof activeParams.score === 'number') {
        // Typical LLM providers intentionally set a high batch size and become CPU-bound,
        // to maximize batch throughput at the expense of individual request latency.
        // However, when we don't know the API speed,
        // let's estimate it on an H100 SXM in memory-bound regime in FP16.
        const memoryBandwidth = 3.35e12; // bytes per second, cf. https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet?ncid=no-ncid
        const weightTransferSize = activeParams.score * 1e9 * 2; // bytes per token (in FP16, 2 bytes per parameter)
        tokensPerSecond = memoryBandwidth / weightTransferSize;
        // Add the output speed.
        model.benchmarks.push({
          name: 'Output speed',
          score: tokensPerSecond,
          source: 'Estimated from Active Parameters and H100 SXM Memory Bandwidth',
          stdDev: 0
        });
      }

      if (tokensPerSecond > 0) {
        const latency = consumedTokens / tokensPerSecond;
        model.benchmarks.push({
          name: 'Completion Latency',
          score: latency,
          source: 'Calculated from AA Output Tokens per Task and Active Parameters',
          stdDev: 0
        });
      }
    }
  });

  return benchmarks;
}

// Estimate Input cost and Output cost per million tokens for models
// where the price benchmarks are missing
// but Active parameters and Output speed are known.
// The estimate is then used by addCostOf1KChats()
// to improve the Cost of 1K chats prediction.
//
// Rationale:
// Having prices be estimated from all other benchmark scores is biased,
// since a cheap model with a high score
// will wrongly be assumed to be an expensive model.
// Fundamentally, provider price per token is driven by compute cost,
// as both reads (compute-bound from KV cache computation)
// and writes (not memory-IO-bound but attention-bound
// because of high batch sizes) are compute-bound.
// Compute is proportional to the number of active parameters,
// and to the speed of attention.
// Indeed DeepSeek’s MLA / DSA changes reduce attention FLOPs,
// which let them lower prices,
// and shows up as higher Output speed for the same active parameter count.
// Hence the intensity ratio activeB / speed captures architectural
// efficiency beyond raw parameter count.
//
// Empirical validation on 75 models with both price and speed:
// - Baseline activeB only
//   Input:  Input = 0.399 + 0.00549·activeB   R²=0.164 RMSE=0.650
//   Output: Output = 1.468 + 0.01708·activeB  R²=0.121 RMSE=2.410
//
// - activeB + Output speed
//   Input:  Input = 0.496 + 0.00520·activeB -0.000784·speed   R²=0.183 RMSE=0.643
//   Output: Output = 1.824 + 0.01602·activeB -0.00288·speed  R²=0.140 RMSE=2.383
//   Small improvement; speed alone is weakly correlated with price.
//
// - activeB + ratio = activeB / speed
//   Input:  Input = 0.370 -0.00471·activeB + 0.738·ratio   R²=0.255 RMSE=0.613
//   Output: Output = 1.323 -0.03356·activeB + 3.662·ratio   R²=0.292 RMSE=2.163
//   ~+9% R² for Input and +17% R² for Output vs baseline, RMSE down ~6% and ~10% respectively.
function addInputOutputCost(benchmarks) {
  benchmarks.models.forEach(model => {
    const active = model.benchmarks.find(b => b.name === 'Active parameters');
    const hasActive = active && typeof active.score === 'number';
    if (!hasActive) { return; }
    const activeB = active.score;
    const outSpeed = model.benchmarks.find(b => b.name === 'Output speed');
    const hasSpeed = outSpeed && typeof outSpeed.score === 'number' && outSpeed.score > 0;
    const speed = hasSpeed ? outSpeed.score : null;
    const ratio = hasSpeed ? activeB / speed : null;

    // Estimate Input cost per million tokens
    const inputBench = model.benchmarks.find(b => b.name === 'Input cost');
    const hasInputCost = inputBench && typeof inputBench.score === 'number';
    if (!hasInputCost) {
      let estimatedInput;
      let source;
      if (hasSpeed) {
        estimatedInput = 0.370 - 0.00471 * activeB + 0.738 * ratio;
        source = 'Estimated from Active parameters + Output speed ratio';
      } else {
        // Fallback to activeB-only regression from 75-model set
        estimatedInput = 0.399 + 0.00549 * activeB;
        source = 'Estimated from Active parameters';
      }
      model.benchmarks.push({
        name: 'Input cost',
        score: estimatedInput,
        source,
        stdDev: 0
      });
    }

    // Estimate Output cost per million tokens
    const outputBench = model.benchmarks.find(b => b.name === 'Output cost');
    const hasOutputCost = outputBench && typeof outputBench.score === 'number';
    if (!hasOutputCost) {
      let estimatedOutput;
      let source;
      if (hasSpeed) {
        estimatedOutput = 1.323 - 0.03356 * activeB + 3.662 * ratio;
        source = 'Estimated from Active parameters + Output speed ratio';
      } else {
        // Fallback to activeB-only regression from 75-model set
        estimatedOutput = 1.468 + 0.01708 * activeB;
        source = 'Estimated from Active parameters';
      }
      model.benchmarks.push({
        name: 'Output cost',
        score: estimatedOutput,
        source,
        stdDev: 0
      });
    }
  });
  return benchmarks;
}

// Add timestamp benchmark to all models
function addTimestampBenchmark(benchmarks) {
  benchmarks.models.forEach(model => {
    // Convert release_date to floating-point year
    const date = new Date(model.release_date);
    const year = date.getUTCFullYear();
    const startOfYear = new Date(Date.UTC(year, 0, 1));
    const endOfYear = new Date(Date.UTC(year + 1, 0, 1));
    const yearFraction = (date - startOfYear) / (endOfYear - startOfYear);
    const yearValue = year + yearFraction;

    model.benchmarks.push({
      name: 'Release date',
      score: yearValue,
      source: model.url,
    });
  });

  return benchmarks;
}

// Add synthetic benchmark: Release Date × log(Size)
function addReleaseDateSizeProduct(benchmarks) {
  benchmarks.models.forEach(model => {
    // Find Release date and Size benchmarks
    const releaseDateBench = model.benchmarks.find(b => b.name === 'Release date');
    const sizeBench = model.benchmarks.find(b => b.name === 'Size');

    // Only add if both benchmarks are present
    if (releaseDateBench && sizeBench && typeof releaseDateBench.score === 'number' && typeof sizeBench.score === 'number') {
      // Calculate log(Size) using natural logarithm, then multiply by Release Date
      const logSize = Math.log(sizeBench.score);
      const product = releaseDateBench.score * logSize;

      model.benchmarks.push({
        name: 'Release date × log(Size)',
        score: product,
        source: model.url,
      });
    }
  });

  return benchmarks;
}

function computeBenchmarks() {
  const rawScores = loadScoresSync();
  let benchmarks = addTimestampBenchmark(rawScores);
  benchmarks = addReleaseDateSizeProduct(benchmarks);
  benchmarks = adjustScoresByCapabilities(benchmarks);
  benchmarks = addCompletionLatency(benchmarks);
  benchmarks = addInputOutputCost(benchmarks);  // Depends on addCompletionLatency()
  benchmarks = estimateMissingBenchmarks(benchmarks);
  benchmarks = addCapabilitiesToPrediction(benchmarks, rawScores);
  benchmarks = addCostOf1KChats(benchmarks);  // Depends on addInputOutputCost()
  benchmarks = addCostOfAgent(benchmarks);  // Depends on addInputOutputCost()
  return benchmarks;
}

// Main execution: read scores, impute missing benchmarks, and print each model
if (require.main === module) {
  const { models, estimators } = computeBenchmarks();
  const benchmarks = { models };
  //printTable(benchmarks);

  const outputPath = path.join(__dirname, '..', 'data', 'models-prediction.json.gz');
  writePredictionsOutput(benchmarks, outputPath);
}

module.exports = {
  computeBenchmarks,
};
