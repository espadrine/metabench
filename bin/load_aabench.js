// Load benchmark data from Artificial Analysis and match it against our models.

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const crypto = require('crypto');
const zlib = require('zlib');
const { normalizeModelName, isUnambiguousModelMatch, levenshteinDistance } = require('../lib/load-bench');
const { loadModels } = require('../lib/load-models');

function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const verbose = args.includes('--verbose') || args.includes('-v');

  const aaBenchData = loadAABenchData("./data/aabench.json");
  const models = loadModels();

  // Match AA models with our data models
  const modelMatches = matchAABenchmarks(aaBenchData, models);

  // Filter matches using simple predicate functions
  const unambiguousModels = modelMatches.filter(isUnambiguousMatch);
  const ambiguousModels = modelMatches.filter(m => !isUnambiguousMatch(m));

  // Log summary
  logMatchSummary(modelMatches, unambiguousModels, ambiguousModels);

  // Update unambiguous matches
  updateUnambiguousModels(unambiguousModels, models, verbose);

  // Store ambiguous/unmatched models
  storeMissingBenchmarks(ambiguousModels, "./data/missing_aabench_benchmarks.json");
}

// Match AA benchmarks with our data models and return match information
// Returns an array of match objects, each containing:
// {
//   aaModel: model object from the AA data, // Original AA model data
//   dataModel: model object or null,         // Matched model from our data, or null if no match
//   benchmarks: [{name, score, source}]     // The AA benchmarks to add/update
// }
function matchAABenchmarks(aaBenchData, models) {
  // First, create a map of AA models to our models using the sophisticated mapping algorithm
  const modelMap = mapModels(aaBenchData, models);

  const matches = [];

  // Look through each model in `aaBenchData`.
  for (const aaModel of aaBenchData.models) {
    if (aaModelsToIgnore.includes(aaModel.name)) {
      continue;
    }

    // Get the mapped model from our model map
    const model = modelMap[aaModel.name];

    // Merge the score fields and the output speed into a single object for processing
    const allBenchmarks = {
      intelligenceIndex: aaModel.intelligenceIndex,
      gpqa: aaModel.gpqa,
      hle: aaModel.hle,
      livecodebench: aaModel.livecodebench,
      scicode: aaModel.scicode,
      aime25: aaModel.aime25,
      ifbench: aaModel.ifbench,
      lcr: aaModel.lcr,
      terminalbenchHard: aaModel.terminalbenchHard,
      terminalbenchV21: aaModel.terminalbenchV21,
      tau2: aaModel.tau2,
      tauBanking: aaModel.tauBanking,
      mmmuPro: aaModel.mmmuPro,
      critpt: aaModel.critpt,
      apexAgents: aaModel.apexAgents,
      omniscience: aaModel.omniscience,
      gdpval: aaModel.gdpval,
      automationBenchPartialScore: aaModel.automationBenchPartialScore,
      analystAgent: aaModel.analystAgent,
      enterpriseOpsGym: aaModel.enterpriseOpsGym,
      itBenchSre: aaModel.itBenchSre,
      agenticIndex: aaModel.agenticIndex,
      price1mInputTokens: aaModel.price1mInputTokens,
      price1mOutputTokens: aaModel.price1mOutputTokens,
      medianOutputSpeed: aaModel.timescaleData?.medianOutputSpeed,
      intelligenceIndexOutputTokensPerTask: aaModel.intelligenceIndexOutputTokensPerTask?.output,
    };

    // Create match object
    const match = {
      aaModel: aaModel,
      dataModel: model,
      benchmarks: []
    };

    // For each AA benchmark for that model,
    // check if it is useful and should be processed
    for (const [aaBenchName, score] of Object.entries(allBenchmarks)) {
      const mappedBenchName = benchNameFromAA[aaBenchName] || benchNameFromAAPricing[aaBenchName] || benchNameFromAATopLevel[aaBenchName];
      if (mappedBenchName && typeof score === 'number') {
        // Check if benchmark is useful. When zero or null, it is not useful.
        const isUseful = score !== 0 && score != null;

        // Check if benchmark should be excluded
        const isExcluded = excludedBenchmarks.includes(mappedBenchName);

        if (isUseful && !isExcluded) {
          // Only scale benchmarks that are on 0-1 scale (not index benchmarks)
          const finalScore = scoreFromAAScore(score, aaBenchName);
          match.benchmarks.push({
            name: mappedBenchName,
            score: Math.round(finalScore * 100) / 100, // Round to 2 decimal places
            source: "https://artificialanalysis.ai/api/v2/data/llms/models"
          });
        }
      }
    }

    if (match.benchmarks.length > 0) {
      matches.push(match);
    }
  }

  return matches;
}

// Map AA models to our data models using the same sophisticated algorithm as LMArena
// Return a map from AA model name to our model
function mapModels(aaBenchData, models) {
  // 1. Compute the levenshtein distance for each possible mapping.
  // We create a list of {aaModelName, modelName, distance}.
  const modelMappings = [];
  for (const aaModel of aaBenchData.models) {
    if (aaModelsToIgnore.includes(aaModel.name)) {
      continue;
    }

    const aaNameNormalized = normalizeModelName(aaModel.name);
    for (const model of models.models) {
      const modelNameNormalized = normalizeModelName(model.name);
      const distance = levenshteinDistance(aaNameNormalized, modelNameNormalized);
      modelMappings.push({aaModelName: aaModel.name, modelName: model.name, distance});
    }
  }

  // 2. Assign known mappings.
  const modelMap = {};
  for (const aaModel of aaBenchData.models) {
    if (aaModelsToIgnore.includes(aaModel.name)) {
      continue;
    }

    const knownMappingName = modelNameFromAA[aaModel.name];
    if (knownMappingName) {
      const model = models.models.find(m => m.name === knownMappingName);
      if (model) {
        modelMap[aaModel.name] = model;
      }
    }
  }

  // 3. Assign unambiguous mappings.
  for (const aaModel of aaBenchData.models) {
    if (aaModelsToIgnore.includes(aaModel.name)) {
      continue;
    }

    for (const model of models.models) {
      const notAlreadyMapped = !modelMap[aaModel.name];
      if (isUnambiguousModelMatch(aaModel, model, modelNameFromAA) && notAlreadyMapped) {
        modelMap[aaModel.name] = model;
      }
    }
  }

  // 4. Assign the mapping with the best levenshtein match, then iterate mappings.
  const sortedModelMappings = modelMappings.sort((a, b) => a.distance - b.distance);
  const assignedModels = new Set();
  for (const mapping of sortedModelMappings) {
    const aaModelName = mapping.aaModelName;
    const modelName = mapping.modelName;

    // If these models are already mapped, skip.
    if (modelMap[aaModelName] || assignedModels.has(modelName)) {
      continue;
    }

    // Assign the mapping
    const model = models.models.find(m => m.name === modelName);
    modelMap[aaModelName] = model;
  }

  return modelMap;
}

// Check if a match is unambiguous (for auto-update)
// Uses the shared isUnambiguousModelMatch function
function isUnambiguousMatch(match) {
  if (!match.dataModel) {
    return false;
  }

  return isUnambiguousModelMatch(match.aaModel, match.dataModel, modelNameFromAA);
}

// Update model files with new AA benchmarks for unambiguous matches
function updateUnambiguousModels(unambiguousModels, models, verbose = false) {
  if (unambiguousModels.length === 0) {
    console.error('No unambiguous model matches to update.');
    return;
  }

  let updatedCount = 0;

  for (const match of unambiguousModels) {
    const modelName = match.dataModel.name;
    const filePath = findModelFilePath(modelName, models);

    if (!filePath || !fs.existsSync(filePath)) {
      console.error(`⚠️  Model file not found for ${modelName}: ${filePath}`);
      continue;
    }

    // Read the existing model file
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const modelData = JSON.parse(fileContent);

    // Find the specific model in the file
    const modelToUpdate = modelData.models.find(m => m.name === modelName);

    if (!modelToUpdate) {
      console.error(`⚠️  Model ${modelName} not found in file ${filePath}`);
      continue;
    }

    // Process each benchmark for this model
    for (const benchmark of match.benchmarks) {
      // Check if benchmark already exists
      const existingBenchmarkIndex = modelToUpdate.benchmarks.findIndex(
        b => b.name === benchmark.name && b.source === benchmark.source
      );

      if (existingBenchmarkIndex >= 0) {
        // Benchmark exists - update it if the score is different
        const existingBenchmark = modelToUpdate.benchmarks[existingBenchmarkIndex];
        if (existingBenchmark.score !== benchmark.score) {
          modelToUpdate.benchmarks[existingBenchmarkIndex] = benchmark;
          console.error(`🔄 Updated existing benchmark for ${modelName} (${match.aaModel.name}): ${existingBenchmark.score} → ${benchmark.score}`);
        } else {
          if (verbose) {
            console.error(`ℹ️  Benchmark already exists for ${modelName} (${match.aaModel.name}) with same score (${benchmark.score}), no update needed`);
          }
          continue;
        }
      } else {
        // Add the new benchmark
        modelToUpdate.benchmarks.push(benchmark);
        console.error(`✅ Added new benchmark for ${modelName} (${match.aaModel.name}): ${benchmark.score}`);
      }
    }

    // Write the updated data back to the file
    fs.writeFileSync(filePath, JSON.stringify(modelData, null, 2), 'utf8');
    updatedCount++;
  }

  console.error(`📊 Successfully updated ${updatedCount} model files with AA benchmarks`);
}

// Find the file path for a given model name by searching through all model files
function findModelFilePath(modelName, models) {
  // First, try to find the model in the loaded models
  const model = models.models.find(m => m.name === modelName);

  if (!model) {
    return null;
  }

  // Get the company name and try to find the corresponding file
  const company = model.company || 'unknown';
  // Handle special cases like "Z.ai" -> "zai"
  const companyFileName = company.toLowerCase().replace(/\s+/g, '').replace(/\./g, '');
  const filePath = `./data/models/${companyFileName}.json`;

  // Check if the file exists
  if (fs.existsSync(filePath)) {
    return filePath;
  }

  // If company-based file doesn't exist, search through all model files
  const modelsDir = './data/models/';
  const files = fs.readdirSync(modelsDir);
  for (const file of files) {
    if (file.endsWith('.json')) {
      const fullPath = path.join(modelsDir, file);
      const content = fs.readFileSync(fullPath, 'utf8');
      const data = JSON.parse(content);

      // Check if this model is in this file
      if (data.models && data.models.some(m => m.name === modelName)) {
        return fullPath;
      }
    }
  }

  return null;
}

// Log summary of the matching results
function logMatchSummary(modelMatches, unambiguousModels, ambiguousModels) {
  const existingBenchmarks = modelMatches.filter(m =>
    m.dataModel &&
    m.benchmarks.some(b =>
      m.dataModel.benchmarks &&
      m.dataModel.benchmarks.some(existingB =>
        existingB.name === b.name && existingB.source === b.source
      )
    )
  ).length;

  console.error(`📊 Summary:`);
  console.error(`   Total AA models processed: ${modelMatches.length}`);
  console.error(`   Unambiguous matches (will auto-update/add): ${unambiguousModels.length}`);
  console.error(`   Ambiguous/unmatched (need manual review): ${ambiguousModels.length}`);
  console.error(`   Existing benchmarks (will be updated if score changed): ${existingBenchmarks}`);
}

const aaModelsToIgnore = [
  // Will import later:
  "GPT-3.5 Turbo",
  "GPT-4.5 (Preview)",
  "GPT-4o (Aug '24)",
  "GPT-4o (Nov '24)",
  "GPT-4o mini",
  "GPT-5.1 Codex mini (high)",
  "GPT-5.2 (medium)",
  "GPT-5.2 (Non-reasoning)",
  "GPT-5.2 Codex (xhigh)",
  "Mistral Large 2 (Nov '24)",
  "Mistral Small 3.2",
  "Claude 2.1",
  "Claude 4 Sonnet (Non-reasoning)",
  "Claude 4.5 Haiku (Non-reasoning)",
  "Claude Instant",
  "Gemini 1.5 Flash (Sep '24)",
  "Gemini 1.5 Pro (Sep '24)",
  "Gemini 2.0 Flash (Feb '25)",
  "Gemini 2.0 Flash-Lite (Feb '25)",
  "Gemini 2.5 Flash (Non-reasoning)",
  "Gemini 2.5 Flash-Lite (Non-reasoning)",
  "Gemini 2.5 Flash-Lite (Reasoning)",
  "Gemma 3n E2B Instruct",
  "Gemma 3n E4B Instruct",
  "DeepSeek V3.1 (Non-reasoning)",
  "DeepSeek V3.1 Terminus (Non-reasoning)",
  "DeepSeek V3.1 Terminus (Reasoning)",
  "DeepSeek V3.2 (Non-reasoning)",
  "DeepSeek V3.2 Exp (Non-reasoning)",
  "DeepSeek-Coder-V2",
  "DeepSeek-V2-Chat",
  "DeepSeek-V2.5",
  "DeepSeek-V2.5 (Dec '24)",
  "GLM-4.5V (Non-reasoning)",
  "GLM-4.5V (Reasoning)",
  "GLM-4.6 (Non-reasoning)",
  "GLM-4.6V (Non-reasoning)",
  "GLM-4.6V (Reasoning)",
  "GLM-4.7 (Non-reasoning)",
  "Doubao Seed Code",
  "Doubao-Seed-1.8",
  "ERNIE 4.5 300B A47B",
  "Kimi K2.5 (Non-reasoning)",
  // Don't care, too edge-case, not relevant anymore:
  "Gemini 2.5 Flash Preview (Sep '25) (Non-reasoning)",
  "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
  "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
  "Gemini 2.5 Flash-Lite Preview (Sep '25) (Non-reasoning)",
  "DeepSeek R1 Distill Llama 70B",
  "DeepSeek R1 Distill Llama 8B",
  "DeepSeek R1 Distill Qwen 1.5B",
  "DeepSeek R1 Distill Qwen 14B",
  "DeepSeek R1 Distill Qwen 32B",
  "DeepSeek Coder V2 Lite Instruct",
  "DeepSeek LLM 67B Chat (V1)",
  "DeepSeek-OCR",
  "Llama 3.1 Tulu3 405B",
  "Hermes 3 - Llama-3.1 70B",
  "Llama 3.1 Nemotron Instruct 70B",
  "Llama 3.1 Nemotron Ultra 253B v1 (Reasoning)",
  "Llama 3.1 Nemotron Nano 4B v1.1 (Reasoning)",
  "Llama 3.3 Nemotron Super 49B v1 (Non-reasoning)",
  "Llama Nemotron Super 49B v1.5 (Non-reasoning)",
  "Llama 3.3 Nemotron Super 49B v1 (Reasoning)",
  "Llama Nemotron Super 49B v1.5 (Reasoning)",
  "DeepHermes 3 - Llama-3.1 8B Preview (Non-reasoning)",
  "Hermes 4 - Llama-3.1 405B (Non-reasoning)",
  "Hermes 4 - Llama-3.1 405B (Reasoning)",
  "Hermes 4 - Llama-3.1 70B (Non-reasoning)",
  "Hermes 4 - Llama-3.1 70B (Reasoning)",
  "DeepHermes 3 - Mistral 24B Preview (Non-reasoning)",
  "DBRX Instruct",
  "Mistral Saba",
  "Apriel-v1.5-15B-Thinker",
  "Apriel-v1.6-15B-Thinker",
  "Arctic Instruct",
  "Cogito v2.1 (Reasoning)",
  "Exaone 4.0 1.2B (Non-reasoning)",
  "Exaone 4.0 1.2B (Reasoning)",
  "EXAONE 4.0 32B (Non-reasoning)",
  "EXAONE 4.0 32B (Reasoning)",
  "Falcon-H1R-7B",
  "Gemini 2.0 Flash (experimental)",
  "Gemini 2.0 Flash-Lite (Preview)",
  "Gemini 2.0 Flash Thinking Experimental (Dec '24)",
  "Gemini 2.0 Flash Thinking Experimental (Jan '25)",
  "Gemini 2.0 Pro Experimental (Feb '25)",
  "Gemini 2.5 Flash Preview (Non-reasoning)",
  "Gemini 2.5 Flash Preview (Reasoning)",
  "Gemini 2.5 Pro Preview (Mar' 25)",
  "Gemini 2.5 Pro Preview (May' 25)",
  "Gemma 3n E4B Instruct Preview (May '25)",
  "Granite 3.3 8B (Non-reasoning)",
  "Granite 4.0 1B",
  "Granite 4.0 350M",
  "Granite 4.0 H 1B",
  "Granite 4.0 H 350M",
  "Granite 4.0 H Small",
  "Granite 4.0 Micro",
  // Other reasons
  "GPT-4o (ChatGPT)",
  "GPT-4o (March 2025, chatgpt-4o-latest)",
  "GPT-5 (ChatGPT)",  // Don't study chat constructs for now.
];

function equalEpsilon(a, b, epsilon = 0.0001) {
  return Math.abs(a - b) < epsilon;
}

// - The aaModelName is a string from AA data.
// - models is the raw data from data/models/ company model files
// Return the model from `models` that best matches `aaModelName`,
// or null if no good match is found.
function findModel(aaModelName, models) {
  if (modelNameFromAA[aaModelName] != null) {
    aaModelName = modelNameFromAA[aaModelName];
  }

  // Lowercase the model names for comparison.
  const aaNameLower = aaModelName.toLowerCase();
  let bestMatch = null;
  let bestDistance = Infinity;

  for (const model of models.models) {
    const modelNameLower = model.name.toLowerCase();
    const distance = levenshteinDistance(aaNameLower, modelNameLower);

    // If the distance is too high (more than 30% of the length of the AA model name),
    // skip this match
    const maxAllowedDistance = aaNameLower.length * 0.3;

    if (distance < bestDistance && distance <= maxAllowedDistance) {
      bestDistance = distance;
      bestMatch = model;
    }
  }

  return bestMatch;
}

const modelNameFromAA = {
  // AABench name: Our data name
  // Append the latest at the top.
  "Grok 4.7 (xhigh)": "Grok 4.7",
  "DeepSeek V4.1 Flash (Reasoning, Max Effort)": "DeepSeek V4.1 Flash",
  "GPT-6 Astra (max)": "GPT-6 Astra",
  "GPT-6 Astra (xhigh)": "GPT-6 Astra xhigh",
  "GPT-6 Astra (high)": "GPT-6 Astra high",
  "GPT-6 Astra (medium)": "GPT-6 Astra medium",
  "GPT-6 Astra (low)": "GPT-6 Astra low",
  "Gemini 3.8 Flash (high)": "Gemini 3.8 Flash",
  "Gemini 3.8 Flash (medium)": "Gemini 3.8 Flash medium",
  "Gemini 3.8 Flash (low)": "Gemini 3.8 Flash low",
  "Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)": "Claude Fable 5.1",
  "Muse Spark 1.3 (max)": "Muse Spark 1.3",
  "Qwen3.8-Flash-Next": "Qwen3.8-Flash-Next",
  "GLM-5.3-Flash": "GLM-5.3-Flash",
  "DeepSeek V4 Flash Vision (Reasoning, Max Effort)": "DeepSeek-V4-Vision-Exp",
  "LFM2.5-2.6B": "LFM2.5-2.6B",
  "GLM-5.3 (max)": "GLM-5.3",
  "Qwen3.8 27B (xhigh)": "Qwen3.8-27B",
  "Qwen3.8 27B (medium)": "Qwen3.8-27B medium",
  "Qwen3.8 27B (low)": "Qwen3.8-27B low",
  "Qwen3.8 27B (Non-reasoning)": "Qwen3.8-27B none",
  "Grok 4.6 (high)": "Grok 4.6",
  "Gemini 3.7 Flash (high)": "Gemini 3.7 Flash",
  "Gemini 3.7 Flash (medium)": "Gemini 3.7 Flash medium",
  "Gemini 3.7 Flash (low)": "Gemini 3.7 Flash low",
  "DeepSeek V4 Pro 0813 (Reasoning, Max Effort)": "DeepSeek V4 Pro Max 0813",
  "Qwen3.8 Max": "Qwen3.8-Max",
  "Olmo 3.1 32B Think": "Olmo 3.1 32B Think",
  "Olmo 3.1 32B Instruct": "Olmo 3.1 32B Instruct",
  "Olmo 3 32B Think": "Olmo 3 32B Think",
  "Olmo 3 7B Think": "Olmo 3 7B Think",
  "Olmo 3 7B Instruct": "Olmo 3 7B Instruct",
  "Muse Glimmer (high)": "Muse Glimmer",
  "Inkling Small": "Inkling Small",
  "Inkling (xhigh)": "Inkling",
  "LFM2.5-8B-A1B": "LFM2.5-8B-A1B",
  "o3": "o3 (high)",
  "gpt-oss-120b (high)": "gpt-oss-120b High",
  "gpt-oss-120b (low)": "gpt-oss-120b Low",
  "Command A+": "Command A+",
  "Command A": "Command-A",
  "Command-R (Mar '24)": "Command-R",
  "Command-R+ (Apr '24)": "Command-R+",
  "Claude 2.0": "Claude 2",
  "Claude 3.5 Sonnet (Oct '24)": "Claude 3.5 Sonnet (new)",
  "Claude 3.5 Sonnet (June '24)": "Claude 3.5 Sonnet",
  "Claude 3.7 Sonnet (Non-reasoning)": "Claude Sonnet 3.7",
  "Claude 3.7 Sonnet (Reasoning)": "Claude Sonnet 3.7 Thinking",
  "Claude 4 Opus (Non-reasoning)": "Claude Opus 4",
  "Claude 4 Opus (Reasoning)": "Claude Opus 4 Thinking",
  "Claude 4 Sonnet (Reasoning)": "Claude Sonnet 4 Thinking",
  "Claude 4.1 Opus (Non-reasoning)": "Claude Opus 4.1",
  "Claude 4.1 Opus (Reasoning)": "Claude Opus 4.1 Thinking",
  "Claude 4.5 Haiku (Non-reasoning)": "Claude Haiku 4.5",
  "Claude 4.5 Haiku (Reasoning)": "Claude Haiku 4.5 Thinking",
  "Claude 4.5 Sonnet (Non-reasoning)": "Claude Sonnet 4.5",
  "Claude 4.5 Sonnet (Reasoning)": "Claude Sonnet 4.5 Thinking",
  "Claude Sonnet 4.6 (Adaptive Reasoning, Max Effort)": "Claude Sonnet 4.6 Thinking",
  "Claude Sonnet 4.6 (Non-reasoning, High Effort)": "Claude Sonnet 4.6",
  "Claude Opus 4.5 (Non-reasoning)": "Claude Opus 4.5",
  "Claude Opus 4.5 (Reasoning)": "Claude Opus 4.5 Thinking",
  "Claude Opus 4.6 (Non-reasoning, High Effort)": "Claude Opus 4.6",
  "Claude Opus 4.6 (Adaptive Reasoning, Max Effort)": "Claude Opus 4.6 Thinking",
  "Claude Opus 4.7 (Non-reasoning, High Effort)": "Claude Opus 4.7",
  "Claude Opus 4.7 (Adaptive Reasoning, Max Effort)": "Claude Opus 4.7 Thinking",
  "Claude Opus 4.8 (Adaptive Reasoning, Max Effort)": "Claude Opus 4.8 Thinking",
  "Claude Fable 5 (Adaptive Reasoning, Max Effort, Opus 4.8 Fallback)": "Claude Fable 5 Thinking",
  "Claude Sonnet 5 (Adaptive Reasoning, Max Effort)": "Claude Sonnet 5 Thinking",
  "Claude Opus 5 (Adaptive Reasoning, Low Effort)": "Claude Opus 5 low",
  "Claude Opus 5 (Adaptive Reasoning, Medium Effort)": "Claude Opus 5 medium",
  "Claude Opus 5 (Adaptive Reasoning, High Effort)": "Claude Opus 5 high",
  "Claude Opus 5 (Adaptive Reasoning, Xhigh Effort)": "Claude Opus 5 xhigh",
  "Claude Opus 5 (Adaptive Reasoning, Max Effort)": "Claude Opus 5 max",
  "DeepSeek R1 (Jan '25)": "DeepSeek R1",
  "DeepSeek R1 0528 (May '25)": "DeepSeek R1 0528",
  "DeepSeek V3 (Dec '24)": "DeepSeek V3",
  "DeepSeek V3.1 (Reasoning)": "DeepSeek V3.1",
  "DeepSeek V3.2 (Reasoning)": "DeepSeek V3.2",
  "DeepSeek V3.2 Exp (Reasoning)": "DeepSeek V3.2 Exp",
  "DeepSeek V4 Pro (Non-reasoning)": "DeepSeek V4 Pro Non-Think",
  "DeepSeek V4 Pro (Reasoning, High Effort)": "DeepSeek V4 Pro",
  "DeepSeek V4 Pro (Reasoning, Max Effort)": "DeepSeek V4 Pro Max",
  "DeepSeek V4 Flash 0731 (Reasoning, Max Effort)": "DeepSeek V4 Flash Max 0731",
  "DeepSeek V4 Flash (Non-reasoning)": "DeepSeek V4 Flash Non-Think",
  "DeepSeek V4 Flash (Reasoning, High Effort)": "DeepSeek V4 Flash",
  "DeepSeek V4 Flash (Reasoning, Max Effort)": "DeepSeek V4 Flash Max",
  "Mistral Small 4 (Reasoning)": "Mistral Small 4 Reasoning",
  "Mistral Small 4 (Non-reasoning)": "Mistral Small 4 Instruct",
  "Devstral 2": "Devstral 2 123B",
  "Devstral Medium": "Devstral Medium 1",
  "Devstral Small (Jul '25)": "Devstral Small 1.1",
  "Devstral Small (May '25)": "Devstral Small 1.0",
  "Devstral Small 2": "Devstral Small 2 24B",
  "Magistral Medium 1": "Magistral Medium 1.0",
  "Magistral Small 1": "Magistral Small 1.0",
  "Ministral 3 14B": "Ministral 3 14B Instruct",
  "Ministral 3 8B": "Ministral 3 8B Instruct",
  "Ministral 3 3B": "Ministral 3 3B Instruct",
  "Mistral 7B Instruct": "Mistral 7B",
  "Mistral Large (Feb '24)": "Mistral Large 1",
  "Mistral Large 2 (Jul '24)": "Mistral Large 2",
  "Mistral Medium": "Mistral Medium 1",
  "Mistral Medium 3.5": "Mistral Medium 3.5",
  "Mistral Small (Feb '24)": "Mistral Small 1 2402",
  "Mistral Small (Sep '24)": "Mistral Small 2 2409",
  "Mixtral 8x22B Instruct": "Mixtral 8x22B",
  "Mixtral 8x7B Instruct": "Mixtral 8x7B",
  "Gemini 1.5 Flash (May '24)": "Gemini 1.5 Flash",
  "Gemini 1.5 Pro (May '24)": "Gemini 1.5 Pro",
  "Gemini 2.0 Flash Thinking Experimental (Dec '24)": "Gemini 2.0 Flash",
  "Gemini 2.5 Flash (Reasoning)": "Gemini 2.5 Flash Thinking 0520",
  "Gemini 3 Pro Preview (high)": "Gemini 3 Pro",
  "Gemini 3 Pro Preview (low)": "Gemini 3 Pro Low",
  "Gemini 3 Flash Preview (Reasoning)": "Gemini 3 Flash",
  "Gemini 3 Flash Preview (Non-reasoning)": "Gemini 3 Flash Low",
  "Gemini 3.1 Pro Preview": "Gemini 3.1 Pro",
  "Gemini 3.1 Flash-Lite": "Gemini 3.1 Flash-Lite",
  "Gemini 3.5 Flash (high)": "Gemini 3.5 Flash high",
  "Gemini 3.5 Flash (medium)": "Gemini 3.5 Flash",
  "Gemini 3.5 Flash (minimal)": "Gemini 3.5 Flash minimal",
  "Gemini 3.5 Flash-Lite": "Gemini 3.5 Flash-Lite",
  "Gemini 3.6 Flash (high)": "Gemini 3.6 Flash",
  "Gemma 4 26B A4B (Reasoning)": "Gemma 4 26B-A4B",
  "Gemma 4 31B (Reasoning)": "Gemma 4 31B",
  "Gemma 4 E2B (Reasoning)": "Gemma 4 E2B",
  "Gemma 4 E4B (Reasoning)": "Gemma 4 E4B",
  "DiffusionGemma 26B A4B": "DiffusionGemma",
  "GLM-4.5 (Reasoning)": "GLM-4.5",
  "GLM-4.6 (Reasoning)": "GLM-4.6",
  "GLM-4.7 (Reasoning)": "GLM-4.7",
  "GLM-4.7-Flash (Reasoning)": "GLM-4.7 Flash",
  "GLM-5 (Reasoning)": "GLM-5",
  "GLM-5.1 (Reasoning)": "GLM-5.1",
  "GLM-5.2 (max)": "GLM-5.2",
  "GPT-4": "GPT-4",
  "GPT-4o (May '24)": "GPT-4o",
  "gpt-oss-20b (high)": "gpt-oss-20b High",
  "gpt-oss-20b (low)": "gpt-oss-20b Low",
  "GPT-5 (high)": "GPT-5 High",
  "GPT-5 (low)": "GPT-5 Low",
  "GPT-5 (medium)": "GPT-5 Medium",
  "GPT-5 (minimal)": "GPT-5 Minimal",
  "GPT-5 Codex (high)": "GPT-5 Codex High",
  "GPT-5 mini (high)": "GPT-5 mini High",
  "GPT-5 mini (medium)": "GPT-5 mini Medium",
  "GPT-5 mini (minimal)": "GPT-5 mini Minimal",
  "GPT-5 nano (high)": "GPT-5 nano High",
  "GPT-5 nano (medium)": "GPT-5 nano Medium",
  "GPT-5 nano (minimal)": "GPT-5 nano Minimal",
  "GPT-5.1 (Non-reasoning)": "GPT-5.1 None",
  "GPT-5.1 (high)": "GPT-5.1 High",
  "GPT-5.1 Codex (high)": "GPT-5 Codex",
  "GPT-5.2 (xhigh)": "GPT-5.2 xhigh",
  "GPT-5.3 Codex (xhigh)": "GPT-5.3 Codex",
  "GPT-5.4 (xhigh)": "GPT-5.4",
  "GPT-5.4 Pro (xhigh)": "GPT-5.4 Pro",
  "GPT-5.4 (low)": "GPT-5.4 low",
  "GPT-5.5 (Non-reasoning)": "GPT-5.5 none",
  "GPT-5.5 (low)": "GPT-5.5 low",
  "GPT-5.5 (medium)": "GPT-5.5 medium",
  "GPT-5.5 (high)": "GPT-5.5 high",
  "GPT-5.5 (xhigh)": "GPT-5.5 xhigh",
  "GPT-5.5 Pro (xhigh)": "GPT-5.5 Pro",
  "GPT-5.6 Luna (Non-reasoning)": "GPT-5.6 Luna none",
  "GPT-5.6 Luna (low)": "GPT-5.6 Luna low",
  "GPT-5.6 Luna (medium)": "GPT-5.6 Luna medium",
  "GPT-5.6 Luna (high)": "GPT-5.6 Luna high",
  "GPT-5.6 Luna (xhigh)": "GPT-5.6 Luna xhigh",
  "GPT-5.6 Luna (max)": "GPT-5.6 Luna max",
  "GPT-5.6 Terra (Non-reasoning)": "GPT-5.6 Terra none",
  "GPT-5.6 Terra (low)": "GPT-5.6 Terra low",
  "GPT-5.6 Terra (medium)": "GPT-5.6 Terra medium",
  "GPT-5.6 Terra (high)": "GPT-5.6 Terra high",
  "GPT-5.6 Terra (xhigh)": "GPT-5.6 Terra xhigh",
  "GPT-5.6 Terra (max)": "GPT-5.6 Terra max",
  "GPT-5.6 Sol (Non-reasoning)": "GPT-5.6 Sol none",
  "GPT-5.6 Sol (low)": "GPT-5.6 Sol low",
  "GPT-5.6 Sol (medium)": "GPT-5.6 Sol medium",
  "GPT-5.6 Sol (max)": "GPT-5.6 Sol max",
  "GPT-5.6 Sol (high)": "GPT-5.6 Sol high",
  "GPT-5.6 Sol (xhigh)": "GPT-5.6 Sol xhigh",
  "Llama 3.2 Instruct 90B (Vision)": "Llama 3.2 Instruct 90B Vision",
  "Llama 3.2 Instruct 11B (Vision)": "Llama 3.2 Instruct 11B Vision",
  "Llama 2 Chat 70B": "Llama 2 70B Chat",
  "Llama 2 Chat 13B": "Llama 2 13B Chat",
  "Llama 2 Chat 7B": "Llama 2 7B Chat",
  "Qwen3 VL 235B A22B (Reasoning)": "Qwen3-VL 235B-A22B Thinking",
  "Qwen3 VL 8B (Reasoning)": "Qwen3-VL 8B Thinking",
  "Qwen3 VL 4B (Reasoning)": "Qwen3-VL 4B Thinking",
  "Qwen3 Next 80B A3B (Reasoning)": "Qwen3-Next Thinking",
  "Qwen3 14B (Reasoning)": "Qwen3-14B Thinking",
  "Qwen3.5 27B (Reasoning)": "Qwen3.5-27B",
  "Qwen3.5 35B A3B (Reasoning)": "Qwen3.5-35B-A3B",
  "Qwen3.5 397B A17B (Reasoning)": "Qwen3.5-397B-A17B Reasoning",
  "Qwen3.6 27B (Reasoning)": "Qwen3.6-27B",
  "Qwen3.6 35B A3B (Reasoning)": "Qwen3.6-35B-A3B",
  "Qwen3.6 Plus": "Qwen3.6-Plus",
  "Qwen3.7 Plus": "Qwen3.7-Plus",
  "Kimi K2.5 (Reasoning)": "Kimi K2.5",
  "Kimi K2.6": "Kimi K2.6",
  "Kimi K2.7 Code": "Kimi K2.7 Code",  "MiniMax-M2.7": "MiniMax M2.7",
  "MiniMax-M3": "MiniMax M3",
  "Grok 4.1 Fast (Reasoning)": "Grok 4.1 Fast Reasoning",
  "Grok 4.1 Fast (Non-reasoning)": "Grok 4.1 Fast Non-Reasoning",  "Grok 4.3 (high)": "Grok 4.3",
  "Grok 4.3 (Non-reasoning)": "Grok 4.3 none",
  "Grok 4.5 (high)": "Grok 4.5",
  "Muse Spark": "Muse Spark",
  "Muse Spark 1.1 (xhigh)": "Muse Spark 1.1",
  "Muse Spark 1.2 (xhigh)": "Muse Spark 1.2",
  "ERNIE 5.0 Thinking Preview": "ERNIE 5",
};

// Store the missing benchmarks into outputFilePath as JSON.
// If the file exists, overwrite it.
// Accepts the new format: array of match objects
function storeMissingBenchmarks(missingBenchmarks, outputFilePath) {
  const outputPath = path.resolve(outputFilePath);

  // Convert match objects to storage format
  const modelsToStore = missingBenchmarks.map(match => {
    const modelBase = match.dataModel ? {
      // Use the dataModel directly when it exists
      ...match.dataModel
    } : {
      // Fallback for cases where no dataModel exists
      name: null,
      company: match.aaModel.creator?.name || '',
      url: '',
      release_date: match.aaModel.releaseDate || '',
      capabilities: { input: [], output: [] }
    };

    return {
      aa_name: match.aaModel.name,
      ...modelBase,
      benchmarks: match.benchmarks,
      aa_metadata: match.aaModel
    };
  }).sort((a, b) => a.aa_name.localeCompare(b.aa_name));

  const sortedBenchmarks = { models: modelsToStore };

  fs.writeFileSync(outputPath, JSON.stringify(sortedBenchmarks, null, 2), 'utf8');
  console.error(`Stored ${modelsToStore.length} models with ambiguous/unmatched benchmarks to ${outputPath}`);
}

// Load the AA benchmark data from a JSON file.
// Return the JSON data as a JS object.
// It has the form {
//  models: [{
//      id, name, slug, releaseDate,
//      creator: {id, name, slug},
//      intelligenceIndex, gpqa, hle, livecodebench, scicode, aime25,
//      ifbench, lcr, terminalbenchHard, terminalbenchV21, tau2, tauBanking,
//      price1mInputTokens, price1mOutputTokens,
//      timescaleData: {medianOutputSpeed, ...}}]
// }
function loadAABenchData(pathToJSONFile) {
  const filePath = path.resolve(pathToJSONFile);

  if (!fs.existsSync(filePath)) {
    console.error(`AA benchmark data not found at ${filePath}, downloading...`);
    downloadAABenchData(filePath);
  }

  console.error(`Loading AA benchmark data from ${filePath}`);
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Download the AA benchmark data from https://artificialanalysis.ai.
// It stores the decoded JSON into pathToStoreJSONFile and returns it as a JS object.
function downloadAABenchData(pathToStoreJSONFile) {
  // The homepage HTML contains manifests with the path of an encrypted
  // payload and its key, e.g.
  //   "manifest":{"path":"/data/2a1c2b25faec2404.txt","key":"c320ae7d..."}
  // Each payload is AES-GCM encrypted and gzip compressed.
  // Fetch each manifest payload in turn and use the first one that
  // decrypts to the model list. Store the decoded JSON into
  // pathToStoreJSONFile and return it as a JS object.

  // Fetch the homepage and extract all manifest paths and keys.
  const homepage = execSync('curl -sSL https://artificialanalysis.ai', { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
  const manifests = findAAManifests(homepage);
  if (manifests.length === 0) {
    throw new Error('Could not find the data manifest in the Artificial Analysis homepage');
  }

  // Fetch each encrypted payload as raw bytes and use the first one
  // that decrypts to the model list.
  for (const { path: manifestPath, key: manifestKey } of manifests) {
    const encrypted = execSync(`curl -sSL https://artificialanalysis.ai${manifestPath}`, { encoding: 'buffer', maxBuffer: 32 * 1024 * 1024 });
    let data;
    try {
      data = decryptAAData(encrypted, manifestKey);
    } catch {
      continue;
    }
    if (data && Array.isArray(data.models)) {
      // Store the decoded JSON
      fs.writeFileSync(pathToStoreJSONFile, JSON.stringify({ models: data.models }, null, 2), 'utf8');
      console.error(`Downloaded and stored AA benchmark data to ${pathToStoreJSONFile}`);
      return { models: data.models };
    }
  }

  throw new Error('Could not download the AA model list from the Artificial Analysis homepage');
}

// Extract the {path, key} manifests from the Artificial Analysis homepage HTML.
// The manifests are embedded with escaped quotes, e.g.
//   \"manifest\":{\"path\":\"/data/2a1c2b25faec2404.txt\",\"key\":\"c320ae7d...\"}
// Return an array of {path, key} objects.
function findAAManifests(homepage) {
  const manifests = [];
  const manifestPattern = /\\"manifest\\":\{[^}]*\}/g;
  for (const block of homepage.match(manifestPattern) || []) {
    const manifestPath = block.match(/\\"path\\":\\"([^"\\]+)\\"/);
    const manifestKey = block.match(/\\"key\\":\\"([0-9a-f]+)\\"/);
    if (manifestPath && manifestKey) {
      manifests.push({ path: manifestPath[1], key: manifestKey[1] });
    }
  }
  return manifests;
}

// Decrypt an encrypted AA payload with its manifest key.
// The key is a hex string; the IV is the first 12 bytes of its SHA-256 hash,
// and the last 16 bytes of the payload are the GCM authentication tag.
// The decrypted bytes are gzip compressed JSON.
// Return the decoded JSON data as a JS object.
function decryptAAData(encryptedBytes, keyHex) {
  const key = Buffer.from(keyHex, 'hex');
  const iv = crypto.createHash('sha256').update(key).digest().slice(0, 12);
  const ciphertext = encryptedBytes.slice(0, -16);
  const authTag = encryptedBytes.slice(-16);

  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(authTag);
  const compressed = Buffer.concat([decipher.update(ciphertext), decipher.final()]);

  return JSON.parse(zlib.gunzipSync(compressed).toString('utf8'));
}


function scoreFromAAScore(aaScore, aaBenchName) {
  // Determine if we should scale the score
  if (shouldScaleBenchmark(aaBenchName)) {
    return aaScore * 100;
  } else {
    return aaScore;
  }
}

// Determine if a benchmark should be scaled from 0-1 to 0-100
function shouldScaleBenchmark(aaBenchName) {
  // Benchmarks already on their final scale (indices, costs, speeds, token counts) are not scaled
  const indexBenchmarks = [
    "intelligenceIndex",
    "omniscience",
    "gdpval",
    "agenticIndex",
    "price1mInputTokens",
    "price1mOutputTokens",
    "medianOutputSpeed",
    "intelligenceIndexOutputTokensPerTask",
  ];

  return !indexBenchmarks.includes(aaBenchName);
}

const benchNameFromAA = {
  "intelligenceIndex": "ArtificialAnalysis Intelligence Index",
  "gpqa": "GPQA Diamond",
  "hle": "Humanity's Last Exam",
  "livecodebench": "LiveCodeBench",
  "scicode": "SciCode",
  "aime25": "AIME 2025",
  "ifbench": "IFBench",
  "lcr": "LCR",
  "terminalbenchHard": "Terminal-Bench-Hard",
  "terminalbenchV21": "Terminal-Bench 2.1",
  "tau2": "τ²-Bench",
  "tauBanking": "τ³ Banking",
  "mmmuPro": "MMMU-Pro",
  "critpt": "CritPt",
  "apexAgents": "APEX-Agents",
  "omniscience": "AA-Omniscience Index",
  "gdpval": "GDPval-AA",
  "automationBenchPartialScore": "AA-AutomationBench Partial Score",
  "analystAgent": "AA-AnalystAgent",
  "enterpriseOpsGym": "EnterpriseOps-Gym",
  "itBenchSre": "ITBench SRE",
  "agenticIndex": "AA-Agentic Index",
  "intelligenceIndexOutputTokensPerTask": "AA Output Tokens per Task",
};

const benchNameFromAAPricing = {
  "price1mInputTokens": "Input cost",
  "price1mOutputTokens": "Output cost",
};

const benchNameFromAATopLevel = {
  "medianOutputSpeed": "Output speed",
};

// Benchmarks to exclude from automatic processing
const excludedBenchmarks = ["Input cost", "Output cost"];

if (require.main === module) {
  main();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    modelNameFromAA,
    aaModelsToIgnore,
    benchNameFromAA,
    benchNameFromAAPricing,
    benchNameFromAATopLevel,
    isUnambiguousMatch,
    matchAABenchmarks,
    mapModels,
    loadAABenchData,
    downloadAABenchData,
    findAAManifests,
    decryptAAData,
  };
}
