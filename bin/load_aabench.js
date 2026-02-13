// Download the benchmark data from Artificial Analysis.
// They have an API that can be used in that fashion:
// curl -X GET https://artificialanalysis.ai/api/v2/data/llms/models -H "x-api-key: $ARTIFICIAL_ANALYSIS_API_KEY"
// (ARTIFICIAL_ANALYSIS_API_KEY is set in .env)

const fs = require('fs');
const path = require('path');
const { normalizeModelName, isUnambiguousModelMatch, levenshteinDistance } = require('../lib/load-bench');

function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const verbose = args.includes('--verbose') || args.includes('-v');

  const aaBenchData = loadAABenchData("./data/aabench.json");
  const models = loadModelData();

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
//   aaModel: {name, evaluations, pricing},  // Original AA model data
//   dataModel: model object or null,         // Matched model from our data, or null if no match
//   benchmarks: [{name, score, source}]     // The AA benchmarks to add/update
// }
function matchAABenchmarks(aaBenchData, models) {
  // First, create a map of AA models to our models using the sophisticated mapping algorithm
  const modelMap = mapModels(aaBenchData, models);

  const matches = [];

  // Look through each model in `aaBenchData`.
  for (const aaModel of aaBenchData.data) {
    if (aaModelsToIgnore.includes(aaModel.name)) {
      continue;
    }

    // Get the mapped model from our model map
    const model = modelMap[aaModel.name];

    // Merge evaluations and pricing data into a single object for processing
    const allBenchmarks = {
      ...(aaModel.evaluations || {}),
      ...(aaModel.pricing || {})
    };

    // Create match object
    const match = {
      aaModel: aaModel,
      dataModel: model,
      benchmarks: []
    };

    // For each AA benchmark for that model (both evaluations and pricing),
    // check if it is useful and should be processed
    for (const [aaBenchName, score] of Object.entries(allBenchmarks)) {
      const mappedBenchName = benchNameFromAA[aaBenchName] || benchNameFromAAPricing[aaBenchName];
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
  for (const aaModel of aaBenchData.data) {
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
  for (const aaModel of aaBenchData.data) {
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
  for (const aaModel of aaBenchData.data) {
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
  "ERNIE 5.0 Thinking Preview",
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
  "o3": "o3 (high)",
  "gpt-oss-120B (high)": "gpt-oss-120b High",
  "gpt-oss-120B (low)": "gpt-oss-120b Low",
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
  "Claude Opus 4.5 (Non-reasoning)": "Claude Opus 4.5",
  "Claude Opus 4.5 (Reasoning)": "Claude Opus 4.5 Thinking",
  "Claude Opus 4.6 (Adaptive Reasoning)": "Claude Opus 4.6 Thinking",
  "Claude Opus 4.6 (Non-reasoning)": "Claude Opus 4.6",
  "DeepSeek R1 (Jan '25)": "DeepSeek R1",
  "DeepSeek R1 0528 (May '25)": "DeepSeek R1 0528",
  "DeepSeek V3 (Dec '24)": "DeepSeek V3",
  "DeepSeek V3.1 (Reasoning)": "DeepSeek V3.1",
  "DeepSeek V3.2 (Reasoning)": "DeepSeek V3.2",
  "DeepSeek V3.2 Exp (Reasoning)": "DeepSeek V3.2 Exp",
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
  "GLM-4.5 (Reasoning)": "GLM-4.5",
  "GLM-4.6 (Reasoning)": "GLM-4.6",
  "GLM-4.7 (Reasoning)": "GLM-4.7",
  "GLM-4.7-Flash (Reasoning)": "GLM-4.7 Flash",
  "GLM-5 (Reasoning)": "GLM-5",
  "GPT-4o (May '24)": "GPT-4o",
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
  "gpt-oss-20B (high)": "gpt-oss-20b High",
  "gpt-oss-20B (low)": "gpt-oss-20b Low",
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
  "Kimi K2.5 (Reasoning)": "Kimi K2.5",
  "Grok 4.1 Fast (Reasoning)": "Grok 4.1 Fast Reasoning",
  "Grok 4.1 Fast (Non-reasoning)": "Grok 4.1 Fast Non-Reasoning",
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
      company: match.aaModel.model_creator?.name || '',
      url: '',
      release_date: match.aaModel.release_date || '',
      capabilities: { input: [], output: [] }
    };

    return {
      aa_name: match.aaModel.name,
      ...modelBase,
      benchmarks: match.benchmarks,
      aa_metadata: {
        evaluations: match.aaModel.evaluations,
        pricing: match.aaModel.pricing
      }
    };
  }).sort((a, b) => a.aa_name.localeCompare(b.aa_name));

  const sortedBenchmarks = { models: modelsToStore };

  fs.writeFileSync(outputPath, JSON.stringify(sortedBenchmarks, null, 2), 'utf8');
  console.error(`Stored ${modelsToStore.length} models with ambiguous/unmatched benchmarks to ${outputPath}`);
}

// If ./aabench.json is present, load it from there.
// If not, download it.
// Return the JSON data as a JS object.
// It has the form {
//  status,
//  prompt_options: {parallel_queries, prompt_length},
//  data: [{
//      id, name, slug, release_date,
//      model_creator: {id, name, slug},
//      evaluations: {<name>: score},
//      pricing: {price_1m_blended_3_to_1, price_1m_input_tokens, price_1m_output_tokens},
//      median_output_tokens_per_second,
//      median_time_to_first_token_seconds,
//      median_time_to_first_answer_token}]
// }
function loadAABenchData(pathToStoreJSONFile) {
  const filePath = path.resolve(pathToStoreJSONFile);

  if (fs.existsSync(filePath)) {
    console.error(`Loading AA benchmark data from ${filePath}`);
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content);
  } else {
    console.error(`Downloading AA benchmark data...`);
    return downloadAABenchData(filePath);
  }
}

// Fetch the data through the ArtificialAnalysis API.
// Clean up the JSON, then store it into ./aabench.json
// Return the JSON data as a JS object.
function downloadAABenchData(pathToStoreJSONFile) {
  // Load API key from .env
  const envPath = path.resolve('.env');
  let apiKey = '';

  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf8');
    const match = envContent.match(/ARTIFICIAL_ANALYSIS_API_KEY=(.+)/);
    if (match) {
      apiKey = match[1].trim();
    }
  }

  if (!apiKey) {
    throw new Error('ARTIFICIAL_ANALYSIS_API_KEY not found in .env file');
  }

  // Use curl command to fetch data
  const { execSync } = require('child_process');
  const curlCommand = `curl -X GET https://artificialanalysis.ai/api/v2/data/llms/models -H "x-api-key: ${apiKey}"`;

  try {
    const result = execSync(curlCommand, { encoding: 'utf8' });
    const data = JSON.parse(result);

    // Store the cleaned-up JSON
    fs.writeFileSync(pathToStoreJSONFile, JSON.stringify(data, null, 2), 'utf8');
    console.error(`Downloaded and stored AA benchmark data to ${pathToStoreJSONFile}`);

    return data;
  } catch (error) {
    throw new Error(`Failed to download AA benchmark data: ${error.message}`);
  }
}

// Load the data from data/models/ company model files
function loadModelData() {
  const { loadModels } = require('../lib/load-models');
  return loadModels();
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
  // Index benchmarks are already on 0-100 scale, don't scale them
  const indexBenchmarks = [
    "artificial_analysis_intelligence_index",
    "artificial_analysis_coding_index",
    "artificial_analysis_math_index",
    "price_1m_input_tokens",
    "price_1m_output_tokens",
  ];

  return !indexBenchmarks.includes(aaBenchName);
}

const benchNameFromAA = {
  "artificial_analysis_intelligence_index": "ArtificialAnalysis Intelligence Index",
  "artificial_analysis_coding_index": "ArtificialAnalysis Coding Index",
  "artificial_analysis_math_index": "ArtificialAnalysis Math Index",
  "mmlu_pro": "MMLU-Pro",
  "gpqa": "GPQA Diamond",
  "hle": "Humanity's Last Exam",
  "livecodebench": "LiveCodeBench",
  "scicode": "SciCode",
  "math_500": "MATH",
  "aime": "AIME 2024",
  "aime_25": "AIME 2025",
  "ifbench": "IFBench",
  "lcr": "LCR",
  "terminalbench_hard": "Terminal-Bench-Hard",
  "tau2": "τ²-Bench",
};

const benchNameFromAAPricing = {
  "price_1m_input_tokens": "Input cost",
  "price_1m_output_tokens": "Output cost",
};

// Benchmarks to exclude from automatic processing
const excludedBenchmarks = ["Input cost", "Output cost"];

main();
