// Process the LMArena benchmark data downloaded directly from the API.
// The data is fetched from https://lmarena.ai/leaderboard/text with RSC header.

const fs = require('fs');
const path = require('path');

function main() {
  const lmarenaData = loadLMArenaData("./data/lmarena.json");
  const models = loadModelData();

  // Match LMArena models with our data models
  const modelMatches = matchBenchmarks(lmarenaData, models);

  // Filter matches using simple predicate functions
  const unambiguousModels = modelMatches.filter(isUnambiguousMatch);
  const ambiguousModels = modelMatches.filter(m => !isUnambiguousMatch(m));

  // Log summary
  logMatchSummary(modelMatches, unambiguousModels, ambiguousModels);

  // Update unambiguous matches
  updateUnambiguousModels(unambiguousModels, models);

  // Store ambiguous/unmatched models
  storeMissingBenchmarks(ambiguousModels, "./data/missing_lmarena_benchmarks.json");
}

// Match LMArena models with our data models and return match information
// Returns an array of match objects, each containing:
// {
//   lmarenaModel: {name, rating, metadata},  // Original LMArena model data
//   dataModel: model object or null,         // Matched model from our data, or null if no match
//   benchmark: {name, score, source},        // The LMArena benchmark to add
// }
function matchBenchmarks(lmarenaData, models) {
  // Get the benchmark name from the data (should be "text" for text arena)
  const benchmarkName = Object.keys(lmarenaData)[0];

  // First, filter out the models we want to ignore.
  const lmArenaModelNames = Object.keys(lmarenaData[benchmarkName]).filter(name => !ignoreModel(name));
  const textLMArenaData = lmArenaModelNames.map(name => ({
    name,
    rating: lmarenaData[benchmarkName][name].rating,
    metadata: lmarenaData[benchmarkName][name],
  }));

  // We now want the matching score of all the LMArena models against our model data.
  // It is a map from LMArena model name to our model.
  const modelMap = mapModels(textLMArenaData, models);

  const matches = [];

  console.warn(`Processing ${lmArenaModelNames.length} models from LMArena '${benchmarkName}' benchmark`);

  // Process each model in the benchmark
  for (const arenaModel of textLMArenaData) {
    // Check if this model exists in our current data
    const model = modelMap[arenaModel.name];

    const benchmarkFullName = `LMArena ${benchmarkName.charAt(0).toUpperCase() + benchmarkName.slice(1)}`;
    const matchingBenchmark = findBenchmark(benchmarkFullName, model);

    matches.push({
      lmarenaModel: arenaModel,
      dataModel: model,
      benchmark: {
        name: benchmarkFullName,
        score: Math.round(arenaModel.rating),
        source: "https://lmarena.ai/"
      },
    });
  }

  return matches;
}

// Map LMArena models to our data models.
// Return a map from LMArena model name to our model.
function mapModels(lmarenaData, models) {
  // 1. Compute the levenshtein distance for each possible mapping.
  // We create a list of {arenaModelName, modelName, distance}.
  const modelMappings = [];
  for (const arenaModel of lmarenaData) {
    const arenaNameNormalized = normalizeModelName(arenaModel.name);
    for (const model of models.models) {
      const modelNameNormalized = normalizeModelName(model.name);
      const distance = levenshteinDistance(arenaNameNormalized, modelNameNormalized);
      modelMappings.push({arenaModelName: arenaModel.name, modelName: model.name, distance});
    }
  }

  // 2. Assign unambiguous mappings.
  const modelMap = {};
  for (const arenaModel of lmarenaData) {
    for (const model of models.models) {
      if (isUnambiguousModelMatch(arenaModel, model)) {
        modelMap[arenaModel.name] = model;
      }
    }
  }

  // 3. Assign the mapping with the best levenshtein match, then iterate mappings.
  const sortedModelMappings = modelMappings.sort((a, b) => a.distance - b.distance);
  const assignedModels = new Set();
  for (const mapping of sortedModelMappings) {
    const arenaModelName = mapping.arenaModelName;
    const modelName = mapping.modelName;

    // If these models are already mapped, skip.
    if (modelMap[arenaModelName] || assignedModels.has(modelName)) {
      continue;
    }

    // Assign the mapping
    const model = models.models.find(m => m.name === modelName);
    modelMap[arenaModelName] = model;
  }

  return modelMap;
}

// Known model name mappings for cases where LMArena and our data use different naming conventions
const KNOWN_MODEL_MAPPINGS = {
  // LMArena name: Our data name
  "claude-opus-4-20250514-thinking-16k": "Claude Opus 4 Thinking",
  "claude-opus-4-5-20251101-thinking-32k": "Claude Opus 4.5 Thinking",
  "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (new)",
  "claude-haiku-4-5-20251001": "Claude Haiku 4.5 Thinking",
  "claude-3-7-sonnet-20250219-thinking-32k": "Claude Sonnet 3.7 Thinking",
  "claude-3-7-sonnet-20250219": "Claude Sonnet 3.7",
  "claude-opus-4-1-20250805-thinking-16k": "Claude Opus 4.1 Thinking",
  "claude-sonnet-4-20250514-thinking-32k": "Claude Sonnet 4 Thinking",
  "gpt-4o-2024-05-13": "GPT-4o",
  "command-r": "Command-R",
  "command-r-plus": "Command-R+",
};

const MODEL_PREFIXES_TO_IGNORE = [
  // Ignore for now, reconsider later as we add new companies etc.
  "amazon-",
  "c4ai-aya-expanse-",
  "chatglm",
  // Ignore forever.
  "gpt-4o-2024-08-06",
  "chatgpt-4o-latest-20250326",
  "athene-",
  "alpaca-13b",
];

function ignoreModel(modelName) {
  for (const prefix of MODEL_PREFIXES_TO_IGNORE) {
    if (modelName.startsWith(prefix)) {
      return true;
    }
  }
  return false;
}

// Normalize model name for comparison (lowercase, alphanumeric only)
function normalizeModelName(name) {
  return name.toLowerCase()
    .replace(/[0-9]{8}/, '')  // Remove date.
    .replace(/[\-\.]/g, ' ');
}

// Check if two model names represent an unambiguous match
// Returns true if the models match exactly (normalized) or via known mapping
function isUnambiguousModelMatch(lmarenaModel, ourModel) {
  if (!ourModel) {
    return false;
  }

  const lmarenaName = lmarenaModel.name;
  const ourModelName = ourModel.name;

  // 1. Check for known mappings
  if (KNOWN_MODEL_MAPPINGS[lmarenaName] === ourModelName) {
    return true;
  }

  // 2. Check for exact match (case-insensitive, normalized)
  if (normalizeModelName(lmarenaName) === normalizeModelName(ourModelName)) {
    return true;
  }

  // If none of the above criteria are met, consider it ambiguous
  return false;
}

// Check if a match is unambiguous (for auto-update)
// Uses isUnambiguousModelMatch to determine if the match is clear
function isUnambiguousMatch(match) {
  if (!match.dataModel) {
    return false;
  }

  return isUnambiguousModelMatch(match.lmarenaModel, match.dataModel);
}

// Convert match objects to the format expected for storage
function convertMatchToStorageFormat(match) {
  const modelBase = match.dataModel ? {
    // Use the dataModel directly when it exists
    ...match.dataModel
  } : {
    // Fallback for cases where no dataModel exists
    name: null,
    company: match.lmarenaModel.metadata.modelOrganization || '',
    url: match.lmarenaModel.metadata.modelUrl || '',
    release_date: '',
    capabilities: { input: [], output: [] }
  };

  return {
    lmarena_name: match.lmarenaModel.name,
    ...modelBase,
    benchmarks: [match.benchmark],
    lmarena_metadata: match.lmarenaModel.metadata
  };
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

// Find a benchmark named `benchmarkName` in the `model`.
// `matchingModel` is a model object from aggregated company model files
// Return the benchmark object if found, or null if not found.
function findBenchmark(benchmarkName, model) {
  if (!model || !model.benchmarks) {
    return null;
  }

  return model.benchmarks.find(benchmark =>
    benchmark.name === benchmarkName
  ) || null;
}

// Calculate Levenshtein distance between two strings
function levenshteinDistance(str1, str2) {
  const matrix = [];

  // Initialize matrix
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }

  // Fill matrix
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1,     // insertion
          matrix[i - 1][j] + 1      // deletion
        );
      }
    }
  }

  return matrix[str2.length][str1.length];
}

// Update model files with new LMArena benchmarks for unambiguous matches
function updateUnambiguousModels(unambiguousModels, models) {
  if (unambiguousModels.length === 0) {
    console.warn('No unambiguous model matches to update.');
    return;
  }

  let updatedCount = 0;

  for (const match of unambiguousModels) {
    const modelName = match.dataModel.name;
    const filePath = findModelFilePath(modelName, models);

    if (!filePath || !fs.existsSync(filePath)) {
      console.warn(`⚠️  Model file not found for ${modelName}: ${filePath}`);
      continue;
    }

    // Read the existing model file
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const modelData = JSON.parse(fileContent);

    // Find the specific model in the file
    const modelToUpdate = modelData.models.find(m => m.name === modelName);

    if (!modelToUpdate) {
      console.warn(`⚠️  Model ${modelName} not found in file ${filePath}`);
      continue;
    }

    // Check if benchmark already exists
    const existingBenchmarkIndex = modelToUpdate.benchmarks.findIndex(
      b => b.name === match.benchmark.name
    );

    if (existingBenchmarkIndex >= 0) {
      // Benchmark exists - update it if the rating is different
      const existingBenchmark = modelToUpdate.benchmarks[existingBenchmarkIndex];
      if (existingBenchmark.score !== match.benchmark.score) {
        modelToUpdate.benchmarks[existingBenchmarkIndex] = match.benchmark;
        console.warn(`🔄 Updated existing benchmark for ${modelName} (${match.lmarenaModel.name}): ${existingBenchmark.score} → ${match.benchmark.score}`);
      } else {
        console.warn(`ℹ️  Benchmark already exists for ${modelName} (${match.lmarenaModel.name}) with same rating (${match.benchmark.score}), no update needed`);
        continue;
      }
    } else {
      // Add the new benchmark
      modelToUpdate.benchmarks.push(match.benchmark);
      console.warn(`✅ Added new benchmark for ${modelName} (${match.lmarenaModel.name}): ${match.benchmark.score}`);
    }

    // Write the updated data back to the file
    fs.writeFileSync(filePath, JSON.stringify(modelData, null, 2), 'utf8');
    updatedCount++;

  }

  console.warn(`📊 Successfully updated ${updatedCount} model files with LMArena benchmarks`);
}

// Store the missing benchmarks into outputFilePath as JSON.
// If the file exists, overwrite it.
// Accepts the new format: array of match objects
function storeMissingBenchmarks(missingBenchmarks, outputFilePath) {
  const outputPath = path.resolve(outputFilePath);

  // Convert match objects to storage format
  const modelsToStore = missingBenchmarks
    .map(convertMatchToStorageFormat)
    .sort((a, b) => a.lmarena_name.localeCompare(b.lmarena_name));

  const sortedBenchmarks = { models: modelsToStore };

  fs.writeFileSync(outputPath, JSON.stringify(sortedBenchmarks, null, 2), 'utf8');
  console.warn(`Stored ${modelsToStore.length} models with ambiguous/unmatched benchmarks to ${outputPath}`);
}

// Load the LMArena data from the downloaded JSON file
// If the file exists, load it from there.
// If not, download it from the LMArena API.
function loadLMArenaData(pathToJSONFile) {
  const filePath = path.resolve(pathToJSONFile);

  // Download if file doesn't exist
  if (!fs.existsSync(filePath)) {
    console.warn(`Downloading LMArena data from API...`);
    downloadLMArenaData(filePath);
  }

  // Always load from file (either existing or newly downloaded)
  console.warn(`Loading LMArena data from ${filePath}`);
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Fetch the data directly from LMArena API
// The endpoint returns React Server Component data that contains JSON
function downloadLMArenaData(pathToStoreJSONFile) {
  const { execSync } = require('child_process');

  // Use curl to fetch the RSC data
  const curlCommand = `curl -s 'https://lmarena.ai/leaderboard/text' -H 'RSC: 1'`;

  try {
    const result = execSync(curlCommand, { encoding: 'utf8' });

    // Parse the RSC response to extract JSON data
    const parsedData = parseRSCResponse(result);

    if (!parsedData) {
      throw new Error('Could not extract JSON data from RSC response');
    }

    // Store the extracted JSON
    fs.writeFileSync(pathToStoreJSONFile, JSON.stringify(parsedData, null, 2), 'utf8');
    console.warn(`Downloaded and stored LMArena data to ${pathToStoreJSONFile}`);
  } catch (error) {
    throw new Error(`Failed to download LMArena data: ${error.message}`);
  }
}

// Parse React Server Component response to extract JSON data
// The RSC response contains JSON data embedded in the response
function parseRSCResponse(rscResponse) {
  // Extract arena slug to use as benchmark name
  const arenaPattern = /"arena"\s*:\s*\{[\s\S]*?"slug"\s*:\s*"([^"]+)"[\s\S]*?\}/;
  const arenaMatch = rscResponse.match(arenaPattern);
  const benchmarkName = arenaMatch ? arenaMatch[1] : "text"; // Default to "text" if not found

  // Extract the complete leaderboard data from the leaderboard entries array
  const leaderboardPattern = /"leaderboard"\s*:\s*\{[\s\S]*?"entries"\s*:\s*(\[[\s\S]*?\])[\s\S]*?\}/;
  const leaderboardMatch = rscResponse.match(leaderboardPattern);

  if (leaderboardMatch) {
    const entriesStr = leaderboardMatch[1];
    // Clean up the JSON string (remove trailing commas, etc.)
    const cleaned = entriesStr.replace(/,\s*\}/g, '}').replace(/,\s*\]/g, ']');
    const entries = JSON.parse(cleaned);

    if (Array.isArray(entries) && entries.length > 0) {
      console.warn(`Successfully extracted ${entries.length} models from leaderboard entries`);

      // Convert entries to the expected format with all available fields
      const fullData = {};
      for (const entry of entries) {
        if (entry.modelDisplayName && entry.rating !== undefined) {
          fullData[entry.modelDisplayName] = {
            rating: Math.round(entry.rating),
            modelOrganization: entry.modelOrganization || '',
            ratingUpper: entry.ratingUpper !== undefined ? Math.round(entry.ratingUpper) : null,
            ratingLower: entry.ratingLower !== undefined ? Math.round(entry.ratingLower) : null,
            license: entry.license || '',
            modelUrl: entry.modelUrl || '',
            // Include any other fields that might be useful
            ...(entry.modelName && { modelName: entry.modelName }),
            ...(entry.modelId && { modelId: entry.modelId }),
            ...(entry.arenaId && { arenaId: entry.arenaId }),
            ...(entry.leaderboardId && { leaderboardId: entry.leaderboardId })
          };
        }
      }

      return { [benchmarkName]: fullData };
    }
  }

  // If extraction fails, log the response for debugging
  console.warn('RSC Response (first 1000 chars):', rscResponse.substring(0, 1000));
  return null;
}

// Load the data from aggregated company model files
function loadModelData() {
  const { loadModels } = require('../lib/load-models');
  return loadModels();
}

// Log summary of the matching results
function logMatchSummary(modelMatches, unambiguousModels, ambiguousModels) {
  const existingBenchmarks = modelMatches.filter(m => m.dataModel && m.dataModel.benchmarks.some(b => b.name === m.benchmark.name)).length;
  console.warn(`📊 Summary:`);
  console.warn(`   Total LMArena models processed: ${modelMatches.length}`);
  console.warn(`   Unambiguous matches (will auto-update/add): ${unambiguousModels.length}`);
  console.warn(`   Ambiguous/unmatched (need manual review): ${ambiguousModels.length}`);
  console.warn(`   Existing benchmarks (will be updated if rating changed): ${existingBenchmarks}`);
}

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    isUnambiguousMatch,
    isUnambiguousModelMatch
  };
  // Only run main() when executed directly, not when required as a module
  if (require.main === module) {
    main();
  }
}
