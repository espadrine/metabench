// Process the LMArena benchmark data downloaded via Docker.
// The data is stored in data/lmarena.json after running the Docker command.

const fs = require('fs');
const path = require('path');

function main() {
  const lmarenaData = loadLMArenaData("./data/lmarena.json");
  const models = loadModelData("./data/models.json");
  const missingBenchmarks = findMissingBenchmarks(lmarenaData, models);
  storeMissingBenchmarks(missingBenchmarks, "./data/missing_lmarena_benchmarks.json");
}

// Return the list of benchmarks from `lmarenaData`
// which are not already present in `models`.
// The list should be in the same format as ./data/models.json benchmarks:
// {models: [{name, company, url, release_date, capabilities, benchmarks: [{name, score, source}]}]}
function findMissingBenchmarks(lmarenaData, models) {
  const missingBenchmarks = { models: [] };

  // Only use the "full" benchmark from LMArena data
  // It is of the form {<model_name>: {rating: <number>, …}, …}
  const lmarenaBenchmarks = lmarenaData.full;

  console.error(`Processing ${Object.keys(lmarenaBenchmarks).length} models from LMArena 'full' benchmark`);

  // Process each model in the "full" benchmark
  for (const [arenaModelName, ratingData] of Object.entries(lmarenaBenchmarks)) {
    // Check if this model exists in our current data
    const model = findModel(arenaModelName, models);
    let newModel = model;

    if (!model) {
      // If there is no match, add a new model.
      newModel = {
        name: arenaModelName,
        company: '',
        url: '',
        release_date: '',
        capabilities: { input: [], output: [] },
        benchmarks: [],
      };
    }
    newModel = {
      lmarena_name: arenaModelName,
      ...newModel,
      benchmarks: [],
    };

    const matchingBenchmark = findBenchmark("LMArena Full", model);
    if (!matchingBenchmark) {
      newModel.benchmarks.push({
        name: "LMArena Full",
        score: Math.round(ratingData.rating),
        source: "https://lmarena.ai/"
      });
      missingBenchmarks.models.push(newModel);
    }
  }

  console.error(`Found ${missingBenchmarks.models.length} models with missing LMArena Full benchmark`);

  return missingBenchmarks;
}

// - The arenaModelName is a model name string from LMArena data.
// - models is the raw data from ./data/models.json
// Return the model from `models` that best matches `arenaModelName`,
// or null if no good match is found.
function findModel(arenaModelName, models) {
  // Lowercase the model names for comparison.
  const arenaNameLower = arenaModelName.toLowerCase();

  let bestMatch = null;
  let bestDistance = Infinity;

  // Use the levenshtein distance to find the best match.
  for (const model of models.models) {
    const modelNameLower = model.name.toLowerCase();
    const distance = levenshteinDistance(arenaNameLower, modelNameLower);

    // If the distance is too high (more than 30% of the length of the AA model name),
    // skip this match and loop.
    const maxAllowedDistance = Math.floor(arenaNameLower.length * 0.3);

    if (distance <= maxAllowedDistance && distance < bestDistance) {
      bestDistance = distance;
      bestMatch = model;
    }
  }

  return bestMatch;
}

// Find a benchmark named `benchmarkName` in the `model`.
// `matchingModel` is a model object from ./data/models.json
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

// Store the missing benchmarks into outputFilePath as JSON.
// If the file exists, overwrite it.
function storeMissingBenchmarks(missingBenchmarks, outputFilePath) {
  const outputPath = path.resolve(outputFilePath);

  // Sort models by name before storing
  const sortedBenchmarks = {
    models: missingBenchmarks.models.sort((a, b) => a.name.localeCompare(b.name))
  };

  fs.writeFileSync(outputPath, JSON.stringify(sortedBenchmarks, null, 2), 'utf8');
  console.error(`Stored ${missingBenchmarks.models.length} models with missing benchmarks to ${outputPath}`);
}

// Load the LMArena data from the downloaded JSON file
function loadLMArenaData(pathToJSONFile) {
  const filePath = path.resolve(pathToJSONFile);

  if (fs.existsSync(filePath)) {
    console.error(`Loading LMArena data from ${filePath}`);
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content);
  } else {
    console.error(`LMArena data file not found at ${filePath}`);
    console.error('Please run "make lmarena" first to download the data');
    process.exit(1);
  }
}

// Load the data from ./data/models.json
function loadModelData(modelsFilePath) {
  const filePath = path.resolve(modelsFilePath);
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

main();
