// Process the LMArena benchmark data downloaded directly from the API.
// The data is fetched from https://lmarena.ai/leaderboard/text with RSC header.

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

  // Get the benchmark name from the data (should be "text" for text arena)
  const benchmarkName = Object.keys(lmarenaData)[0];
  const lmarenaBenchmarks = lmarenaData[benchmarkName];

  console.error(`Processing ${Object.keys(lmarenaBenchmarks).length} models from LMArena '${benchmarkName}' benchmark`);

  // Process each model in the benchmark
  for (const [arenaModelName, ratingData] of Object.entries(lmarenaBenchmarks)) {
    // Check if this model exists in our current data
    const model = findModel(arenaModelName, models);
    let newModel = model;

    if (!model) {
      // If there is no match, add a new model with extracted metadata
      newModel = {
        name: arenaModelName,
        company: ratingData.modelOrganization || '',
        url: ratingData.modelUrl || '',
        release_date: '',
        capabilities: { input: [], output: [] },
        benchmarks: [],
        // Store additional LMArena metadata
        lmarena_metadata: {
          ratingUpper: ratingData.ratingUpper,
          ratingLower: ratingData.ratingLower,
          license: ratingData.license,
          modelOrganization: ratingData.modelOrganization,
          modelUrl: ratingData.modelUrl,
          ...(ratingData.modelName && { modelName: ratingData.modelName }),
          ...(ratingData.modelId && { modelId: ratingData.modelId }),
          ...(ratingData.arenaId && { arenaId: ratingData.arenaId }),
          ...(ratingData.leaderboardId && { leaderboardId: ratingData.leaderboardId })
        }
      };
    }
    newModel = {
      lmarena_name: arenaModelName,
      ...newModel,
      benchmarks: [],
    };

    const matchingBenchmark = findBenchmark(`LMArena ${benchmarkName.charAt(0).toUpperCase() + benchmarkName.slice(1)}`, model);
    if (!matchingBenchmark) {
      newModel.benchmarks.push({
        name: `LMArena ${benchmarkName.charAt(0).toUpperCase() + benchmarkName.slice(1)}`,
        score: Math.round(ratingData.rating),
        source: "https://lmarena.ai/"
      });
      missingBenchmarks.models.push(newModel);
    }
  }

  console.error(`Found ${missingBenchmarks.models.length} models with missing LMArena ${benchmarkName.charAt(0).toUpperCase() + benchmarkName.slice(1)} benchmark`);

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
// If the file exists, load it from there.
// If not, download it from the LMArena API.
function loadLMArenaData(pathToJSONFile) {
  const filePath = path.resolve(pathToJSONFile);

  // Download if file doesn't exist
  if (!fs.existsSync(filePath)) {
    console.error(`Downloading LMArena data from API...`);
    downloadLMArenaData(filePath);
  }

  // Always load from file (either existing or newly downloaded)
  console.error(`Loading LMArena data from ${filePath}`);
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
    console.error(`Downloaded and stored LMArena data to ${pathToStoreJSONFile}`);
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
    try {
      const entriesStr = leaderboardMatch[1];
      // Clean up the JSON string (remove trailing commas, etc.)
      const cleaned = entriesStr.replace(/,\s*\}/g, '}').replace(/,\s*\]/g, ']');
      const entries = JSON.parse(cleaned);

      if (Array.isArray(entries) && entries.length > 0) {
        console.error(`Successfully extracted ${entries.length} models from leaderboard entries`);

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
    } catch (e) {
      console.error('Failed to parse leaderboard entries:', e.message);
    }
  }

  // If extraction fails, log the response for debugging
  console.error('RSC Response (first 1000 chars):', rscResponse.substring(0, 1000));
  return null;
}

// Load the data from ./data/models.json
function loadModelData(modelsFilePath) {
  const filePath = path.resolve(modelsFilePath);
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

main();
