// Download the benchmark data from Artificial Analysis.
// They have an API that can be used in that fashion:
// curl -X GET https://artificialanalysis.ai/api/v2/data/llms/models -H "x-api-key: $ARTIFICIAL_ANALYSIS_API_KEY"
// (ARTIFICIAL_ANALYSIS_API_KEY is set in .env)

const fs = require('fs');
const path = require('path');

function main() {
  const aaBenchData = loadAABenchData("./data/aabench.json");
  const models = loadModelData("./data/models.json");
  const missingBenchmarks = findMissingBenchmarks(aaBenchData, models);
  storeMissingBenchmarks(missingBenchmarks, "./data/missing_aabench_benchmarks.json");
}

// Return the list of benchmarks from `aaBenchData`
// which are not already present in `models`.
// The list is in the same format as ./data/models.json benchmarks:
// {models: [{name, company, url, release_date, capabilities, benchmarks: [{name, score, source}]}]}
function findMissingBenchmarks(aaBenchData, models) {
  const missingBenchmarks = { models: [] };

  // Look through each model in `aaBenchData`.
  for (const aaModel of aaBenchData.data) {
    if (aaModelsToIgnore.includes(aaModel.name)) {
      continue;
    }

    // Merge evaluations and pricing data into a single object for processing
    const allBenchmarks = {
      ...(aaModel.evaluations || {}),
      ...(aaModel.pricing || {})
    };

    // For each model, find the best match by name in `models`
    const model = findModel(aaModel.name, models);
    let newModel = model;

    if (!model) {
      // If there is no match, add all its benchmarks to the list of missing benchmarks
      newModel = {
        name: aaModel.name,
        company: aaModel.model_creator?.name || '',
        url: '',
        release_date: aaModel.release_date || '',
        capabilities: { input: [], output: [] },
        benchmarks: []
      };
    }

    newModel = {
      aa_name: aaModel.name,
      ...newModel,
      benchmarks: []
    };

    // For each AA benchmark for that model (both evaluations and pricing),
    // check if it is already present in `models`
    for (const [aaBenchName, score] of Object.entries(allBenchmarks)) {
      const mappedBenchName = benchNameFromAA[aaBenchName] || benchNameFromAAPricing[aaBenchName];
      if (mappedBenchName && typeof score === 'number') {
        // Check if benchmark is already present
        const existingBenchmark = model?.benchmarks?.find(b =>
          b.name === mappedBenchName &&
          equalEpsilon(b.score, scoreFromAAScore(score, aaBenchName))
        );
        // Check if benchmark is useful. When zero or null, it is not useful.
        const isUseful = score !== 0 && score != null;

        // If the benchmark is not present, add it to the list of missing benchmarks
        if (!existingBenchmark && isUseful) {
          // Only scale benchmarks that are on 0-1 scale (not index benchmarks)
          const finalScore = scoreFromAAScore(score, aaBenchName);
          newModel.benchmarks.push({
            name: mappedBenchName,
            score: Math.round(finalScore * 100) / 100, // Round to 2 decimal places
            source: "https://artificialanalysis.ai/api/v2/data/llms/models"
          });
        }
      }
    }

    if (newModel.benchmarks.length > 0) {
      missingBenchmarks.models.push(newModel);
    }
  }

  return missingBenchmarks;
}

const aaModelsToIgnore = [
  // Already imported, but AA price is different
  "Grok-1",
  "Llama 3.3 Instruct 70B",
  "Llama 3.1 Instruct 405B",
  "Llama 4 Scout",
  // Don't care, too edge-case, not relevant anymore:
  "Gemini 2.5 Flash Preview (Sep '25) (Non-reasoning)",
  "Gemini 2.5 Flash-Lite Preview (Sep '25) (Reasoning)",
  "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)",
  "Gemini 2.5 Flash-Lite Preview (Sep '25) (Non-reasoning)",
  "DeepSeek R1 Distill Llama 70B",
  "DeepSeek R1 Distill Llama 8B",
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
  // Other reasons
  "GPT-5 (ChatGPT)",  // Don't study chat constructs for now.
  "GPT-5.1 (Non-reasoning)",  // Already imported; name mismatch.
];

function equalEpsilon(a, b, epsilon = 0.0001) {
  return Math.abs(a - b) < epsilon;
}

// - The aaModelName is a string from AA data.
// - models is the raw data from ./data/models.json
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
};

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

// Load the data from ./data/models.json
function loadModelData(modelsFilePath) {
  const filePath = path.resolve(modelsFilePath);
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Levenshtein distance implementation
function levenshteinDistance(str1, str2) {
  const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));

  for (let i = 0; i <= str1.length; i++) {
    matrix[0][i] = i;
  }

  for (let j = 0; j <= str2.length; j++) {
    matrix[j][0] = j;
  }

  for (let j = 1; j <= str2.length; j++) {
    for (let i = 1; i <= str1.length; i++) {
      const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[j][i] = Math.min(
        matrix[j][i - 1] + 1, // deletion
        matrix[j - 1][i] + 1, // insertion
        matrix[j - 1][i - 1] + indicator // substitution
      );
    }
  }

  return matrix[str2.length][str1.length];
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
  "mmlu_pro": "MMLU Pro",
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

main();
