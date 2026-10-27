// Load benchmark data from Vals AI and match it against our models.

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { normalizeModelName, isUnambiguousModelMatch, levenshteinDistance } = require('../lib/load-bench');
const { loadModels } = require('../lib/load-models');

function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const verbose = args.includes('--verbose') || args.includes('-v');

  const valsData = loadValsData("./data/valsai.json");
  const models = loadModels();

  // Match Vals AI models with our data models
  const modelMatches = matchValsBenchmarks(valsData, models);

  // Filter matches using simple predicate functions
  const unambiguousModels = modelMatches.filter(isUnambiguousMatch);
  const ambiguousModels = modelMatches.filter(m => !isUnambiguousMatch(m));

  // Log summary
  logMatchSummary(modelMatches, unambiguousModels, ambiguousModels);

  // Update unambiguous matches
  updateUnambiguousModels(unambiguousModels, models, verbose);

  // Store ambiguous/unmatched models
  storeMissingBenchmarks(ambiguousModels, models, "./data/missing_valsai_benchmarks.csv");
}

// Match Vals AI benchmarks with our data models and return match information.
// Returns an array of match objects, each containing:
// {
//   valsModel: model object from the Vals AI data, // Original Vals AI model data
//   dataModel: model object or null,               // Matched model from our data, or null if no match
//   benchmarks: [{name, score, source}]             // The Vals AI benchmarks to add/update
// }
function matchValsBenchmarks(valsData, models) {
  // First, create a map of Vals AI models to our models using the same
  // mapping algorithm as Artificial Analysis.
  const modelMap = mapModels(valsData, models);

  // Collect the benchmark scores for each Vals AI model key.
  // Only the "overall" task of each benchmark is used.
  const benchmarksByModel = {};
  for (const [slug, benchmarkData] of Object.entries(valsData.benchmarks)) {
    const benchName = benchNameFromVals[slug];
    if (!benchName) {
      continue;
    }
    const overall = benchmarkData.tasks && benchmarkData.tasks.overall;
    if (!overall) {
      continue;
    }
    for (const [modelKey, entry] of Object.entries(overall)) {
      const score = entry && entry.accuracy;
      if (typeof score !== 'number' || score === 0) {
        continue;
      }
      benchmarksByModel[modelKey] = benchmarksByModel[modelKey] || [];
      benchmarksByModel[modelKey].push({
        name: benchName,
        score: Math.round(score * 100) / 100, // Round to 2 decimal places
        source: `https://www.vals.ai/models/${modelKey.replace('/', '_')}`
      });
    }
  }

  const matches = [];
  for (const [modelKey, benchmarks] of Object.entries(benchmarksByModel)) {
    const model = modelMap[modelKey];
    const match = {
      valsModel: {
        key: modelKey,
        name: matchNameFromVals(modelKey)
      },
      dataModel: model,
      benchmarks
    };
    matches.push(match);
  }

  return matches;
}

// Map Vals AI models to our data models using the same algorithm as Artificial Analysis.
// Return a map from Vals AI model key to our model.
function mapModels(valsData, models) {
  // 1. Compute the levenshtein distance for each possible mapping.
  // We create a list of {valsKey, valsName, modelName, distance}.
  const modelMappings = [];
  const valsNames = valsModelNames(valsData);
  for (const [valsKey, valsName] of valsNames) {
    const valsNameNormalized = normalizeModelName(valsName);
    for (const model of models.models) {
      const modelNameNormalized = normalizeModelName(model.name);
      const distance = levenshteinDistance(valsNameNormalized, modelNameNormalized);
      modelMappings.push({valsKey, modelName: model.name, distance});
    }
  }

  // 2. Assign known mappings.
  const modelMap = {};
  for (const [valsKey] of valsNames) {
    const knownMappingName = modelNameFromVals[valsKey];
    if (knownMappingName) {
      const model = models.models.find(m => m.name === knownMappingName);
      if (model) {
        modelMap[valsKey] = model;
      }
    }
  }

  // 3. Assign unambiguous mappings.
  for (const [valsKey] of valsNames) {
    for (const model of models.models) {
      const notAlreadyMapped = !modelMap[valsKey];
      if (isUnambiguousModelMatch({name: matchNameFromVals(valsKey)}, model) && notAlreadyMapped) {
        modelMap[valsKey] = model;
      }
    }
  }

  // 4. Assign the mapping with the best levenshtein match, then iterate mappings.
  const sortedModelMappings = modelMappings.sort((a, b) => a.distance - b.distance);
  const assignedModels = new Set();
  for (const mapping of sortedModelMappings) {
    const valsKey = mapping.valsKey;
    const modelName = mapping.modelName;

    // If these models are already mapped, skip.
    if (modelMap[valsKey] || assignedModels.has(modelName)) {
      continue;
    }

    // Assign the mapping
    const model = models.models.find(m => m.name === modelName);
    modelMap[valsKey] = model;
    assignedModels.add(modelName);
  }

  return modelMap;
}

// Collect the model keys present in the Vals AI data,
// with their derived match names, sorted by key.
// Returns an array of [key, matchName] pairs.
function valsModelNames(valsData) {
  const names = [];
  for (const benchmarkData of Object.values(valsData.benchmarks)) {
    const overall = benchmarkData.tasks && benchmarkData.tasks.overall;
    if (!overall) {
      continue;
    }
    for (const modelKey of Object.keys(overall)) {
      if (!names.some(([key]) => key === modelKey)) {
        names.push([modelKey, matchNameFromVals(modelKey)]);
      }
    }
  }
  return names;
}

// Derive the model name used for matching a Vals AI model key.
// The keys have the form "provider/model-name", sometimes with a trailing
// date, e.g. "openai/gpt-5.1-2025-11-13" or "anthropic/claude-3-7-sonnet-20250219".
// Known mappings in modelNameFromVals take precedence.
function matchNameFromVals(valsKey) {
  const knownMappingName = modelNameFromVals[valsKey];
  if (knownMappingName) {
    return knownMappingName;
  }
  let name = valsKey.includes('/') ? valsKey.split('/').slice(1).join('/') : valsKey;
  name = name.replace(/-(\d{4}-\d{2}-\d{2}|\d{8})$/, '');
  return name;
}

// Check if a match is unambiguous (for auto-update)
function isUnambiguousMatch(match) {
  if (!match.dataModel) {
    return false;
  }

  return isUnambiguousModelMatch(match.valsModel, match.dataModel);
}

// Update model files with new Vals AI benchmarks for unambiguous matches
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
      console.error(`WARNING: Model file not found for ${modelName}: ${filePath}`);
      continue;
    }

    // Read the existing model file
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const modelData = JSON.parse(fileContent);

    // Find the specific model in the file
    const modelToUpdate = modelData.models.find(m => m.name === modelName);

    if (!modelToUpdate) {
      console.error(`WARNING: Model ${modelName} not found in file ${filePath}`);
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
          console.error(`Updated existing benchmark for ${modelName} (${match.valsModel.key}): ${existingBenchmark.score} -> ${benchmark.score}`);
        } else {
          if (verbose) {
            console.error(`Benchmark already exists for ${modelName} (${match.valsModel.key}) with same score (${benchmark.score}), no update needed`);
          }
          continue;
        }
      } else {
        // Add the new benchmark
        modelToUpdate.benchmarks.push(benchmark);
        console.error(`Added new benchmark for ${modelName} (${match.valsModel.key}): ${benchmark.name} = ${benchmark.score}`);
      }
    }

    // Write the updated data back to the file
    fs.writeFileSync(filePath, JSON.stringify(modelData, null, 2), 'utf8');
    updatedCount++;
  }

  console.error(`Successfully updated ${updatedCount} model files with Vals AI benchmarks`);
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

  console.error(`Summary:`);
  console.error(`   Total Vals AI models processed: ${modelMatches.length}`);
  console.error(`   Unambiguous matches (will auto-update/add): ${unambiguousModels.length}`);
  console.error(`   Ambiguous/unmatched (need manual review): ${ambiguousModels.length}`);
  console.error(`   Existing benchmarks (will be updated if score changed): ${existingBenchmarks}`);
}

// Store the missing benchmarks into outputFilePath as CSV.
// If the file exists, overwrite it.
// One line per ambiguous/unmatched Vals AI model, tab-separated:
// <vals_name>\t<closest Levenshtein match among model names in data/models>\t<release_date>
// where <release_date> is that of the closest matching model.
function storeMissingBenchmarks(missingBenchmarks, models, outputFilePath) {
  const outputPath = path.resolve(outputFilePath);

  const lines = missingBenchmarks.map(match => {
    const closest = findClosestModel(match.valsModel.key, models);
    return [match.valsModel.key, closest.name, closest.release_date].join('\t');
  }).sort();

  fs.writeFileSync(outputPath, lines.join('\n') + '\n', 'utf8');
  console.error(`Stored ${lines.length} models with ambiguous/unmatched benchmarks to ${outputPath}`);
}

// Return our model whose name is closest (Levenshtein distance)
// to the given Vals AI model key.
function findClosestModel(valsKey, models) {
  const valsNameNormalized = normalizeModelName(matchNameFromVals(valsKey));
  let closestModel = null;
  let bestDistance = Infinity;
  for (const model of models.models) {
    const distance = levenshteinDistance(valsNameNormalized, normalizeModelName(model.name));
    if (distance < bestDistance) {
      bestDistance = distance;
      closestModel = model;
    }
  }
  return closestModel;
}

// Load the Vals AI benchmark data from a JSON file.
// Return the JSON data as a JS object.
// It has the form {
//  benchmarks: {
//    "<benchmark slug>": {
//      metadata: {benchmark, slug, description, benchmark_id, family, version, updated, ...},
//      tasks: {"overall": {"<provider/model>": {accuracy, latency, stderr, ...}, ...}, ...}}}
// }
function loadValsData(pathToJSONFile) {
  const filePath = path.resolve(pathToJSONFile);

  if (!fs.existsSync(filePath)) {
    console.error(`Vals AI benchmark data not found at ${filePath}, downloading...`);
    downloadValsData(filePath);
  }

  console.error(`Loading Vals AI benchmark data from ${filePath}`);
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

// Download the Vals AI benchmark data from https://www.vals.ai.
// Each benchmark page embeds a link to a static JSON asset with the
// full leaderboard data, e.g. /_astro/benchmark_view_vals_index.DXmu6JHS.json.
// Fetch the benchmark list from the /benchmarks page, then download each
// benchmark's JSON asset, and store them all into pathToStoreJSONFile.
function downloadValsData(pathToStoreJSONFile) {
  const benchmarksPage = execSync('curl -sSL https://www.vals.ai/benchmarks', { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
  const slugs = findValsBenchmarkSlugs(benchmarksPage);
  if (slugs.length === 0) {
    throw new Error('Could not find the benchmark list in the Vals AI /benchmarks page');
  }

  const benchmarks = {};
  for (const slug of slugs) {
    const benchmarkData = downloadValsBenchmark(slug);
    if (benchmarkData) {
      benchmarks[slug] = benchmarkData;
    }
  }

  if (Object.keys(benchmarks).length === 0) {
    throw new Error('Could not download any benchmark data from Vals AI');
  }

  fs.writeFileSync(pathToStoreJSONFile, JSON.stringify({ benchmarks }, null, 2), 'utf8');
  console.error(`Downloaded and stored Vals AI benchmark data to ${pathToStoreJSONFile}`);
}

// Extract the benchmark slugs from the Vals AI /benchmarks page HTML.
// Return an array of slug strings, e.g. ["vals_index", "gpqa", ...].
function findValsBenchmarkSlugs(benchmarksPage) {
  const slugs = [];
  for (const match of benchmarksPage.matchAll(/\/benchmarks\/([a-zA-Z0-9_-]+)/g)) {
    if (!slugs.includes(match[1])) {
      slugs.push(match[1]);
    }
  }
  return slugs;
}

// Download the leaderboard data of a single Vals AI benchmark.
// Fetch the benchmark page and extract the path of its JSON asset,
// then download and parse it. Return null when the page has no data.
function downloadValsBenchmark(slug) {
  const benchmarkPage = execSync(`curl -sSL https://www.vals.ai/benchmarks/${slug}`, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });

  // The JSON asset paths have the form /_astro/benchmark_view_<name>.<hash>.json
  const assetPaths = [...new Set(
    [...benchmarkPage.matchAll(/href="(\/_astro\/[a-zA-Z0-9_.-]+\.json)"/g)].map(m => m[1])
  )];

  for (const assetPath of assetPaths) {
    const asset = execSync(`curl -sSL https://www.vals.ai${assetPath}`, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
    let data;
    try {
      data = JSON.parse(asset);
    } catch {
      continue;
    }
    if (data && data.metadata && data.tasks && data.tasks.overall) {
      return data;
    }
  }

  console.error(`No leaderboard data found for Vals AI benchmark ${slug}`);
  return null;
}

// Vals AI benchmark slug: our benchmark name.
// Only benchmarks with an entry here are loaded.
// The "overall" task of each benchmark is used as the score.
// The Vals AI pages for TaxEval, CorpFin, CaseLaw and Finance Agent are the
// v2 versions: they are considered continuous with the original benchmarks.
// Excluded:
// - "poker_agent": the Agent Poker Bench score is not a percentage.
// - "rsi_index", "mental-health": their pages have no leaderboard data asset.
// - "public-benefits-bench-v1": superseded by "public-benefits-bench".
const benchNameFromVals = {
  "vals_index": "Vals Index",
  "vals_multimodal_index": "Vals Multimodal Index",
  "web_search": "Web Search Index",
  "legal_research": "Legal Research Bench",
  "hlab": "Harvey LAB (Vals)",
  "legal_bench": "LegalBench",
  "case_law_v2": "CaseLaw",
  "fabv2": "Finance Agent v2",
  "emb": "EMB",
  "tax_agent_bench": "Tax Agent Bench",
  "corp_fin_v2": "CorpFin",
  "mortgage_tax": "MortgageTax",
  "tax_eval_v2": "TaxEval",
  "medcode": "MedCode",
  "medscribe": "MedScribe",
  "medqa": "MedQA",
  "proof_bench": "ProofBench",
  "aime": "AIME 2024+2025",
  "math500": "MATH500",
  "mgsm": "MGSM",
  "terminal-bench-science": "Terminal-Bench-Science 0.1",
  "biomysterybench": "BioMystery Bench",
  "mysterymechanism": "MysteryMechanism",
  "gpqa": "GPQA Diamond",
  "mmlu_pro": "MMLU-Pro",
  "mmmu": "MMMU",
  "sage": "SAGE",
  "voice-code-bench": "VoiceCodeBench",
  "vibe-code": "Vibe Code Bench",
  "vcb-1-100": "Vibe Code Bench 1-100",
  "code-migration": "Code Migration",
  "programbench": "ProgramBench",
  "terminal-bench-4": "Terminal-Bench 4.0",
  "skillsbench": "SkillsBench",
  "ioi": "IOI",
  "swebench-multimodal": "SWE-bench Multimodal",
  "terminal-bench-2-1": "Terminal-Bench 2.1",
  "swebench": "SWE-bench Verified",
  "lcb": "LiveCodeBench",
  "terminal-bench-2": "Terminal-Bench 2.0",
  "cyber": "CyberBench",
  "srebench": "SRE-Bench",
  "cua_bench": "CUA-bench",
  "time_horizon_index": "Time Horizon Index: KSP",
  "public-benefits-bench": "Public Benefits Bench",
};

// Vals AI model key: our data name.
// Append the latest at the top.
// Keys have the form "provider/model-name", e.g. "openai/gpt-5.1-2025-11-13".
// When no entry exists, the provider prefix and a trailing date are stripped
// from the key, and the remainder is compared against our model names.
const modelNameFromVals = {
  "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash 1017",
  "meta/muse-spark-open": "Muse Spark Open",
  "openai/gpt-5.4-2026-03-05-high": "GPT-5.4 high",
  "openai/gpt-5.2-2025-12-11": "GPT-5.2 xhigh",
  "openai/gpt-5.1-2025-11-13": "GPT-5.1 High",
  "openai/gpt-5-codex": "GPT-5 Codex High",
  "openai/gpt-5-2025-08-07": "GPT-5 Medium",
  "openai/gpt-3.5-turbo": "GPT-3.5",
  "openai/o3-2025-04-16": "o3 (high)",
  "anthropic/claude-sonnet-5": "Claude Sonnet 5 Thinking",
  "anthropic/claude-fable-5": "Claude Fable 5 Thinking",
  "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
  "anthropic/claude-3-7-sonnet-20250219-thinking": "Claude Sonnet 3.7 Thinking",
  "anthropic/claude-3-7-sonnet-20250219": "Claude Sonnet 3.7",
  "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
  "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
  "google/gemini-3-pro-preview": "Gemini 3 Pro",
  "google/gemini-3-flash-preview": "Gemini 3 Flash",
  "google/gemini-2.5-pro-preview-03-25": "Gemini 2.5 Pro 0325",
  "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
  "google/gemini-1.5-pro-002": "Gemini 1.5 Pro",
  "google/gemini-1.5-flash-002": "Gemini 1.5 Flash",
  "google/gemini-1.0-pro-002": "Gemini 1.0 Pro",
  "google/gemma-4-31b-it": "Gemma 4 31B",
  "grok/grok-4.20-0309-reasoning": "Grok 4.2",
  "grok/grok-4-fast-reasoning": "Grok 4 Fast",
  "grok/grok-4-0709": "Grok 4",
  "grok/grok-3-mini-fast-high-reasoning": "Grok 3 Mini High",
  "zai/glm-5-thinking": "GLM-5",
  "kimi/kimi-k2.5-thinking": "Kimi K2.5",
  "meta/muse_spark_1_3_max": "Muse Spark 1.3",
  "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
  "fireworks/gpt-oss-120b": "gpt-oss-120b High",
  "fireworks/gpt-oss-20b": "gpt-oss-20b High",
  "fireworks/deepseek-v3p2": "DeepSeek V3.2",
  "fireworks/deepseek-v3p1": "DeepSeek V3.1",
  "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro Max 0813",
  "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash Max 0731",
  "mistralai/mistral-large-2512": "Mistral Large 4",
  "mistralai/mistral-large-2512": "Mistral Large 3",
  "mistralai/mistral-large-2411": "Pixtral Large",
  "mistralai/mistral-medium-2505": "Mistral Medium 3",
  "mistralai/mistral-small-2503": "Mistral Small 3.1",
  "mistralai/mistral-small-2402": "Mistral Small 1 2402",
  "mistralai/magistral-medium-2509": "Magistral Medium 1.2",
  "mistralai/magistral-small-2509": "Magistral Small 1.2",
  "mistralai/devstral-2512": "Devstral 2 123B",
  "mistralai/labs-devstral-small-2512": "Devstral Small 2 24B",
  "cohere/command-a-plus-05-2026": "Command A+",
  "cohere/command-a-03-2025": "Command-A",
  "cohere/command-r-plus": "Command-R+",
  "together/meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
  "together/meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 Instruct 70B",
  "together/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": "Llama 3.2 Instruct 90B Vision",
  "together/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": "Llama 3.2 Instruct 11B Vision",
  "together/meta-llama/Llama-3.1-405B-Instruct-Turbo": "Llama 3.1 Instruct 405B",
  "together/meta-llama/Llama-3.1-70B-Instruct-Turbo": "Llama 3.1 Instruct 70B",
  "together/meta-llama/Llama-3.1-8B-Instruct-Turbo": "Llama 3.1 Instruct 8B",
  "together/mistralai/Mixtral-8x22B-Instruct-v0.1": "Mixtral 8x22B",
  "together/mistralai/Mixtral-8x7B-v0.1": "Mixtral 8x7B",
  "together/mistralai/Mistral-7B-v0.1": "Mistral 7B",
  "together/google/gemma-2-27b-it": "Gemma 2 27B",
  "together/google/gemma-2-9b-it": "Gemma 2 9B",
  "together/moonshotai/Kimi-K2-Instruct": "Kimi K2",
};

if (require.main === module) {
  main();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    modelNameFromVals,
    benchNameFromVals,
    matchValsBenchmarks,
    mapModels,
    matchNameFromVals,
    loadValsData,
    downloadValsData,
    findValsBenchmarkSlugs,
  };
}
