'use strict';

// Compact encoding for models-prediction.
//
// Schema v1:
// {
//   version: 1,
//   benchmarks: string[],
//   sources: string[],
//   companies: string[],
//   urls: string[],
//   capabilities: string[], // distinct tokens e.g. ["audio","image","text","thinking","tool","video"]
//   models: CompactModel[]
// }
// CompactModel = [name, companyIdx, urlIdx, release_date, capBitmap, BenchTuple[]]
//   0: string model name
//   1: number index into companies
//   2: number index into urls
//   3: string release_date "YYYY-MM-DD"
//   4: number bitmap: bit i = input has capabilities[i], bit (i+|capabilities|) = output has capabilities[i]
//   5: BenchTuple[], one entry per benchmark
// BenchTuple = [score, stdDev, sourceIdx, isGuessed]  score/stdDev rounded to 3 decimals


// Round to 3 decimal places.
// Input:
// - x: floating-point number
// Output: floating-point number rounded to 3 decimal places.
function round3(x) {
  return Math.round(x * 1000) / 1000;
}

// ----- Encoder helpers -----

// Build sorted list of benchmark names.
// Input:
// - models: array of {benchmarks: [{name: string, …}], …}
// Output: sorted array of unique benchmark name strings.
function buildBenchmarks(models) {
  const set = new Set();
  for (const m of models) {
    for (const b of m.benchmarks) {
      set.add(b.name);
    }
  }
  const benchmarks = Array.from(set).sort();
  return benchmarks;
}

// Build sorted sources and index map.
// Input:
// - models: array of {benchmarks: [{source: string, …}], …}
// Output: object {sources: string[], index: Map<string, number>} sorted sources and source -> index.
function buildSources(models) {
  const set = new Set();
  for (const m of models) {
    for (const b of m.benchmarks) {
      set.add(b.source || '');
    }
  }
  const sources = Array.from(set).sort();
  const index = new Map(sources.map((s, i) => [s, i]));
  return { sources, index };
}

// Build sorted companies and index map.
// Input:
// - models: array of {company: string, …}
// Output: object {companies: string[], index: Map<string, number>} sorted companies and company -> index.
function buildCompanies(models) {
  const set = new Set(models.map(m => m.company));
  const companies = Array.from(set).sort();
  const index = new Map(companies.map((c, i) => [c, i]));
  return { companies, index };
}

// Build sorted urls and index map.
// Input:
// - models: array of {url: string, …}
// Output: object {urls: string[], index: Map<string, number>} sorted urls and url -> index.
function buildUrls(models) {
  const set = new Set(models.map(m => m.url));
  const urls = Array.from(set).sort();
  const index = new Map(urls.map((u, i) => [u, i]));
  return { urls, index };
}

// Build sorted capability tokens and position map.
// Input:
// - models: array of {capabilities: {input: string[], output: string[]}, …}
// Output: object {
//   capabilities: string[],   // Index of distinct capability types, sorted.
//   pos: Map<string, number>, // map from capability token to position in the index.
//   capCount: number                 // Number of capabilities tokens.
// } sorted tokens, token -> bit position, and count.
function buildCapabilities(models) {
  const set = new Set();
  for (const m of models) {
    const caps = m.capabilities || { input: [], output: [] };
    for (const t of caps.input || []) {  set.add(t); }
    for (const t of caps.output || []) { set.add(t); }
  }
  const capabilities = Array.from(set).sort();
  const pos = new Map(capabilities.map((t, i) => [t, i]));
  return { capabilities, pos, capCount: capabilities.length };
}

// Encode capabilities into bitmap.
// Input:
// - caps: object {input: string[], output: string[]}
// - capPos: Map<string, number> token -> bit position
// - capCount: number of capability options.
// Output: number bitmap where bit i = input has capabilities[i], bit i+capCount = output has capabilities[i].
function encodeCapabilityBitmap(caps, capPos, capCount) {
  let inMask = 0;
  let outMask = 0;
  for (const t of (caps && caps.input) || []) {
    const p = capPos.get(t);
    if (p !== undefined) { inMask |= 1 << p; }
  }
  for (const t of (caps && caps.output) || []) {
    const p = capPos.get(t);
    if (p !== undefined) { outMask |= 1 << p; }
  }
  return inMask | (outMask << capCount);
}

// Build bench tuples aligned to benchmarks.
// Input:
// - benchmarks: sorted string[] global order
// - model: object {benchmarks: [{name, score, stdDev, source, isGuessed}]}
// - sourceIndex: Map<string, number> source -> index
// Output: BenchTuple[] length B, each [score, stdDev, sourceIdx, isGuessed] rounded to 3 decimals, or null for sparse.
function buildBenchTuples(benchmarks, model, sourceIndex) {
  const map = new Map(model.benchmarks.map(b => [b.name, b]));
  const tuples = new Array(benchmarks.length);
  for (let i = 0; i < benchmarks.length; i++) {
    const entry = map.get(benchmarks[i]);
    if (!entry) {
      tuples[i] = null;
      continue;
    }
    const score = round3(typeof entry.score === 'number' ? entry.score : 0);
    const stdDev = round3(typeof entry.stdDev === 'number' ? entry.stdDev : 0);
    const sIdx = sourceIndex.get(entry.source || '');
    const g = entry.isGuessed ? 1 : 0;
    tuples[i] = [score, stdDev, sIdx, g];
  }
  return tuples;
}

// Encode single model to CompactModel list.
// Input:
// - model: object {name, company, url, release_date, capabilities, benchmarks}
// - ctx: object {benchmarks, sourceIndex, compIndex, urlIndex, capPos, capCount}
// Output: CompactModel list [name, companyIdx, urlIdx, release_date, bitmap, tuples].
function encodeModel(model, ctx) {
  const { benchmarks, sourceIndex, compIndex, urlIndex, capPos, capCount } = ctx;
  const tuples = buildBenchTuples(benchmarks, model, sourceIndex);
  const bitmap = encodeCapabilityBitmap(model.capabilities, capPos, capCount);
  return [model.name, compIndex.get(model.company), urlIndex.get(model.url), model.release_date, bitmap, tuples];
}

// ----- Decoder helpers -----

// Decode capability bitmap.
// Input:
// - bitmap: number encoded as above
// - capabilities: sorted string[] token list
// Output: object {input: string[], output: string[]} decoded capability lists.
function decodeCapabilityBitmap(bitmap, capabilities) {
  const capCount = capabilities.length;
  const inMask = bitmap & ((1 << capCount) - 1);
  const outMask = bitmap >> capCount;
  const input = [];
  const output = [];
  for (let i = 0; i < capCount; i++) {
    if (inMask & (1 << i)) {  input.push(capabilities[i]); }
    if (outMask & (1 << i)) { output.push(capabilities[i]); }
  }
  return { input, output };
}

// Decode bench tuples to benchmark objects.
// Input:
// - tuples: BenchTuple[] aligned to benchmarks
// - benchmarks: sorted string[] global order
// - sources: string[] indexed by sourceIdx
// Output: array of {name, score, stdDev, source, isGuessed} in benchmark order.
function decodeBenchTuples(tuples, benchmarks, sources) {
  const out = [];
  for (let i = 0; i < benchmarks.length; i++) {
    const t = tuples[i];
    if (t == null) continue;
    const [score, stdDev, sIdx, g] = t;
    out.push({ name: benchmarks[i], score, stdDev, source: sources[sIdx], isGuessed: !!g });
  }
  return out;
}

// Decode single CompactModel.
// Input:
// - cm: CompactModel list [name, companyIdx, urlIdx, release_date, bitmap, tuples]
// - dicts: object {benchmarks, sources, companies, urls, capabilities}
// Output: object {name, company, url, release_date, capabilities, benchmarks}.
function decodeModel(cm, dicts) {
  const { benchmarks, sources, companies, urls, capabilities } = dicts;
  const [name, cIdx, uIdx, release_date, bitmap, tuples] = cm;
  return {
    name,
    company: companies[cIdx],
    url: urls[uIdx],
    release_date,
    capabilities: decodeCapabilityBitmap(bitmap, capabilities),
    benchmarks: decodeBenchTuples(tuples, benchmarks, sources),
  };
}

// ----- Public API -----

// Encode prediction output to compact form.
// Input:
// - predictionOutput: object {models: [{name, company, url, release_date, capabilities, benchmarks}]}
// Output: compact object {version, benchmarks, sources, companies, urls, capabilities, models}.
function encodePredictions(predictionOutput) {
  const models = predictionOutput.models;
  const benchmarks = buildBenchmarks(models);
  const { sources, index: sourceIndex } = buildSources(models);
  const { companies, index: compIndex } = buildCompanies(models);
  const { urls, index: urlIndex } = buildUrls(models);
  const { capabilities, pos: capPos, capCount } = buildCapabilities(models);
  const ctx = { benchmarks, sourceIndex, compIndex, urlIndex, capPos, capCount };
  const compactModels = models.map(m => encodeModel(m, ctx));
  return { version: 1, benchmarks, sources, companies, urls, capabilities, models: compactModels };
}

// Decode compact form to prediction output.
// Input:
// - compact: object {version, benchmarks, sources, companies, urls, capabilities, models}
// Output: object {models: [{name, company, url, release_date, capabilities, benchmarks}]}.
function decodePredictions(compact) {
  const { benchmarks, sources, companies, urls, capabilities, models } = compact;
  const dicts = { benchmarks, sources, companies, urls, capabilities };
  return { models: models.map(cm => decodeModel(cm, dicts)) };
}

// Export for Node and browser (web/ symlink)
const api = { encodePredictions, decodePredictions, round3, buildBenchmarks, buildSources, buildCompanies, buildUrls, buildCapabilities };
if (typeof module !== 'undefined' && module.exports) { module.exports = api; }
if (typeof window !== 'undefined') { window.CompactPrediction = api; }
