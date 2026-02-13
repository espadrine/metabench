#!/usr/bin/env node
// Script to extract and list all unique benchmark names from data/models/* files

const fs = require('fs');
const path = require('path');

function main() {
  // Check for help flag
  const args = process.argv.slice(2);
  if (args.includes('--help') || args.includes('-h')) {
    showHelp();
    process.exit(0);
  }

  try {
    const benchmarks = extractBenchmarkNamesFromModelFiles();

    console.log("Available Benchmark Names:");
    console.log("========================");

    // Sort and display benchmarks
    const sortedBenchmarks = Object.keys(benchmarks).sort();
    for (const benchmarkName of sortedBenchmarks) {
      const count = benchmarks[benchmarkName];
      console.log(`${benchmarkName} (found in ${count} models)`);
    }

    console.log(`\nTotal unique benchmarks: ${sortedBenchmarks.length}`);

  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

// Extract all unique benchmark names from all model files in data/models/
function extractBenchmarkNamesFromModelFiles() {
  const benchmarkCounts = {};
  const modelsDir = './data/models/';

  // Read all files in the models directory
  const files = fs.readdirSync(modelsDir);

  for (const file of files) {
    if (file.endsWith('.json')) {
      const filePath = path.join(modelsDir, file);
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        const modelsData = JSON.parse(content);

        // Process each model in this file
        if (modelsData.models && Array.isArray(modelsData.models)) {
          for (const model of modelsData.models) {
            if (model.benchmarks && Array.isArray(model.benchmarks)) {
              for (const benchmark of model.benchmarks) {
                if (benchmark.name) {
                  benchmarkCounts[benchmark.name] = (benchmarkCounts[benchmark.name] || 0) + 1;
                }
              }
            }
          }
        }
      } catch (error) {
        console.error(`⚠️  Warning: Could not process file ${file}: ${error.message}`);
        // Continue with other files even if one fails
      }
    }
  }

  return benchmarkCounts;
}

function showHelp() {
  console.log(`
Usage: node bin/list-benchmarks.js [options]

Options:
  --help, -h    Show this help message

Description:
  Extracts and lists all unique benchmark names from data/models/* files.
  Shows each benchmark name along with the number of models it appears in.

  The script reads from all JSON files in ./data/models/ and displays:
  - All unique benchmark names from all models across all companies
  - Count of models that have each benchmark
  - Total number of unique benchmarks

Examples:
  node bin/list-benchmarks.js
  node bin/list-benchmarks.js --help
`);
}

main();
