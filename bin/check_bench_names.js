#!/usr/bin/env node

const fs = require('fs');
const { levenshteinDistance } = require('../lib/load-bench');

function main() {
  const args = parseArguments();
  const existingBenchmarkNames = loadExistingBenchmarkNames(args.existingBenchmarksPath);
  const newBenchmarkNames = loadNewBenchmarkNames(args.newBenchmarksPath);
  const missingBenchmarks = findMissingBenchmarks(newBenchmarkNames, existingBenchmarkNames);
  const nearestNames = findNearestBenchmarkNames(missingBenchmarks, existingBenchmarkNames);
  printNearestNames(nearestNames);
}

// Parses command-line arguments and validates them.
// Exits the process if help is requested or if arguments are invalid.
// Returns an object with:
// - newBenchmarksPath: string
// - existingBenchmarksPath: string
function parseArguments() {
  const args = process.argv.slice(2);

  // Check for help flag
  if (args.includes('--help') || args.includes('-h')) {
    showHelp();
    process.exit(0);
  }

  if (args.length !== 2) {
    showHelp();
    process.exit(1);
  }

  return {
    newBenchmarksPath: args[0],
    existingBenchmarksPath: args[1]
  };
}

// Loads benchmark names from a JSON file containing benchmarks data.
// The file format is: {"benchmarks": [{name: string}, ...]}
// - filePath: string, Path to the benchmarks JSON file.
// Returns: Array of benchmark names.
function loadNewBenchmarkNames(filePath) {
  const benchmarksData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  return benchmarksData.benchmarks.map(b => b.name);
}

// Loads existing benchmark names from the bm file.
// The benchmark file is a plain text file with one benchmark name per line.
// - path: string, Path to the benchark file.
// Returns: Array of benchmark names from the file.
function loadExistingBenchmarkNames(path) {
  const content = fs.readFileSync(path, 'utf8');
  return content.split('\n').filter(l => l.trim() !== '');
}

// Finds benchmark names that are in newBenchmarks but not in existingBenchmarks.
// - newBenchmarks: Array of new benchmark names to check.
// - existingBenchmarks: Array of known benchmark names.
// Returns: Array of benchmark names that are missing.
function findMissingBenchmarks(newBenchmarks, existingBenchmarks) {
  return newBenchmarks.filter(b => !existingBenchmarks.includes(b));
}

// For each missing benchmark, finds the closest match from existing benchmarks
// using Levenshtein distance, then sorts results by distance (lowest to highest).
// - missingBenchmarks: Array of benchmark names not in existing list.
// - existingBenchmarks: Array of known benchmark names.
// Returns: Array of objects with name, closest match, and distance properties,
// sorted by distance ascending.
function findNearestBenchmarkNames(missingBenchmarks, existingBenchmarks) {
  return missingBenchmarks.map(missingBench => {
    let closestBench = '';
    let minDistance = Infinity;

    for (const existingBench of existingBenchmarks) {
      const distance = levenshteinDistance(missingBench, existingBench);
      if (distance < minDistance) {
        minDistance = distance;
        closestBench = existingBench;
      }
    }

    return { name: missingBench, closest: closestBench, distance: minDistance };
  }).sort((a, b) => a.distance - b.distance);
}

// Prints the nearest benchmark names as tab-separated columns.
// - nearestNames: Array of objects with name and closest properties.
function printNearestNames(nearestNames) {
  const outputLines = nearestNames.map(r => `${r.name}\t${r.closest}`);
  console.log(outputLines.join('\n'));
}

function showHelp() {
  console.log(`
Usage: node bin/check_bench_names.js <benchmarks.json> <existing-benchmarks.txt>

Arguments:
  <benchmarks.json>          Path to the JSON file containing benchmarks to check
  <existing-benchmarks.txt>  Path to the file containing known benchmark names

Description:
  Finds benchmarks listed in <benchmarks.json> that are not in <existing-benchmarks.txt>.
  For each missing benchmark, outputs the benchmark name and its closest match
  from <existing-benchmarks.txt>, ordered by Levenshtein distance from lowest to highest.

  The output has two tab-separated columns:
  - First column: Benchmark name from benchmarks.json that is not in existing benchmarks
  - Second column: Closest matching benchmark from existing benchmarks (by Levenshtein distance)

Example:
  node bin/check_bench_names.js ./benchmarks.json ./bm

Options:
  --help, -h    Show this help message
`);
}

main();
