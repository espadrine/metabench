const { test } = require('node:test');
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// Mock filesystem for testing
test('extractBenchmarkNamesFromModelFiles should extract benchmark names correctly', () => {
  // Create a temporary test directory
  const testDir = './test_temp_models/';
  
  try {
    // Create test directory
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir);
    }
    
    // Create test JSON files
    fs.writeFileSync(path.join(testDir, 'company1.json'), JSON.stringify({
      models: [
        {
          name: "Model 1",
          benchmarks: [
            { name: "Benchmark 1", score: 85.5, source: "Source A" },
            { name: "Benchmark 2", score: 92.3, source: "Source B" }
          ]
        }
      ]
    }));
    
    fs.writeFileSync(path.join(testDir, 'company2.json'), JSON.stringify({
      models: [
        {
          name: "Model 2",
          benchmarks: [
            { name: "Benchmark 1", score: 88.2, source: "Source C" },
            { name: "Benchmark 3", score: 65.2, source: "Source D" }
          ]
        }
      ]
    }));
    
    // Test the function
    const result = extractBenchmarkNamesFromModelFiles(testDir);
    
    assert.deepStrictEqual(result, {
      "Benchmark 1": 2,  // Appears in both files
      "Benchmark 2": 1,  // Only in company1
      "Benchmark 3": 1   // Only in company2
    });
    
  } finally {
    // Clean up - remove test directory
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  }
});

test('extractBenchmarkNamesFromModelFiles should handle empty directory', () => {
  const testDir = './test_temp_empty/';
  
  try {
    // Create empty test directory
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir);
    }
    
    const result = extractBenchmarkNamesFromModelFiles(testDir);
    assert.deepStrictEqual(result, {});
    
  } finally {
    // Clean up
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  }
});

test('extractBenchmarkNamesFromModelFiles should handle files with missing benchmarks', () => {
  const testDir = './test_temp_missing/';
  
  try {
    // Create test directory
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir);
    }
    
    // Create test JSON file without benchmarks
    fs.writeFileSync(path.join(testDir, 'test.json'), JSON.stringify({
      models: [
        {
          name: "Model Without Benchmarks",
          // No benchmarks array
        }
      ]
    }));
    
    const result = extractBenchmarkNamesFromModelFiles(testDir);
    assert.deepStrictEqual(result, {});
    
  } finally {
    // Clean up
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  }
});

// Helper function (copied from the script)
function extractBenchmarkNamesFromModelFiles(modelsDir) {
  const benchmarkCounts = {};
  
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
        // Continue with other files even if one fails
        console.error(`⚠️  Warning: Could not process file ${file}: ${error.message}`);
      }
    }
  }
  
  return benchmarkCounts;
}