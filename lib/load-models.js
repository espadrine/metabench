// Load and aggregate model data from multiple company files
const fs = require('fs');
const path = require('path');

// Load all models from the data/models directory
function loadModels() {
  const modelsDir = path.resolve('./data/models');
  const models = { models: [] };

  // Read all JSON files in the models directory
  const files = fs.readdirSync(modelsDir);

  for (const file of files) {
    if (file.endsWith('.json')) {
      const filePath = path.join(modelsDir, file);
      let content, companyModels;
      try {
        content = fs.readFileSync(filePath, 'utf8');
        companyModels = JSON.parse(content);
      } catch (error) {
        console.error(`Could not load file ${file}: ${error.message}`);
        throw error;
      }

      // Add models from this company file
      if (companyModels.models && Array.isArray(companyModels.models)) {
        models.models.push(...companyModels.models);
      }
    }
  }

  console.error(`Loaded ${models.models.length} models from ${files.length} company files`);
  return models;
}

module.exports = { loadModels };
