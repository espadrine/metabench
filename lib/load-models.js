// Load and aggregate model data from multiple company files
const fs = require('fs');
const path = require('path');

// Load all models from the data/models directory
function loadModels() {
  const modelsDir = path.resolve('./data/models');
  const models = { models: [] };

  try {
    // Read all JSON files in the models directory
    const files = fs.readdirSync(modelsDir);

    for (const file of files) {
      if (file.endsWith('.json')) {
        const filePath = path.join(modelsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const companyModels = JSON.parse(content);

        // Add models from this company file
        if (companyModels.models && Array.isArray(companyModels.models)) {
          models.models.push(...companyModels.models);
        }
      }
    }

    console.error(`Loaded ${models.models.length} models from ${files.length} company files`);
    return models;
  } catch (error) {
    console.error(`Error loading models: ${error.message}`);
    throw error;
  }
}

module.exports = { loadModels };
