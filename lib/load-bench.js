// Shared library for benchmark loading functions

// Normalize model name for comparison (lowercase, alphanumeric only)
function normalizeModelName(name) {
  return name.toLowerCase()
    .replace(/[0-9]{8}/, '')  // Remove date.
    .replace(/[\-\._]/g, ' ')  // Replace dashes, dots, and underscores with spaces
    .replace(/\s+/g, ' ')     // Collapse multiple spaces
    .trim();                   // Remove leading/trailing spaces
}

// Check if two model names represent an unambiguous match
// Returns true if the models match exactly (normalized) or via known mapping
function isUnambiguousModelMatch(benchmarkModel, ourModel, knownMappings = {}) {
  if (!ourModel) {
    return false;
  }

  const benchmarkName = benchmarkModel.name;
  const ourModelName = ourModel.name;

  // 1. Check for known mappings
  if (knownMappings[benchmarkName] === ourModelName) {
    return true;
  }

  // 2. Check for exact match (case-insensitive, normalized)
  if (normalizeModelName(benchmarkName) === normalizeModelName(ourModelName)) {
    return true;
  }

  // If none of the above criteria are met, consider it ambiguous
  return false;
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

module.exports = {
  normalizeModelName,
  isUnambiguousModelMatch,
  levenshteinDistance
};