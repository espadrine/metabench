// storage.js - Client-side storage for metrics using localStorage

const STORAGE_KEY = 'benchmark-metrics';

// Save metrics to localStorage
// - metrics: Array<{ name: string, criteria: Array<{ bench: string, weight: number }> }>
function saveMetrics(metrics) {
  try {
    const serialized = JSON.stringify(metrics);
    localStorage.setItem(STORAGE_KEY, serialized);
    console.log(`Saved ${metrics.length} metrics to localStorage`);
    return true;
  } catch (error) {
    console.error('Failed to save metrics to localStorage:', error);
    return false;
  }
}

// Load metrics from localStorage
// Returns: Array<{ name: string, criteria: Array<{ bench: string, weight: number }> }> | []
function loadMetrics() {
  try {
    const serialized = localStorage.getItem(STORAGE_KEY);
    if (!serialized) {
      console.log('No metrics found in localStorage');
      return [];
    }
    
    const metrics = JSON.parse(serialized);
    
    // Validate the loaded data structure
    if (!Array.isArray(metrics)) {
      console.warn('Invalid metrics data in localStorage, resetting');
      localStorage.removeItem(STORAGE_KEY);
      return [];
    }
    
    // Validate each metric object
    const validMetrics = metrics.filter(metric => 
      metric && 
      typeof metric.name === 'string' && 
      Array.isArray(metric.criteria) &&
      metric.criteria.every(criterion => 
        criterion && 
        typeof criterion.bench === 'string' && 
        typeof criterion.weight === 'number'
      )
    );
    
    if (validMetrics.length !== metrics.length) {
      console.warn('Some metrics were invalid and filtered out');
    }
    
    console.log(`Loaded ${validMetrics.length} metrics from localStorage`);
    return validMetrics;
  } catch (error) {
    console.error('Failed to load metrics from localStorage:', error);
    // Clear corrupted data
    localStorage.removeItem(STORAGE_KEY);
    return [];
  }
}

// Clear all metrics from localStorage
function clearMetrics() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    console.log('Cleared metrics from localStorage');
    return true;
  } catch (error) {
    console.error('Failed to clear metrics from localStorage:', error);
    return false;
  }
}

// Get storage statistics
function getStorageInfo() {
  try {
    const serialized = localStorage.getItem(STORAGE_KEY);
    if (!serialized) {
      return { exists: false, size: 0 };
    }
    
    const metrics = JSON.parse(serialized);
    return {
      exists: true,
      size: serialized.length,
      metricCount: Array.isArray(metrics) ? metrics.length : 0
    };
  } catch (error) {
    return { exists: false, size: 0, error: error.message };
  }
}