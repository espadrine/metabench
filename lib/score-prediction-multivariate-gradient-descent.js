// Multivariate Gradient Descent approach for benchmark score prediction
//
// This approach:
// 1. Initializes scores with weighed bivariate regression
// 2. Trains multivariate regression models on initialized data to get αk and βkj parameters
// 3. Predicts missing scores using trained models
// 4. Uses the multivariate regression formula as a loss: L = Σi (si - (αk + Σj βkj sj))² 
//    where the sum is over known scores only
// 5. Optimizes only αk and βkj parameters using gradient descent
// 6. Predicted scores are Constants recomputed from αk and βkj after each step

const { Variable, Constant } = require('./autograd');

// Estimate missing benchmark scores using multivariate gradient descent.
//
// Parameters:
// - benchmarks: the benchmark scores. {models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]}
// - iterations: number of gradient descent iterations (number, default 10000)
// Returns: estimated benchmark scores ({models: [{name, benchmarks: [{name, score: number, source, stdDev}]}]})
function estimateMissingBenchmarks(benchmarks, iterations = 100) {
  // Index the input data for easier processing
  const indexedData = indexBenchmarkData(benchmarks);

  // Get initial parameters and scores using weighed bivariate regression
  const { parameters, scores } = getInitialParametersAndScores(indexedData);

  // Build computational graph
  const { modelGraph, parameterGraph } = buildComputationalGraph(indexedData, scores, parameters);

  // Perform gradient descent
  performGradientDescent(modelGraph, parameterGraph, indexedData, iterations);

  // Extract final scores and return
  return extractFinalScores(modelGraph, parameterGraph, indexedData);
}

// Get initial parameters and scores using weighed bivariate regression
function getInitialParametersAndScores(indexedData) {
  const weighedBivariate = require('./score-prediction-weighed-bivariate.js');
  const multivariateRegression = require('./score-prediction.js');

  // Initialize scores with weighed bivariate regression
  const initialScores = weighedBivariate.computeAllBenchmarks(deepCopy(indexedData));

  // Create a copy of indexedData with initialized scores
  const initializedData = deepCopy(indexedData);
  for (const [modelName, model] of Object.entries(initialScores.modelFromName)) {
    for (const [benchName, bench] of Object.entries(model)) {
      if (initializedData.modelFromName[modelName][benchName].score == null) {
        initializedData.modelFromName[modelName][benchName].score = bench.score;
      }
    }
  }

  // Get parameters by training models on the initialized data
  const parameters = {};
  const finalScores = deepCopy(initializedData);

  for (const benchName of indexedData.benchmarkNames) {
    const regressionResult = multivariateRegression.trainModelForBenchmark(initializedData, benchName);
    parameters[benchName] = {
      alpha: regressionResult.bias,
      betas: regressionResult.coefficients
    };

    // Predict missing scores using the trained model
    for (const [modelName, model] of Object.entries(initializedData.modelFromName)) {
      if (initializedData.modelFromName[modelName][benchName].score == null) {
        const prediction = multivariateRegression.predictMissingScore(model, benchName, regressionResult);
        finalScores.modelFromName[modelName][benchName].score = prediction.prediction;
      }
    }
  }

  return { parameters, scores: finalScores };
}

// Build computational graph with variables for parameters and constants for scores
function buildComputationalGraph(indexedData, initialScores, initialParameters) {
  const modelGraph = {};
  const parameterGraph = {};

  // Build model graph: all scores are Constants (known scores from original data, 
  // predicted scores from weighed bivariate + multivariate regression initialization)
  for (const [modelName, model] of Object.entries(indexedData.modelFromName)) {
    modelGraph[modelName] = {};
    for (const benchName of indexedData.benchmarkNames) {
      const bench = model[benchName];
      if (bench && bench.score != null) {
        // Known score - use original value
        modelGraph[modelName][benchName] = new Constant(bench.score);
      } else {
        // Missing score - use initialized estimate
        const initVal = initialScores.modelFromName[modelName][benchName].score;
        modelGraph[modelName][benchName] = new Variable(initVal);
      }
    }
  }

  // Build parameter graph: αk (bias) and βkj (coefficients) for each benchmark
  for (const benchName of indexedData.benchmarkNames) {
    const initialParams = initialParameters[benchName];
    parameterGraph[benchName] = {
      alpha: new Variable(initialParams.alpha), // αk (bias)
      betas: {}
    };

    // Initialize βkj coefficients from multivariate regression
    for (const otherBench of indexedData.benchmarkNames) {
      if (otherBench !== benchName) {
        const betaValue = initialParams.betas[otherBench] || 0;
        parameterGraph[benchName].betas[otherBench] = new Variable(betaValue);
      }
    }
  }

  return { modelGraph, parameterGraph };
}

// Perform gradient descent optimization
function performGradientDescent(modelGraph, parameterGraph, indexedData, iterations) {
  // Collect only parameter variables for gradient descent (not score variables)
  const parameterVariables = collectParameterVariables(parameterGraph);
  const scoreVariables = collectScoreVariables(modelGraph);
  const variables = parameterVariables.concat(scoreVariables);

  let stepSize = 1;
  for (let i = 0; i < iterations; i++) {
    // Compute loss and gradients
    const loss = computeMultivariateLoss(modelGraph, parameterGraph, indexedData);
    loss.computeGradients();
    console.error(`Iteration ${i + 1}/${iterations} loss: ${loss.value} step size ${stepSize}`);

    // Find a step size that will reduce the loss.
    let newLoss = loss;
    let modelGraphCopy = copyModelGraph(modelGraph);
    let parameterGraphCopy = copyParameterGraph(parameterGraph);
    let parameterVariablesCopy = collectParameterVariables(parameterGraphCopy);
    let scoreVariablesCopy = collectScoreVariables(modelGraphCopy);
    let variablesCopy = parameterVariablesCopy.concat(scoreVariablesCopy);
    for (let j = 0; j < 100 && newLoss.value >= loss.value; j++) {
      gradientDescent(variablesCopy, stepSize);
      newLoss = computeMultivariateLoss(modelGraphCopy, parameterGraphCopy, indexedData);
      if (newLoss.value >= loss.value || Number.isNaN(newLoss)) {
        // This step would be worse; we restore variables.
        modelGraphCopy = copyModelGraph(modelGraph);
        parameterGraphCopy = copyParameterGraph(parameterGraph);
        parameterVariablesCopy = collectParameterVariables(parameterGraphCopy);
        scoreVariablesCopy = collectScoreVariables(modelGraphCopy);
        variablesCopy = parameterVariablesCopy.concat(scoreVariablesCopy);
        stepSize /= 2;
      }
    }
    if (newLoss.value >= loss.value || Number.isNaN(newLoss.value)) { break; }

    gradientDescent(variables, stepSize);
  }
  updatePredictedScores(modelGraph, parameterGraph, indexedData);
}

// Update predicted scores based on current parameter values
function updatePredictedScores(modelGraph, parameterGraph, indexedData) {
  for (const [modelName, model] of Object.entries(modelGraph)) {
    for (const benchName of indexedData.benchmarkNames) {
      const originalBench = indexedData.modelFromName[modelName][benchName];

      // Only update predicted scores (not known scores)
      if (originalBench.score == null) {
        const predictedScore = computePredictedScoreValue(model, benchName, parameterGraph[benchName]);
        modelGraph[modelName][benchName].value = predictedScore;
      }
    }
  }
}

// Compute predicted score value (not computational graph node)
function computePredictedScoreValue(model, targetBench, parameters) {
  let prediction = parameters.alpha.value;

  for (const [benchName, beta] of Object.entries(parameters.betas)) {
    if (benchName !== targetBench) {
      const score = model[benchName].value;
      prediction += beta.value * score;
    }
  }

  return prediction;
}

// Compute multivariate loss: L = Σi (si - (αk + Σj βkj sj))² over known scores only
function computeMultivariateLoss(modelGraph, parameterGraph, indexedData) {
  let loss = new Constant(0);

  for (const [modelName, model] of Object.entries(modelGraph)) {
    for (const benchName of indexedData.benchmarkNames) {
      const originalBench = indexedData.modelFromName[modelName][benchName];

      // Only include known scores in the loss
      if (originalBench.score != null) {
        const actualScore = model[benchName];
        const predictedScore = computePredictedScore(model, benchName, parameterGraph[benchName]);
        const error = actualScore.subtract(predictedScore);
        const squaredError = error.power(2);
        loss = loss.add(squaredError);
      }
    }
  }

  return loss;
}

// Compute predicted score for a benchmark: αk + Σj βkj sj (computational graph version)
function computePredictedScore(model, targetBench, parameters) {
  let prediction = parameters.alpha;

  for (const [benchName, beta] of Object.entries(parameters.betas)) {
    if (benchName !== targetBench) {
      const score = model[benchName];
      prediction = prediction.add(beta.multiply(score));
    }
  }

  return prediction;
}

function copyModelGraph(modelGraph) {
  const copy = {};
  for (const [modelName, model] of Object.entries(modelGraph)) {
    copy[modelName] = {};
    for (const [benchName, node] of Object.entries(model)) {
      if (node instanceof Variable) {
        copy[modelName][benchName] = new Variable(node.value);
        copy[modelName][benchName].gradient = node.gradient;
      } else if (node instanceof Constant) {
        copy[modelName][benchName] = new Constant(node.value);
        copy[modelName][benchName].gradient = node.gradient;
      } else {
        throw new Error('Unknown node type in model graph');
      }
    }
  }
  return copy;
}

function copyParameterGraph(parameterGraph) {
  const copy = {};
  for (const [benchName, params] of Object.entries(parameterGraph)) {
    copy[benchName] = {
      alpha: new Variable(params.alpha.value),
      betas: {}
    };
    copy[benchName].alpha.gradient = params.alpha.gradient;
    for (const [otherBench, beta] of Object.entries(params.betas)) {
      copy[benchName].betas[otherBench] = new Variable(beta.value);
      copy[benchName].betas[otherBench].gradient = beta.gradient;
    }
  }
  return copy;
}

// Collect only parameter variables for gradient descent
function collectParameterVariables(parameterGraph) {
  const variables = [];

  // Collect parameter variables (αk and βkj)
  for (const params of Object.values(parameterGraph)) {
    if (params.alpha instanceof Variable) {
      variables.push(params.alpha);
    }
    for (const beta of Object.values(params.betas)) {
      if (beta instanceof Variable) {
        variables.push(beta);
      }
    }
  }

  return variables;
}

function collectScoreVariables(modelGraph) {
  const variables = [];

  // Collect score variables (predicted scores only)
  for (const model of Object.values(modelGraph)) {
    for (const scoreNode of Object.values(model)) {
      if (scoreNode instanceof Variable) {
        variables.push(scoreNode);
      }
    }
  }

  return variables;
}

// Perform gradient descent step
function gradientDescent(variables, stepSize) {
  // Compute gradient norm
  let norm = 0;
  for (const node of variables) {
    norm += node.gradient * node.gradient;
  }
  norm = Math.sqrt(norm);

  // Avoid division by zero
  if (norm < 1e-10) { return; }

  const lr = stepSize / norm;
  for (const node of variables) {
    node.value -= lr * node.gradient;
  }
}

// Extract final scores from computational graph
function extractFinalScores(modelGraph, parameterGraph, indexedData) {
  for (const [modelName, model] of Object.entries(modelGraph)) {
    for (const [benchName, node] of Object.entries(model)) {
      const score = node.value;
      indexedData.modelFromName[modelName][benchName].score = score;
    }
  }

  return unindexBenchmarkData(indexedData);
}

// Helper functions (copied from existing implementations)

function indexBenchmarkData(benchmarks) {
  benchmarks = deepCopy(benchmarks);
  const benchmarkNames = new Set();
  const modelNames = new Set();
  for (const model of benchmarks.models) {
    modelNames.add(model.name);
    for (const benchmark of model.benchmarks) {
      benchmarkNames.add(benchmark.name);
    }
  }

  const modelFromName = {};
  for (const model of modelNames) {
    modelFromName[model] = {};
    for (const bench of benchmarkNames) {
      modelFromName[model][bench] = [];
    }
  }

  for (const model of benchmarks.models) {
    for (const benchmark of model.benchmarks) {
      modelFromName[model.name][benchmark.name].push(benchmark);
    }
  }

  // Merge scores for a given benchmark and model
  for (const model of modelNames) {
    for (const bench of benchmarkNames) {
      const entries = modelFromName[model][bench];
      if (entries.length > 1) {
        let sumScores = 0;
        for (const entry of entries) {
          sumScores += entry.score;
        }
        const avgScore = sumScores / entries.length;

        let sumErrors = 0;
        for (const entry of entries) {
          const error = (entry.score - avgScore);
          sumErrors += error * error;
        }
        const stdDev = Math.sqrt(sumErrors / (entries.length - 1));

        const source = 'Multiple: ' + entries.map(e => `Score ${e.score} at ${e.source}`).join('; ');

        modelFromName[model][bench] = {
          name: bench,
          score: avgScore,
          source: source,
          stdDev: stdDev,
        };
      } else if (entries.length === 1) {
        const { name, score, source, stdDev } = entries[0];
        modelFromName[model][bench] = {
          name,
          score,
          source: source || 'Original',
          stdDev: stdDev || 0,
        };
      } else {
        modelFromName[model][bench] = { name: bench, score: null, source: 'Multivariate gradient descent', stdDev: 0 };
      }
    }
  }

  return { benchmarkNames: Array.from(benchmarkNames), modelFromName };
}

function unindexBenchmarkData(benchmarks) {
  const models = [];
  for (const modelName in benchmarks.modelFromName) {
    const modelBenchmarks = [];
    for (const benchName of benchmarks.benchmarkNames) {
      modelBenchmarks.push(benchmarks.modelFromName[modelName][benchName]);
    }
    models.push({
      name: modelName,
      benchmarks: modelBenchmarks,
    });
  }
  return { models };
}

function deepCopy(data) {
  return JSON.parse(JSON.stringify(data));
}

module.exports = {
  estimateMissingBenchmarks
};
