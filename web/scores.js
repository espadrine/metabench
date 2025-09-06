// scores.js – fetches the predicted scores JSON, renders a sortable leaderboard,
// and provides UI to configure weighted sorting.

// ----- Rendering -----

function render(state, widgets) {
  renderSortControls(state, widgets);
  renderTable(state, widgets);
}

// Leaderboard table.
// - state:
//   - models: List of { name, benchmarks: { score: number, stddev: number } }
//   - benchmarkNames: list of benchmark names (strings)
//   - sortingCriteria: array of { bench: string, weight: number }
// - widgets: { container, sortContainer }
function renderTable(state, widgets) {
  // Build table.
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  const thModel = document.createElement('th');
  thModel.textContent = 'Model';
  headerRow.appendChild(thModel);
  state.benchmarkNames.forEach((b) => {
    const th = document.createElement('th');
    th.textContent = b;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  state.models.forEach(({ name, benchmarks }) => {
    const row = document.createElement('tr');
    const tdModel = document.createElement('td');
    tdModel.textContent = name;
    row.appendChild(tdModel);
    state.benchmarkNames.forEach((b) => {
      const td = document.createElement('td');
      const entry = benchmarks[b];
      if (entry) {
        const { score, stddev, source } = entry;
        const fmtScore = Number(score.toFixed(2));
        if (stddev && stddev > 0) {
          const twoSigma = 2 * stddev;
          td.textContent = `${fmtScore}±${Number(twoSigma.toFixed(2))}`;
        } else {
          td.textContent = `${fmtScore}`;
        }
        // Tooltip showing source of benchmark evaluation
        if (source) {
          td.title = source;
        }
      } else {
        td.textContent = '';
      }
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);

  // Replace previous content.
  widgets.container.innerHTML = '';
  widgets.container.appendChild(table);
}

// Control to determine which benchmarks we sort the table by.
function renderSortControls(state, widgets) {
  // Clear container.
  widgets.sortContainer.innerHTML = '';
  const title = document.createElement('div');
  title.textContent = 'Sort By:';
  title.style.fontWeight = 'bold';
  widgets.sortContainer.appendChild(title);

  // List of active criteria.
  const list = document.createElement('ul');
  list.style.listStyle = 'none';
  list.style.padding = '0';
  state.sortingCriteria.forEach((c, idx) => {
    const li = document.createElement('li');
    li.style.marginBottom = '4px';
    const label = document.createElement('span');
    label.textContent = `${c.bench} (weight ${c.weight}) `;
    const btn = document.createElement('button');
    btn.textContent = '✕';
    btn.title = 'Remove';
    btn.style.marginLeft = '8px';
    btn.addEventListener('click', () => removeCriterion(state, widgets, idx));
    li.appendChild(label);
    li.appendChild(btn);
    list.appendChild(li);
  });
  widgets.sortContainer.appendChild(list);

  // Form to add a new criterion.
  const form = document.createElement('div');
  form.style.marginTop = '8px';
  const select = document.createElement('select');
  state.benchmarkNames.forEach((b) => {
    const opt = document.createElement('option');
    opt.value = b;
    opt.textContent = b;
    select.appendChild(opt);
  });
  const weightInput = document.createElement('input');
  weightInput.type = 'number';
  weightInput.min = '0';
  weightInput.step = 'any';
  weightInput.value = '1';
  weightInput.style.width = '60px';
  weightInput.title = 'Weight';
  const addBtn = document.createElement('button');
  addBtn.textContent = 'Add benchmark to sort';
  addBtn.style.marginLeft = '4px';
  addBtn.addEventListener('click', () => {
    const bench = select.value;
    const weight = parseFloat(weightInput.value);
    addCriterion(state, widgets, bench, weight);
    render(state, widgets);
  });
  // Label for weight input
  const weightLabel = document.createElement('label');
  weightLabel.textContent = 'Weight:';
  weightLabel.style.marginLeft = '8px';
  weightLabel.appendChild(weightInput);

  form.appendChild(select);
  form.appendChild(weightLabel);
  form.appendChild(addBtn);
  widgets.sortContainer.appendChild(form);
}

// ----- State changes -----

function addCriterion(state, widgets, bench, weight) {
  if (!bench || isNaN(weight) || weight <= 0) return;

  // If already present, update weight.
  const existingIdx = state.sortingCriteria.findIndex((c) => c.bench === bench);
  if (existingIdx >= 0) {
    state.sortingCriteria[existingIdx].weight = weight;
  } else {
    state.sortingCriteria.push({ bench, weight });
  }

  sortModels(state, widgets);
  renderSortControls(state, widgets);
}

function removeCriterion(state, widgets, idx) {
  state.sortingCriteria.splice(idx, 1);
  sortModels(state);
  renderSortControls(state, widgets);
}

function sortModels(state, widgets) {
  if (state.sortingCriteria.length > 0) {
    state.models.sort((a, b) => {
      const scoreA = computeWeightedScore(a.benchmarks, state.sortingCriteria);
      const scoreB = computeWeightedScore(b.benchmarks, state.sortingCriteria);
      return scoreB - scoreA;
    });
  } else {
    // Fallback: alphabetical order for deterministic output.
    state.models.sort((a, b) => a.name.localeCompare(b.name));
  }
  renderTable(state, widgets);
}

// ----- Utility functions -----

// Returns a list of models { name, benchmarks: { bench: { score, stddev } } }.
async function fetchScores() {
  // The HTML file lives in web/, the JSON is now in the sibling data/
  // directory and has the structure:
  // { models: [ { name, benchmarks: [ { name, score, source, stdDev } ] } ] }
  const response = await fetch('../data/models-prediction.json');
  if (!response.ok) throw new Error(`Failed to load JSON: ${response.status}`);
  const data = await response.json();
  return data.models.map(model => ({
    name: model.name,
    benchmarks: model.benchmarks.reduce((acc, b) => {
      acc[b.name] = {
        score: b.score,
        stddev: b.stdDev,
        source: b.source,
      };
      return acc;
    }, {}),
  }));
}

// Determine the full set of benchmark names across all models.
// - models: list of { name: model name, benchmarks: { bench: { score: number, stddev: number } } }
// Returns a sorted array of benchmark names.
function gatherBenchmarkNames(models) {
  const benchSet = new Set();
  for (const model of models) {
    Object.keys(model.benchmarks).forEach(b => benchSet.add(b));
  }
  return Array.from(benchSet).sort();
}

// Compute weighted average score for a single model based on current criteria.
// - modelData: { bench: { score: number, stddev: number } }
// - sortingCriteria: array of { bench: string, weight: number }
function computeWeightedScore(modelData, sortingCriteria) {
  if (sortingCriteria.length === 0) return 0;
  let sum = 0;
  let weightSum = 0;
  sortingCriteria.forEach(({ bench, weight }) => {
    const entry = modelData[bench];
    if (entry && typeof entry.score === 'number') {
      sum += weight * entry.score;
      weightSum += weight;
    }
  });
  return weightSum === 0 ? 0 : sum / weightSum;
}


(async () => {
  // ----- DOM elements -----
  const widgets = {
    container: document.getElementById('leaderboard'),
    sortContainer: document.getElementById('sort-controls'),
  }
  for (const [key, el] of Object.entries(widgets)) {
    if (!el) {
      console.error(`Missing DOM element: ${key}`);
      return;
    }
  }

  // ----- State -----
  const state = {};

  // ----- Main flow -----
  try {
    // List of { name, benchmarks: { score: number, stddev: number } }
    state.models = await fetchScores();
    // List of string
    state.benchmarkNames = gatherBenchmarkNames(state.models);
    // Array of { bench: string, weight: number }
    state.sortingCriteria = [];

    render(state, widgets);
  } catch (err) {
    console.error('Error loading leaderboard data:', err);
    widgets.container.textContent = 'Failed to load leaderboard.';
  }
})();
