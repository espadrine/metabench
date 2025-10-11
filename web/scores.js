// scores.js – fetches the predicted scores JSON, renders a sortable leaderboard,
// and provides UI to configure weighted metrics.

// ----- State -----
let state = {
  // List of { name: string, benchmarks: { bench: { score: number, stddev: number, source: string } } }
  models: [],
  // List of string - all unique benchmark names across all models
  benchmarkNames: [],
  // List of { name: string, criteria: Array<{ bench: string, weight: number }> }
  metrics: [],
  // number | null - index of currently selected metric in metrics array
  currentMetricIndex: null,
  // Array<{ bench: string, weight: number }> - criteria for new metric being created
  newMetricCriteria: [],
  // number | null - index of metric being edited in metrics array
  editingMetricIndex: null,
  // Array<{ bench: string, weight: number }> - criteria for metric being edited
  editMetricCriteria: []
};

// ----- DOM elements -----
const widgets = {
  container: document.getElementById('leaderboard'),
  metricControls: document.getElementById('metric-controls'),
  newMetricModal: document.getElementById('new-metric-modal'),
  newMetricForm: document.getElementById('new-metric-form'),
  closeNewMetric: document.getElementById('close-new-metric'),
  editMetricModal: document.getElementById('edit-metric-modal'),
  editMetricForm: document.getElementById('edit-metric-form'),
  closeEditMetric: document.getElementById('close-edit-metric')
};

// ----- Rendering -----

function render(state, widgets) {
  renderMetricControls(state, widgets);
  renderTable(state, widgets);
}

// Metric controls at the top of the page
function renderMetricControls(state, widgets) {
  widgets.metricControls.innerHTML = '';

  // Metrics dropdown (only show if metrics exist)
  if (state.metrics.length > 0) {
    const dropdownContainer = document.createElement('div');
    dropdownContainer.style.display = 'inline-block';
    dropdownContainer.style.marginRight = '16px';

    const label = document.createElement('label');
    label.textContent = 'Current Metric: ';
    label.style.marginRight = '8px';
    dropdownContainer.appendChild(label);

    const select = document.createElement('select');
    state.metrics.forEach((metric, index) => {
      const option = document.createElement('option');
      option.value = index;
      option.textContent = metric.name;
      // Select the first metric by default if no current metric is set
      if (index === 0 && state.currentMetricIndex === null) {
        option.selected = true;
        state.currentMetricIndex = 0;
      } else if (index === state.currentMetricIndex) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
      state.currentMetricIndex = parseInt(e.target.value);
      render(state, widgets);
    });

    dropdownContainer.appendChild(select);

    // Edit metric button
    const editBtn = document.createElement('button');
    editBtn.textContent = 'Edit metric';
    editBtn.style.marginLeft = '8px';
    editBtn.addEventListener('click', () => showEditMetricModal(state, widgets));
    dropdownContainer.appendChild(editBtn);

    widgets.metricControls.appendChild(dropdownContainer);
  }

  // Add metric button
  const addMetricBtn = document.createElement('button');
  addMetricBtn.textContent = 'Add metric';
  addMetricBtn.addEventListener('click', () => showNewMetricModal(state, widgets));
  widgets.metricControls.appendChild(addMetricBtn);
}

// Leaderboard table.
function renderTable(state, widgets) {
  // Determine column order
  let columnOrder = ['Model'];

  if (state.currentMetricIndex !== null && state.metrics[state.currentMetricIndex]) {
    const metric = state.metrics[state.currentMetricIndex];
    // When a metric is active, only show columns associated with that metric
    const sortedCriteria = [...metric.criteria].sort((a, b) => b.weight - a.weight);
    columnOrder = ['Model', ...sortedCriteria.map(c => c.bench)];
  } else {
    // When no metric is active, show all benchmarks
    columnOrder = ['Model', ...state.benchmarkNames];
  }

  // Build table.
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');

  columnOrder.forEach((columnName) => {
    const th = document.createElement('th');
    th.textContent = columnName;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');

  // Sort models if we have a current metric
  let modelsToRender = [...state.models];
  if (state.currentMetricIndex !== null && state.metrics[state.currentMetricIndex]) {
    const metric = state.metrics[state.currentMetricIndex];
    modelsToRender.sort((a, b) => {
      const scoreA = computeWeightedScore(a.benchmarks, metric.criteria);
      const scoreB = computeWeightedScore(b.benchmarks, metric.criteria);
      return scoreB - scoreA;
    });
  } else {
    // Fallback: alphabetical order
    modelsToRender.sort((a, b) => a.name.localeCompare(b.name));
  }

  modelsToRender.forEach(({ name, benchmarks }) => {
    const row = document.createElement('tr');

    columnOrder.forEach((columnName) => {
      const td = document.createElement('td');

      if (columnName === 'Model') {
        td.textContent = name;
      } else {
        const entry = benchmarks[columnName];
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

// ----- Modal functions -----

function showNewMetricModal(state, widgets) {
  state.newMetricCriteria = [];
  renderNewMetricForm(state, widgets);
  widgets.newMetricModal.style.display = 'block';
}

function closeNewMetricModal() {
  widgets.newMetricModal.style.display = 'none';
}

function showEditMetricModal(state, widgets) {
  if (state.currentMetricIndex === null) return;

  const metric = state.metrics[state.currentMetricIndex];
  if (!metric) return;

  state.editingMetricIndex = state.currentMetricIndex;
  state.editMetricCriteria = [...metric.criteria];
  renderEditMetricForm(state, widgets);
  widgets.editMetricModal.style.display = 'block';
}

function closeEditMetricModal() {
  widgets.editMetricModal.style.display = 'none';
}

function renderNewMetricForm(state, widgets) {
  // Store current name input value if it exists
  const currentName = widgets.newMetricForm.querySelector('input[type="text"]')?.value || '';

  widgets.newMetricForm.innerHTML = '';

  // Metric name input
  const nameLabel = document.createElement('label');
  nameLabel.textContent = 'Metric Name:';
  nameLabel.style.display = 'block';
  nameLabel.style.marginBottom = '8px';
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.placeholder = 'Enter metric name';
  nameInput.value = currentName; // Preserve the existing value
  nameInput.style.width = '100%';
  nameInput.style.marginBottom = '16px';
  widgets.newMetricForm.appendChild(nameLabel);
  widgets.newMetricForm.appendChild(nameInput);

  // Sort by section title
  const sortTitle = document.createElement('div');
  sortTitle.textContent = 'Sort by:';
  sortTitle.style.fontWeight = 'bold';
  sortTitle.style.marginBottom = '8px';
  widgets.newMetricForm.appendChild(sortTitle);

  // List of active criteria
  const list = document.createElement('ul');
  list.style.listStyle = 'none';
  list.style.padding = '0';
  state.newMetricCriteria.forEach((c, idx) => {
    const li = document.createElement('li');
    li.style.marginBottom = '4px';
    const label = document.createElement('span');
    label.textContent = `${c.bench} (weight ${c.weight}) `;
    const btn = document.createElement('button');
    btn.textContent = '✕';
    btn.title = 'Remove';
    btn.style.marginLeft = '8px';
    btn.addEventListener('click', () => removeNewMetricCriterion(state, widgets, idx));
    li.appendChild(label);
    li.appendChild(btn);
    list.appendChild(li);
  });
  widgets.newMetricForm.appendChild(list);

  // Form to add a new criterion
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
  addBtn.textContent = 'Add benchmark to metric';
  addBtn.style.marginLeft = '4px';
  addBtn.addEventListener('click', () => {
    const bench = select.value;
    const weight = parseFloat(weightInput.value);
    addNewMetricCriterion(state, widgets, bench, weight);
  });

  // Label for weight input
  const weightLabel = document.createElement('label');
  weightLabel.textContent = 'Weight:';
  weightLabel.style.marginLeft = '8px';
  weightLabel.appendChild(weightInput);

  form.appendChild(select);
  form.appendChild(weightLabel);
  form.appendChild(addBtn);
  widgets.newMetricForm.appendChild(form);

  // Add metric button
  const addMetricBtn = document.createElement('button');
  addMetricBtn.textContent = 'Add metric';
  addMetricBtn.style.marginTop = '16px';
  addMetricBtn.style.display = 'block';
  addMetricBtn.addEventListener('click', () => {
    const metricName = nameInput.value.trim();
    if (!metricName) {
      alert('Please enter a metric name');
      return;
    }
    if (state.newMetricCriteria.length === 0) {
      alert('Please add at least one benchmark to the metric');
      return;
    }

    // Add the new metric
    state.metrics.push({
      name: metricName,
      criteria: [...state.newMetricCriteria]
    });

    // Set as current metric
    state.currentMetricIndex = state.metrics.length - 1;

    closeNewMetricModal();
    render(state, widgets);
  });
  widgets.newMetricForm.appendChild(addMetricBtn);
}

function renderEditMetricForm(state, widgets) {
  widgets.editMetricForm.innerHTML = '';

  const metric = state.metrics[state.editingMetricIndex];
  if (!metric) return;

  // Metric name input
  const nameLabel = document.createElement('label');
  nameLabel.textContent = 'Metric Name:';
  nameLabel.style.display = 'block';
  nameLabel.style.marginBottom = '8px';
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = metric.name;
  nameInput.style.width = '100%';
  nameInput.style.marginBottom = '16px';
  widgets.editMetricForm.appendChild(nameLabel);
  widgets.editMetricForm.appendChild(nameInput);

  // Sort by section title
  const sortTitle = document.createElement('div');
  sortTitle.textContent = 'Sort by:';
  sortTitle.style.fontWeight = 'bold';
  sortTitle.style.marginBottom = '8px';
  widgets.editMetricForm.appendChild(sortTitle);

  // List of active criteria
  const list = document.createElement('ul');
  list.style.listStyle = 'none';
  list.style.padding = '0';
  state.editMetricCriteria.forEach((c, idx) => {
    const li = document.createElement('li');
    li.style.marginBottom = '4px';
    const label = document.createElement('span');
    label.textContent = `${c.bench} (weight ${c.weight}) `;
    const btn = document.createElement('button');
    btn.textContent = '✕';
    btn.title = 'Remove';
    btn.style.marginLeft = '8px';
    btn.addEventListener('click', () => removeEditMetricCriterion(state, widgets, idx));
    li.appendChild(label);
    li.appendChild(btn);
    list.appendChild(li);
  });
  widgets.editMetricForm.appendChild(list);

  // Form to add a new criterion
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
  addBtn.textContent = 'Add benchmark to metric';
  addBtn.style.marginLeft = '4px';
  addBtn.addEventListener('click', () => {
    const bench = select.value;
    const weight = parseFloat(weightInput.value);
    addEditMetricCriterion(state, widgets, bench, weight);
  });

  // Label for weight input
  const weightLabel = document.createElement('label');
  weightLabel.textContent = 'Weight:';
  weightLabel.style.marginLeft = '8px';
  weightLabel.appendChild(weightInput);

  form.appendChild(select);
  form.appendChild(weightLabel);
  form.appendChild(addBtn);
  widgets.editMetricForm.appendChild(form);

  // Save metric button
  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save metric';
  saveBtn.style.marginTop = '16px';
  saveBtn.style.marginRight = '8px';
  saveBtn.addEventListener('click', () => {
    const newName = nameInput.value.trim();
    if (!newName) {
      alert('Please enter a metric name');
      return;
    }
    if (state.editMetricCriteria.length === 0) {
      alert('Please add at least one benchmark to the metric');
      return;
    }

    // Update the metric
    if (state.editingMetricIndex !== null) {
      state.metrics[state.editingMetricIndex].name = newName;
      state.metrics[state.editingMetricIndex].criteria = [...state.editMetricCriteria];

      // Update current metric if it was the one being edited
      if (state.currentMetricIndex === state.editingMetricIndex) {
        // No need to update index, just the name in the metric object
      }
    }

    closeEditMetricModal();
    render(state, widgets);
  });
  widgets.editMetricForm.appendChild(saveBtn);

  // Delete metric button
  const deleteBtn = document.createElement('button');
  deleteBtn.textContent = 'Delete metric';
  deleteBtn.addEventListener('click', () => {
    if (confirm(`Are you sure you want to delete the metric "${metric.name}"?`)) {
      if (state.editingMetricIndex !== null) {
        state.metrics.splice(state.editingMetricIndex, 1);

        // Update current metric if it was the one being deleted
        if (state.currentMetricIndex === state.editingMetricIndex) {
          state.currentMetricIndex = state.metrics.length > 0 ? 0 : null;
        }
      }

      closeEditMetricModal();
      render(state, widgets);
    }
  });
  widgets.editMetricForm.appendChild(deleteBtn);
}

// ----- New metric criteria management -----

function addNewMetricCriterion(state, widgets, bench, weight) {
  if (!bench || isNaN(weight) || weight <= 0) return;

  // If already present, update weight
  const existingIdx = state.newMetricCriteria.findIndex((c) => c.bench === bench);
  if (existingIdx >= 0) {
    state.newMetricCriteria[existingIdx].weight = weight;
  } else {
    state.newMetricCriteria.push({ bench, weight });
  }

  renderNewMetricForm(state, widgets);
}

function removeNewMetricCriterion(state, widgets, idx) {
  state.newMetricCriteria.splice(idx, 1);
  renderNewMetricForm(state, widgets);
}

// ----- Edit metric criteria management -----

function addEditMetricCriterion(state, widgets, bench, weight) {
  if (!bench || isNaN(weight) || weight <= 0) return;

  // If already present, update weight
  const existingIdx = state.editMetricCriteria.findIndex((c) => c.bench === bench);
  if (existingIdx >= 0) {
    state.editMetricCriteria[existingIdx].weight = weight;
  } else {
    state.editMetricCriteria.push({ bench, weight });
  }

  renderEditMetricForm(state, widgets);
}

function removeEditMetricCriterion(state, widgets, idx) {
  state.editMetricCriteria.splice(idx, 1);
  renderEditMetricForm(state, widgets);
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

// ----- Event listeners -----

widgets.closeNewMetric.addEventListener('click', closeNewMetricModal);
widgets.closeEditMetric.addEventListener('click', closeEditMetricModal);

// Close modal when clicking outside
window.addEventListener('click', (event) => {
  if (event.target === widgets.newMetricModal) {
    closeNewMetricModal();
  }
  if (event.target === widgets.editMetricModal) {
    closeEditMetricModal();
  }
});

// ----- Main flow -----

(async () => {
  for (const [key, el] of Object.entries(widgets)) {
    if (!el) {
      console.error(`Missing DOM element: ${key}`);
      return;
    }
  }

  try {
    // List of { name, benchmarks: { score: number, stddev: number } }
    state.models = await fetchScores();
    // List of string
    state.benchmarkNames = gatherBenchmarkNames(state.models);

    render(state, widgets);
  } catch (err) {
    console.error('Error loading leaderboard data:', err);
    widgets.container.textContent = 'Failed to load leaderboard.';
  }
})();
