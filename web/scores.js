// scores.js – fetches the predicted scores JSON, renders a sortable leaderboard,
// and provides UI to configure weighted metrics.

// ----- State -----
let state = {
  // List of { name: string, company: string, benchmarks: { bench: { score: number, stddev: number, source: string } } }
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
  editMetricCriteria: [],
  // string - current active tab ('chart' or 'table')
  currentTab: 'chart',
  // Chart.js instance
  chart: null
};

// ----- Utility functions -----

// Hardcoded colors for specific companies
function getCompanyColor(company) {
  const colorMap = {
    'xAI': 'darkpurple',
    'OpenAI': 'grey',
    'Anthropic': 'brown',
    'DeepSeek': 'navy',
    'Google': 'green',
    'Moonshot AI': 'darkred',
    'Z.ai': 'black',
    'Alibaba': 'darkgoldenrod',
    'Mistral AI': 'orange',
    'ServiceNow': 'pink',
    'LG': 'pink'
  };

  return colorMap[company] || stringToColor(company);
}

// Generate a consistent color from a string (company name) - fallback for unknown companies
function stringToColor(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }

  const hue = hash % 360;
  return `hsl(${hue}, 70%, 60%)`;
}

// ----- Tab management -----

function renderTabs(state, widgets) {
  // Update tab button states
  widgets.chartTab.classList.toggle('active', state.currentTab === 'chart');
  widgets.tableTab.classList.toggle('active', state.currentTab === 'table');

  // Update tab panel visibility
  widgets.chartContainer.classList.toggle('active', state.currentTab === 'chart');
  widgets.tableContainer.classList.toggle('active', state.currentTab === 'table');
}

function switchTab(state, widgets, tabName) {
  state.currentTab = tabName;
  renderTabs(state, widgets);

  if (tabName === 'chart') {
    renderChart(state, widgets);
  } else {
    renderTable(state, widgets);
  }
}

// ----- Storage integration -----

// Save current metrics to localStorage
function saveStateToStorage() {
  saveMetrics(state.metrics);
}

// Load metrics from localStorage and update state
function loadStateFromStorage() {
  const storedMetrics = loadMetrics();
  const storedMetricNames = new Set(storedMetrics.map(m => m.name));

  // Define benchmark groups (categories) and associated benchmark names.
  const categoryBenchmarks = {
    Knowledge: {
      'FRAMES': 1,
      "Humanity's Last Exam": 1,
      'MMLU-Pro': 1,
      'MMLU-Redux': 1,
      'MMMLU': 1,
      'MMMU': 1,
      'MMMU-Pro': 1,
      'SimpleQA': 1,
    },
    Reasoning: {
      'CharXiv reasoning (python tool)': 1,
      'Graphwalks bfs <128k>': 1,
      'Graphwalks parents <128k>': 1,
    },
    Math: {
      'AIME 2024': 1,
      'AIME 2025': 1,
      'CNMO 2024': 1,
      'HMMT 2025': 1,
      'FrontierMath (python tool)': 1,
    },
    Coding: {
      'LiveCodeBench': 10,
      'Codeforces': 10,
      'SWE-bench Verified': 10,
      'SWE-Lancer': 10,
      'Aider': 1,
      'Terminal-Bench': 1,
    },
    AgenticCoding: {
      'Aider': 10,
      'Terminal-Bench': 10,
      'LiveCodeBench': 5,
      'Codeforces': 5,
      'SWE-bench Verified': 5,
    },
    Agentic: {
      'BFCL_v3_MultiTurn': 1,
      'Internal API instruction following eval (hard)': 1,
      'Tau-Bench Airline': 1,
      'Tau-Bench Retail': 1,
      'Tau2-Bench Airline': 1,
      'Tau2-Bench Retail': 1,
      'Tau2-Bench Telecom': 1,
    },
    Factuality: {
      'FActScore hallucination rate': 1,
      'LongFact-Concepts hallucination rate': 1,
      'LongFact-Objects hallucination rate': 1,
    },
    Retrieval: {
      'BrowseComp Long Context 128k': 1,
      'BrowseComp Long Context 256k': 1,
      'OpenAI-MRCR: 2 needle 128k': 1,
      'OpenAI-MRCR: 2 needle 256k': 1,
    },
    Multimodal: {
      'VideoMME': 1,
      'VideoMMMU': 1,
      'MMMU': 1,
      'MMMU-Pro': 1,
    }
  };

  // Build default metrics: one per category with equal weight for each benchmark.
  const defaultMetrics = Object.entries(categoryBenchmarks)
    .filter(([category]) => !storedMetricNames.has(`${category}`))
    .map(([category, benches]) => ({
      name: `${category}`,
      criteria: Object.entries(benches).map(([bench, weight]) => ({ bench, weight }))
    }));

  // Combine default metrics with any stored user metrics.
  state.metrics = [...defaultMetrics, ...storedMetrics];
  // Select the first metric (the first default) as the current metric.
  state.currentMetricIndex = state.metrics.length > 0 ? 0 : null;
}

// ----- DOM elements -----
const widgets = {
  container: document.getElementById('leaderboard'),
  metricControls: document.getElementById('metric-controls'),
  newMetricModal: document.getElementById('new-metric-modal'),
  newMetricForm: document.getElementById('new-metric-form'),
  closeNewMetric: document.getElementById('close-new-metric'),
  editMetricModal: document.getElementById('edit-metric-modal'),
  editMetricForm: document.getElementById('edit-metric-form'),
  closeEditMetric: document.getElementById('close-edit-metric'),
  tabNavigation: document.getElementById('tab-navigation'),
  chartTab: document.getElementById('chart-tab'),
  tableTab: document.getElementById('table-tab'),
  chartContainer: document.getElementById('chart-container'),
  tableContainer: document.getElementById('table-container'),
  chartElement: document.getElementById('chart')
};

// ----- Rendering -----

function render(state, widgets) {
  renderMetricControls(state, widgets);
  renderTabs(state, widgets);

  if (state.currentTab === 'chart') {
    renderChart(state, widgets);
  } else {
    renderTable(state, widgets);
  }
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

// Chart visualization
function renderChart(state, widgets) {
  // Clear previous chart
  if (state.chart) {
    state.chart.destroy();
    state.chart = null;
  }

  widgets.chartElement.innerHTML = '';

  // Check if we have a metric selected
  if (state.currentMetricIndex === null || !state.metrics[state.currentMetricIndex]) {
    widgets.chartElement.innerHTML = '<p>Please select a metric to view the chart.</p>';
    return;
  }

  const metric = state.metrics[state.currentMetricIndex];

  // Check if Output cost benchmark exists
  if (!state.benchmarkNames.includes('Output cost')) {
    widgets.chartElement.innerHTML = '<p>Output cost benchmark not found in data.</p>';
    return;
  }

  // Prepare data for chart
  const chartData = [];
  const companies = new Set();

  state.models.forEach(model => {
    const metricScore = computeWeightedScore(model.benchmarks, metric.criteria);
    const outputCost = model.benchmarks['Output cost']?.score;

    if (typeof metricScore === 'number' && typeof outputCost === 'number') {
      companies.add(model.company);

      chartData.push({
        model: model.name,
        company: model.company,
        metricScore: metricScore,
        outputCost: outputCost
      });
    }
  });

  if (chartData.length === 0) {
    widgets.chartElement.innerHTML = '<p>No data available for chart.</p>';
    return;
  }

  // Create canvas for chart
  const canvas = document.createElement('canvas');
  canvas.id = 'chart-canvas';
  canvas.style.width = '100%';
  canvas.style.height = '500px';
  widgets.chartElement.appendChild(canvas);

  // Group data by company for coloring
  const companyColors = {};
  Array.from(companies).forEach(company => {
    companyColors[company] = getCompanyColor(company);
  });

  // Create chart
  const ctx = canvas.getContext('2d');
  state.chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: chartData.map(item => ({
        label: item.model,
        data: [{
          x: item.outputCost,
          y: item.metricScore,
          model: item.model,
          company: item.company
        }],
        backgroundColor: companyColors[item.company],
        borderColor: companyColors[item.company],
        borderWidth: 2,
        pointRadius: 8,
        pointHoverRadius: 12
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Output Cost ($/M tokens)'
          },
          type: 'logarithmic',
          position: 'bottom'
        },
        y: {
          title: {
            display: true,
            text: metric.name
          }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function(context) {
              const point = context.raw;
              return [
                `Model: ${point.model}`,
                `Company: ${point.company}`,
                `Output Cost: ${point.x.toFixed(2)}/M tokens`,
                `${metric.name}: ${point.y.toFixed(2)}`
              ];
            }
          }
        },
        legend: {
          display: true,
          position: 'right',
          labels: {
            generateLabels: function(chart) {
              const companies = {};
              chart.data.datasets.forEach(dataset => {
                const company = dataset.data[0].company;
                if (!companies[company]) {
                  companies[company] = {
                    text: company,
                    fillStyle: dataset.backgroundColor,
                    strokeStyle: dataset.borderColor,
                    lineWidth: 2,
                    hidden: false,
                    index: Object.keys(companies).length
                  };
                }
              });
              return Object.values(companies);
            }
          }
        }
      }
    }
  });
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

    // Save to storage
    saveStateToStorage();

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

    // Save to storage
    saveStateToStorage();

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

      // Save to storage
      saveStateToStorage();

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

// Returns a list of models { name, company, benchmarks: { bench: { score, stddev } } }.
async function fetchScores() {
  // The HTML file lives in web/, the JSON is now in the sibling data/
  // directory and has the structure:
  // { models: [ { name, company, benchmarks: [ { name, score, source, stdDev } ] } ] }
  const response = await fetch('../data/models-prediction.json');
  if (!response.ok) throw new Error(`Failed to load JSON: ${response.status}`);
  const data = await response.json();
  return data.models.map(model => ({
    name: model.name,
    company: model.company,
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

// Tab switching
widgets.chartTab.addEventListener('click', () => switchTab(state, widgets, 'chart'));
widgets.tableTab.addEventListener('click', () => switchTab(state, widgets, 'table'));

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
    if (!el && key !== 'sortContainer') {
      console.error(`Missing DOM element: ${key}`);
      return;
    }
  }

  try {
    // List of { name, benchmarks: { score: number, stddev: number } }
    state.models = await fetchScores();
    // List of string
    state.benchmarkNames = gatherBenchmarkNames(state.models);

    // Load metrics from localStorage
    loadStateFromStorage();

    render(state, widgets);
  } catch (err) {
    console.error('Error loading leaderboard data:', err);
    widgets.container.textContent = 'Failed to load leaderboard.';
  }
})();
