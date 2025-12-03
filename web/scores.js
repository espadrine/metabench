// scores.js – fetches the predicted scores JSON, renders a sortable leaderboard,
// and provides UI to configure weighted metrics.

// ----- State -----
let state = {
  // List of { name: string, company: string, benchmarks: { bench: { score: number, stdDev: number, source: string } } }
  models: [],
  // List of string - all unique benchmark names across all models
  benchmarkNames: [],
  // Object mapping benchmark names to { mean: number, stdDev: number }
  benchmarkStats: {},
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
  chart: null,
  // string - current X-axis metric
  xAxisMetric: 'Cost of 1K responses'
};

// ----- Utility functions -----

// Hardcoded colors for specific companies
function getCompanyColor(company) {
  const colorMap = {
    'OpenAI': 'hsl(223, 75%, 22%)',
    'Anthropic': 'hsl(15, 52%, 58%)',
    'Google': 'hsl(136, 53%, 43%)',
    'Mistral AI': 'hsl(17, 96%, 52%)',
    'xAI': 'hsl(213, 4%, 57%)',
    'Alibaba': 'hsl(242, 80%, 65%)',
    'DeepSeek': 'hsl(230, 80%, 52%)',
    'Moonshot AI': 'hsl(212, 99%, 51%)',
    'MiniMax': 'hsl(343, 82%, 56%)',
    'Z.ai': 'hsl(274, 100%, 50%)',
    'Microsoft': 'hsl(199, 96%, 48%)',
  };

  return colorMap[company] || stringToColor(company);
}

// Get display label for X-axis metric
function getXAxisLabel(metricName) {
  const labelMap = {
    'Cost of 1K responses': 'Cost of 1K Responses ($)',
    'Active parameters': 'Billion Active Parameters',
    'Input cost': 'Input Cost ($/M tokens)',
    'Output cost': 'Output Cost ($/M tokens)',
    'Size': 'Size (Billion Parameters)',
    'Release date': 'Release Date (Year)',
    'ArtificialAnalysis Consumed Tokens (Millions)': 'ArtificialAnalysis Consumed Tokens (Millions)'
  };
  return labelMap[metricName] || metricName;
}

// Get scale type (linear/logarithmic) for X-axis metric
function getXAxisScaleType(metricName) {
  // Time-based metrics work best with linear scale
  if (metricName === 'Release date') {
    return 'linear';
  }
  // Default to logarithmic for most metrics (cost, size, etc.)
  return 'logarithmic';
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

// Function to build company colors object for chart
function buildCompanyColors(state, metric) {
  const companyColors = {};
  const companies = new Set();

  // Collect companies that have valid data for the current metric
  state.models.forEach(model => {
    const metricScore = computeWeightedScore(model.benchmarks, metric.criteria);
    const xAxisValue = model.benchmarks[state.xAxisMetric]?.score;

    if (typeof metricScore === 'number' && typeof xAxisValue === 'number') {
      companies.add(model.company);
    }
  });

  // Assign colors to companies
  Array.from(companies).forEach(company => {
    companyColors[company] = getCompanyColor(company);
  });

  return companyColors;
}

// Function to build chart datasets from state and metric
function buildChartDatasets(state, metric, companyColors) {
  const datasets = [];

  state.models.forEach(model => {
    const metricScore = computeWeightedScore(model.benchmarks, metric.criteria);
    const xAxisValue = model.benchmarks[state.xAxisMetric]?.score;

    if (typeof metricScore === 'number' && typeof xAxisValue === 'number') {
      let backgroundColor, borderColor;
      
      if (Object.keys(companyColors).length > 0) {
        // If company colors are provided, use them with transparency
        const baseColor = companyColors[model.company];
        const transparency = calculateTransparency(model.release_date);

        // Convert color to RGBA with transparency
        if (baseColor.startsWith('hsl')) {
          // Convert HSL to RGBA
          const hslMatch = baseColor.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
          if (hslMatch) {
            const h = parseInt(hslMatch[1]);
            const s = parseInt(hslMatch[2]);
            const l = parseInt(hslMatch[3]);
            backgroundColor = `hsla(${h}, ${s}%, ${l}%, ${transparency})`;
            borderColor = `hsla(${h}, ${s}%, ${l}%, ${Math.min(1.0, transparency + 0.2)})`; // Slightly more opaque border
          } else {
            backgroundColor = baseColor;
            borderColor = baseColor;
          }
        } else {
          // For other color formats, use a simpler approach
          backgroundColor = baseColor;
          borderColor = baseColor;
        }
      } else {
        // If no company colors provided, use default colors
        backgroundColor = 'rgba(54, 162, 235, 0.7)';
        borderColor = 'rgba(54, 162, 235, 0.9)';
      }

      datasets.push({
        label: model.name,
        data: [{
          x: xAxisValue,
          y: metricScore,
          model: model.name,
          company: model.company,
          releaseDate: model.release_date,
          outputCost: model.benchmarks['Output cost']?.score
        }],
        backgroundColor: backgroundColor,
        borderColor: borderColor,
        borderWidth: 2,
        pointRadius: 8,
        pointHoverRadius: 12
      });
    }
  });

  return datasets;
}

// Function to generate legend labels with company colors and ordered by best score
function generateLegendLabels(state, companyColors, metric) {
  const companies = {};
  const companyBestScores = {};
  
  // First pass: collect companies and compute their best scores
  state.models.forEach(model => {
    const company = model.company;
    const metricScore = computeWeightedScore(model.benchmarks, metric.criteria);
    
    if (typeof metricScore === 'number') {
      if (!companies[company]) {
        companies[company] = {
          text: company,
          fillStyle: companyColors[company],
          strokeStyle: companyColors[company],
          lineWidth: 2,
          hidden: false,
          index: Object.keys(companies).length
        };
        companyBestScores[company] = metricScore;
      } else {
        // Update best score if current model has a higher score
        companyBestScores[company] = Math.max(companyBestScores[company], metricScore);
      }
    }
  });
  
  // Add best scores to company objects
  Object.keys(companies).forEach(company => {
    companies[company].bestScore = companyBestScores[company];
  });
  
  // Sort companies by best score (descending)
  const sortedCompanies = Object.values(companies).sort((a, b) => b.bestScore - a.bestScore);
  
  // Update indices after sorting
  sortedCompanies.forEach((company, index) => {
    company.index = index;
  });
  
  return sortedCompanies;
}

// Calculate transparency based on release date (older = more transparent)
function calculateTransparency(releaseDate) {
  if (!releaseDate) return 0.7; // Default transparency for models without dates

  const release = new Date(releaseDate);
  const now = new Date();
  const ageInMonths = (now - release) / (1000 * 60 * 60 * 24 * 30.44); // Approximate months
  const ageInYears = (now - release) / (1000 * 60 * 60 * 24 * 365);

  // Older models get more transparent (lower alpha)
  // Models older than 24 months will be very transparent (0.2)
  // New models (0 months) will be fully opaque (1.0)
  const maxAgeMonths = 24;
  //const transparency = Math.max(0.2, 1.0 - (ageInMonths / maxAgeMonths));
  const transparency = Math.max(0.0, 1.0 - ageInYears);
  return transparency;

  return Math.min(1.0, Math.max(0.2, transparency)); // Clamp between 0.2 and 1.0
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

// Define benchmark groups (categories) and associated benchmark names.
const categoryBenchmarks = {
  Knowledge: {
    'ArtificialAnalysis Intelligence Index': 10,
    "Humanity's Last Exam": 10,
    'SimpleQA': 10,
    'MMLU-Pro': 10,
    'MMLU-Redux': 10,
    'MMLU': 10,
    'MMMLU': 1,
    'MMMU': 1,
    'MMMU-Pro': 1,
    'Vibe-Eval': 1,
  },
  Reasoning: {
    'ARC AGI 2': 1,
    'CharXiv reasoning (with tools)': 1,
    'Graphwalks bfs <128k': 1,
    'Graphwalks parents <128k': 1,
    'COLLIE': 1,
    'DROP': 1,
    'ERQA': 1,
    'FinSearchComp-T3': 1,
    'FinSearchComp-global': 1,
    'IFBench': 1,
    'LCR': 1,
    'LOFT (128k)': 1,
  },
  Search: {
    'SimpleQA (with tools)': 1,
    'Reka Research Eval (with tools)': 1,
    'xbench-DeepSearch': 1,
    'Seal-0': 1,
    "Humanity's Last Exam (with tools)": 1,
    'FRAMES': 1,
    'FACTS Grounding': 1,
    'FActScore hallucination rate': -1,
    'BrowseComp Long Context 128k': 1,
    'BrowseComp Long Context 256k': 1,
    'MRCR: 2 needle 128k': 1,
    'MRCR: 2 needle 256k': 1,
    'LongFact-Concepts hallucination rate': -1,
    'LongFact-Objects hallucination rate': -1,
    'FinSearchComp-T3': 1,
    'FinSearchComp-global': 1,
  },
  Math: {
    'IMO-AnswerBench': 1,
    'USAMO 2025': 1,
    'CNMO 2024': 1,
    'AIME 2024': 1,
    'AIME 2025': 1,
    'AIME 2025 (with tools)': 1,
    'HMMT Feb 2025': 1,
    'HMMT Nov 2025': 1,
    'GPQA Diamond': 1,
    'FrontierMath (with tools)': 1,
    'MATH': 1,
    'MGSM': 1,
  },
  Coding: {
    'LiveCodeBench': 10,
    'Codeforces': 10,
    'SWE-bench Verified': 10,
    'SWE-Lancer': 10,
    'ArtifactsBench': 1,
    'BIRD-SQL': 1,
    'HumanEval': 1,
    // Only has one entry:
    //'Natural2Code': 1,
    'OJ-Bench': 1,
    'SciCode': 1,
  },
  'Agentic Coding': {
    'Terminal-Bench': 10,
    'Terminal-Bench-Hard': 10,
    'Terminal-Bench 2.0': 10,
    'Aider': 10,
    'SWE-bench Verified (with tools)': 1,
    // FIXME: need to fix prediction on Codeforces + tool.
    //'Codeforces (with tools)': 1,
  },
  Agentic: {
    'τ-Bench Airline': 1,
    'τ-Bench Retail': 1,
    'τ²-Bench Airline': 1,
    'τ²-Bench Retail': 1,
    'τ²-Bench Telecom': 1,
    'BrowseComp': 1,
    'BrowseComp (with tools)': 1,
    'GAIA (text)': 1,
    'AgentCompany': 1,
    'Finance Agent': 1,
    'BFCL v3': 1,
    'OSWorld': 1,
  },
  Multimodal: {
    'MMMU': 1,
    'MMMU-Pro': 1,
    'ArtifactsBench': 1,
    'VideoMMMU': 1,
    'Video-MME': 1,
    'EgoSchema': 1,
    'OSWorld': 1,
    'Vibe-Eval': 1,
  }
};

// ----- Storage integration -----

// Save current metrics to localStorage
function saveStateToStorage() {
  saveMetrics(state.metrics);
}

// Load metrics from localStorage and update state
function loadStateFromStorage() {
  const storedMetrics = loadMetrics();

  // Build default metrics: one per category with equal weight for each benchmark.
  // Always create all default metrics, regardless of what's stored.
  const defaultMetrics = Object.entries(categoryBenchmarks)
    .map(([category, benches]) => ({
      name: `${category}`,
      criteria: Object.entries(benches).map(([bench, weight]) => ({ bench, weight }))
    }));

  // Combine default metrics with any stored user metrics.
  // Default metrics always come first, followed by user metrics.
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
  xAxisSelect: document.getElementById('x-axis-select'),
  chartElement: document.getElementById('chart')
};

// Add event listener for X-axis dropdown
if (widgets.xAxisSelect) {
  widgets.xAxisSelect.addEventListener('change', (e) => {
    state.xAxisMetric = e.target.value;
    render(state, widgets);
  });
}

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
          const { score, stdDev, source } = entry;
          const fmtScore = Number(score.toFixed(2));
          if (stdDev && stdDev > 0) {
            const ci = confidenceInterval(stdDev, 0.99);
            td.textContent = `${fmtScore}±${Number(ci.toFixed(2))}`;
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

  // Check if selected X-axis benchmark exists
  if (!state.benchmarkNames.includes(state.xAxisMetric)) {
    widgets.chartElement.innerHTML = `<p>${state.xAxisMetric} benchmark not found in data.</p>`;
    return;
  }

  // Group data by company for coloring
  // Build company colors for chart
  const companyColors = buildCompanyColors(state, metric);

  // Build chart datasets with company colors
  const datasets = buildChartDatasets(state, metric, companyColors);

  if (datasets.length === 0) {
    widgets.chartElement.innerHTML = '<p>No data available for chart.</p>';
    return;
  }

  // Create canvas for chart
  const canvas = document.createElement('canvas');
  canvas.id = 'chart-canvas';
  canvas.style.width = '100%';
  canvas.style.height = '500px';
  widgets.chartElement.appendChild(canvas);

  // Create chart
  const ctx = canvas.getContext('2d');
  state.chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: getXAxisLabel(state.xAxisMetric)
          },
          type: getXAxisScaleType(state.xAxisMetric),
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
              const xAxisLabel = getXAxisLabel(state.xAxisMetric);
              const tooltipLines = [
                `Model: ${point.model}`,
                `Company: ${point.company}`,
                `${xAxisLabel}: ${point.x.toFixed(2)}`,
                `Output Cost: $${point.outputCost?.toFixed(2) || 'N/A'}/M tokens`,
                `${metric.name}: ${point.y.toFixed(2)}`
              ];

              if (point.releaseDate) {
                const releaseDate = new Date(point.releaseDate);
                const now = new Date();
                const ageInMonths = Math.round((now - releaseDate) / (1000 * 60 * 60 * 24 * 30.44));
                tooltipLines.push(`Released: ${releaseDate.toLocaleDateString()} (${ageInMonths} months ago)`);
              }

              return tooltipLines;
            }
          }
        },
        legend: {
          display: true,
          position: 'right',
          labels: {
            generateLabels: function(chart) {
              return generateLegendLabels(state, companyColors, metric);
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

// Returns the raw models data including release_date and other fields.
async function fetchScores() {
  // The HTML file lives in web/, the JSON is now in the sibling data/
  // directory and has the structure:
  // { models: [ { name, company, url, release_date, benchmarks: [ { name, score, source, stdDev } ] } ] }
  const response = await fetch('./models-prediction.json');
  if (!response.ok) throw new Error(`Failed to load JSON: ${response.status}`);
  const data = await response.json();

  // Return the raw models data, but convert benchmarks array to object format
  return data.models.map(model => ({
    name: model.name,
    company: model.company,
    url: model.url,
    release_date: model.release_date,
    benchmarks: model.benchmarks.reduce((acc, b) => {
      acc[b.name] = {
        score: b.score,
        stdDev: b.stdDev,
        source: b.source,
      };
      return acc;
    }, {}),
  }));
}

// Determine the full set of benchmark names across all models.
// - models: list of { name: model name, benchmarks: { bench: { score: number, stdDev: number } } }
// Returns a sorted array of benchmark names.
function gatherBenchmarkNames(models) {
  const benchSet = new Set();
  for (const model of models) {
    Object.keys(model.benchmarks).forEach(b => benchSet.add(b));
  }
  return Array.from(benchSet).sort();
}

// Compute mean and standard deviation for each benchmark across all models
function computeBenchmarkStats(models, benchmarkNames) {
  const stats = {};

  benchmarkNames.forEach(benchmarkName => {
    const scores = [];

    // Collect all scores for this benchmark
    models.forEach(model => {
      const entry = model.benchmarks[benchmarkName];
      if (entry && typeof entry.score === 'number') {
        scores.push(entry.score);
      }
    });

    if (scores.length > 0) {
      // Compute mean
      const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;

      // Compute standard deviation
      const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
      const stdDev = Math.sqrt(variance);

      stats[benchmarkName] = { mean, stdDev };
    } else {
      // If no scores available, use default values
      stats[benchmarkName] = { mean: 0, stdDev: 1 };
    }
  });

  return stats;
}

// Compute weighted average score for a single model based on current criteria.
// Uses normalized scores (z-scores) to account for different benchmark ranges.
// - modelData: { bench: { score: number, stdDev: number } }
// - sortingCriteria: array of { bench: string, weight: number }
function computeWeightedScore(modelData, sortingCriteria) {
  if (sortingCriteria.length === 0) return 0;
  let sum = 0;
  let weightSum = 0;
  sortingCriteria.forEach(({ bench, weight }) => {
    const entry = modelData[bench];
    if (entry && typeof entry.score === 'number') {
      const stats = state.benchmarkStats[bench];
      if (stats && stats.stdDev > 0) {
        // Normalize the score: (score - mean) / stdDev
        const normalizedScore = (entry.score - stats.mean) / stats.stdDev;
        sum += weight * normalizedScore;
        weightSum += weight;
      } else {
        // Fallback to original score if no stats available
        sum += weight * entry.score;
        weightSum += weight;
      }
    }
  });
  return weightSum === 0 ? 0 : sum / weightSum;
}

function confidenceInterval(stdDev, confidenceLevel) {
  // For normal distributions, the CI is value ± σ × √2×erf^-1(ρ)
  return stdDev * Math.sqrt(2) * inverseErf(confidenceLevel);
}

function inverseErf(x) {
  const a = 0.147; // Approximation constant
  const ln = Math.log(1 - x * x);
  const b = (2 / (Math.PI * a)) + (ln / 2);
  return Math.sign(x) * Math.sqrt(Math.sqrt(b * b - ln / a) - b);
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
    // List of { name, benchmarks: { score: number, stdDev: number } }
    state.models = await fetchScores();
    // List of string
    state.benchmarkNames = gatherBenchmarkNames(state.models);
    // Compute benchmark statistics for normalization
    state.benchmarkStats = computeBenchmarkStats(state.models, state.benchmarkNames);

    // Load metrics from localStorage
    loadStateFromStorage();

    render(state, widgets);
  } catch (err) {
    console.error('Error loading leaderboard data:', err);
    widgets.container.textContent = 'Failed to load leaderboard.';
  }
})();
