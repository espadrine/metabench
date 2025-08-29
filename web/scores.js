// scores.js – fetches the predicted scores JSON and renders a simple HTML table.
// Expected JSON format (data/scores-prediction.json):
// {
//   "Model A": { "Bench1": { "score": 42, "stddev": 1.2 }, ... },
//   ...
// }

async function fetchScores() {
  // The HTML file lives in web/, the JSON is in the sibling data/ folder.
  const response = await fetch('../data/scores-prediction.json');
  if (!response.ok) throw new Error(`Failed to load JSON: ${response.status}`);
  return await response.json();
}

// Determine the full set of benchmark names across all models.
// - scores: { model: { bench: { score: number, stddev: number } } }
// Returns a sorted array of benchmark names.
function gatherBenchmarkNames(scores) {
  const benchSet = new Set();
  Object.values(scores).forEach(model => {
    Object.keys(model).forEach(b => benchSet.add(b));
  });
  return Array.from(benchSet).sort();
}

function renderTable(state, widgets) {
  // Build the table.
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  const thModel = document.createElement('th');
  thModel.textContent = 'Model';
  headerRow.appendChild(thModel);
  state.benchmarkNames.forEach(b => {
    const th = document.createElement('th');
    th.textContent = b;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  Object.entries(state.scores).forEach(([modelName, modelData]) => {
    const row = document.createElement('tr');
    const tdModel = document.createElement('td');
    tdModel.textContent = modelName;
    row.appendChild(tdModel);
    state.benchmarkNames.forEach(b => {
      const td = document.createElement('td');
      const entry = modelData[b];
      if (entry) {
        const { score, stddev } = entry;
        // Show "±" only when stddev is a positive number.
        if (stddev && stddev > 0) {
          const twoSigma = 2 * stddev;
          td.textContent = `${Number(score.toFixed(2))}±${Number(twoSigma.toFixed(2))}`;
        } else {
          td.textContent = `${Number(score.toFixed(2))}`;
        }
      } else {
        td.textContent = '';
      }
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);

  // Replace placeholder text with the generated table.
  widgets.container.innerHTML = '';
  widgets.container.appendChild(table);
}

(async () => {
  const widgets = {};
  widgets.container = document.getElementById('leaderboard');
  if (!widgets.container) {
    console.error('Missing #leaderboard container');
    return;
  }

  try {
    const state = {};
    state.scores = await fetchScores();
    state.benchmarkNames = gatherBenchmarkNames(state.scores);
    renderTable(state, widgets);

  } catch (err) {
    console.error('Error loading leaderboard data:', err);
    container.textContent = 'Failed to load leaderboard.';
  }
})();

