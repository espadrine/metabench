data/models.json: data/models/*.json bin/*.js lib/*.js
	node bin/leaderboard.js

# Download and create data/aabench.json and data/missing_aabench_benchmarks.json
aabench:
	node bin/load_aabench.js

# Download and create data/lmarena.json and data/missing_lmarena_benchmarks.json
lmarena:
	node bin/load_lmarena.js

test:
	node --test

# UI tests (headless-browser tests for web/scores.js)
test-ui:
	node --test test/ui/chart-legend.test.js

# Development server
serve:
	cd web && python3 -m http.server 8901

# List all benchmark names from AA Bench data
list-benchmarks:
	node bin/list-benchmarks.js

# Default target
.PHONY: test test-ui serve aabench lmarena list-benchmarks
