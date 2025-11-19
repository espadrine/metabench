data/models-prediction.json: data/models.json bin/leaderboard.js
	node bin/leaderboard.js

# Download and create data/aabench.json and data/missing_aabench_benchmarks.json
aabench:
	node bin/load_aabench.js

test:
	node --test

# Development server
serve:
	cd web && python3 -m http.server 8901

# Default target
.PHONY: test serve aabench
