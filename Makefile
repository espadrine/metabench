# Download and create data/aabench.json and data/missing_aabench_benchmarks.json
aabench:
	node bin/load_aabench.js

# Download and create data/lmarena.json and data/missing_lmarena_benchmarks.json
lmarena:
	node bin/load_lmarena.js

test:
	node --test

# Development server
serve:
	cd web && python3 -m http.server 8901

# Default target
.PHONY: test serve aabench lmarena