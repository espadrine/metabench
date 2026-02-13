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

# List all benchmark names from AA Bench data
list-benchmarks:
	node bin/list-benchmarks.js

# Default target
.PHONY: test serve aabench lmarena list-benchmarks