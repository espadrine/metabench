data/models-prediction.json: data/models.json bin/leaderboard.js
	node bin/leaderboard.js

# Download and create data/aabench.json and data/missing_aabench_benchmarks.json
aabench:
	node bin/load_aabench.js

# Download and create data/lmarena.json and data/missing_lmarena_benchmarks.json
lmarena:
	if [ ! -d arena-catalog ]; then \
		git clone git@github.com:lmarena/arena-catalog.git; \
	fi
	docker run --rm \
		-v $(PWD)/arena-catalog:/workspace \
		-v $(PWD)/data:/output \
		-w /workspace \
		python:3.11-slim \
		bash -c "\
			python3 -m venv venv && \
			. venv/bin/activate && \
			pip3 install -r requirements.txt && \
			python update_leaderboard_data.py && \
			cp -f data/leaderboard-text.json /output/lmarena.json"
	node bin/load_lmarena.js

test:
	node --test

# Development server
serve:
	cd web && python3 -m http.server 8901

# Default target
.PHONY: test serve aabench
