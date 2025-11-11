# Makefile for LLM Benchmark Aggregator

# Default target
.PHONY: test serve

# Run all tests
test:
	node --test

# Development server
serve:
	cd web && python3 -m http.server 8901