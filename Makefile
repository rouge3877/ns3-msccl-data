.PHONY: env

export allspark_output := $(shell pwd)/output
export allspark_input := $(shell pwd)/input

env:
	@echo "Environment variables set:"
	@echo "  allspark_output=$(allspark_output)"
	@echo "  allspark_input=$(allspark_input)"
