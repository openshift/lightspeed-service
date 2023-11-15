.PHONY: unit e2e-test

test-unit:
	@echo "Running unit tests..."
	# Command to run unit tests goes here

test-e2e:
	@echo "Running end-to-end tests..."
	# Command to run end-to-end tests goes here

images:
	scripts/build-container.sh

run:
	uvicorn ols:app --reload --port 8080
