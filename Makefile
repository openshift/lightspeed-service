.PHONY: test-unit test-e2e

test-unit:
	@echo "Installing test deps..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	@echo "Running unit tests..."
	pytest test_*.py

test-e2e:
	@echo "Installing test deps..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	@echo "Running end-to-end tests..."
	# Command to run e2e tests goes here

images:
	scripts/build-container.sh

run:
	uvicorn app.main:app --reload --port 8080

verify:
	black . --check

