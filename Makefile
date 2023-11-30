.PHONY: test-unit test-e2e

test-unit:
	@echo "Installing test deps..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	@echo "Running unit tests..."
	# Command to run unit tests goes here
	python -m pytest tests/ --junit-xml=tests/test_results/results.xml

test-e2e:
	# Command to run e2e tests goes here

images:
	scripts/build-container.sh

run:
	uvicorn app.main:app --reload --port 8080

verify:
	black . --check