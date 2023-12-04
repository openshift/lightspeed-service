# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: test test-unit test-e2e images run format verify

images:
	scripts/build-container.sh

install-deps: 
	pip install -r requirements.txt

install-deps-test:
	pip install -r requirements-test.txt

run:
	uvicorn app.main:app --reload --port 8080

test: test-unit test-integration test-e2e

test-unit:
	@echo "Running unit tests..."
	# Command to run unit tests goes here
	python -m pytest tests/unit --cov=app --cov=src --cov=utils --cov-report term-missing --cov-report xml:tests/test_results/unit/coverage.xml --junit-xml=tests/test_results/unit/results.xml

test-integration:
	@echo "Running unit tests..."
	# Command to run unit tests goes here
	python -m pytest tests/integration --cov=app --cov=src --cov=utils --cov-report term-missing --cov-report xml:tests/test_results/integration/coverage.xml --junit-xml=tests/test_results/unit/results.xml

test-e2e:
	# Command to run e2e tests goes here

format:
	black .

verify:
	black . --check

