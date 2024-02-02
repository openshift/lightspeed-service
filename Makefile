# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: test test-unit test-e2e images run format verify

ARTIFACT_DIR := $(if $(ARTIFACT_DIR),$(ARTIFACT_DIR),tests/test_results)


images: ## Build container images
	scripts/build-container.sh

install-tools: ## Install required utilities/tools
	@command -v pdm > /dev/null || { echo >&2 "pdm is not installed. Installing..."; pip install pdm; }

install-deps: install-tools ## Install all required dependencies needed to run the service
	pdm install

install-deps-test: install-tools ## Install all required dependencies needed to test the service
	pdm install --dev

run: ## Run the service locally
	uvicorn ols.app.main:app --reload --port 8080

test: test-unit test-integration test-e2e ## Run all tests

test-unit: ## Run the unit tests
	@echo "Running unit tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	python -m pytest tests/unit --cov=ols --cov-report term-missing --cov-report json:${ARTIFACT_DIR}/coverage_unit.json --junit-xml=${ARTIFACT_DIR}/junit_unit.xml
	python scripts/transform_coverage_report.py ${ARTIFACT_DIR}/coverage_unit.json ${ARTIFACT_DIR}/coverage_unit.out
	scripts/codecov.sh ${ARTIFACT_DIR}/coverage_unit.out

test-integration: ## Run integration tests tests
	@echo "Running integration tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	python -m pytest tests/integration --cov=ols --cov-report term-missing --cov-report json:${ARTIFACT_DIR}/coverage_integration.json --junit-xml=${ARTIFACT_DIR}/junit_integration.xml --cov-fail-under=60
	python scripts/transform_coverage_report.py ${ARTIFACT_DIR}/coverage_integration.json ${ARTIFACT_DIR}/coverage_integration.out
	scripts/codecov.sh ${ARTIFACT_DIR}/coverage_integration.out

test-e2e: ## Run e2e tests - requires running OLS server
	@echo "Running e2e tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	python -m pytest tests/e2e --junit-xml=${ARTIFACT_DIR}/junit_e2e.xml
		

coverage-report:	test-unit ## Export unit test coverage report into interactive HTML
	coverage html

check-types: ## Checks type hints in sources
	mypy ols/

format: ## Format the code into unified format
	black .
	ruff . --fix --per-file-ignores=tests/*:S101

verify: ## Verify the code using various linters
	black . --check
	ruff . --per-file-ignores=tests/*:S101

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''

