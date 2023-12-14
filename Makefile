# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: test test-unit test-e2e images run format verify

ARTIFACT_DIR := $(if $(ARTIFACT_DIR),$(ARTIFACT_DIR),tests/test_results)


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
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	python -m pytest tests/unit --cov=app --cov=src --cov=utils --cov-report term-missing --cov-report json:${ARTIFACT_DIR}/coverage_unit.json --junit-xml=${ARTIFACT_DIR}/junit_unit.xml
	python scripts/transform_coverage_report.py ${ARTIFACT_DIR}/coverage_unit.json ${ARTIFACT_DIR}/coverage_unit.out
	scripts/codecov.sh ${ARTIFACT_DIR}/coverage_unit.out

test-integration:
	@echo "Running unit tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	python -m pytest tests/integration --cov=app --cov=src --cov=utils --cov-report term-missing --cov-report json:${ARTIFACT_DIR}/coverage_integration.json --junit-xml=${ARTIFACT_DIR}/junit_integration.xml
	python scripts/transform_coverage_report.py ${ARTIFACT_DIR}/coverage_integration.json ${ARTIFACT_DIR}/coverage_integration.out
	scripts/codecov.sh ${ARTIFACT_DIR}/coverage_integration.out

test-e2e:
	# Command to run e2e tests goes here

format:
	black .

verify:
	black . --check

