# Usage:
#   make all                        # Run all evals
#   make envvar_missing             # Run a single eval by tag

.PHONY: all envvar_missing batch_failure storage_binding \
        namespace_pod_count scheduled_outage_detection periodic_failure_window \
        config_drift_analysis readiness_probe_diagnosis \
        ingress_rule_mismatch oom wrong_networkpolicy

EVAL_CMD = API_KEY=$$(oc whoami -t) lightspeed-eval \
           --system-config system.yaml \
           --eval-data evals.yaml \
           --output-dir results

all: envvar_missing batch_failure storage_binding \
     namespace_pod_count scheduled_outage_detection periodic_failure_window \
     config_drift_analysis readiness_probe_diagnosis \
     ingress_rule_mismatch oom wrong_networkpolicy

envvar_missing:
	$(EVAL_CMD) --tag envvar_missing

batch_failure:
	$(EVAL_CMD) --tag batch_failure

storage_binding:
	$(EVAL_CMD) --tag storage_binding

namespace_pod_count:
	$(EVAL_CMD) --tag namespace_pod_count

scheduled_outage_detection:
	$(EVAL_CMD) --tag scheduled_outage_detection

periodic_failure_window:
	$(EVAL_CMD) --tag periodic_failure_window

config_drift_analysis:
	$(EVAL_CMD) --tag config_drift_analysis

readiness_probe_diagnosis:
	$(EVAL_CMD) --tag readiness_probe_diagnosis

ingress_rule_mismatch:
	$(EVAL_CMD) --tag ingress_rule_mismatch

oom:
	$(EVAL_CMD) --tag oom

wrong_networkpolicy:
	$(EVAL_CMD) --tag wrong_networkpolicy
