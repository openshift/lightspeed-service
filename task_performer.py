import logging
import sys
from string import Template
from model_context import get_watsonx_predictor


class TaskPerformer:
    def perform_task(self, conversation, model, task, original_query):

        return """
apiVersion: "autoscaling.openshift.io/v1"
kind: "ClusterAutoscaler"
metadata:
  name: "default"
spec:
  resourceLimits:
    maxNodesTotal: 10
  scaleDown: 
    enabled: true 
"""