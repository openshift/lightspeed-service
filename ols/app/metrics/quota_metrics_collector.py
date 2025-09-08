"""Prometheus metrics collector for quota utilization statistics."""

import logging
from typing import Dict, Set

from prometheus_client import Gauge

from ols.app.metrics.quota_metrics_repository import QuotaMetricsRepository

logger = logging.getLogger(__name__)


class QuotaMetricsCollector:
    """Collector for quota-related Prometheus metrics."""
    
    def __init__(self, repository: QuotaMetricsRepository) -> None:
        """Initialize the quota metrics collector."""
        self.repository = repository
        
        # Initialize Prometheus metrics
        self.quota_limit_total = Gauge(
            "ols_quota_limit_total",
            "Total quota limit allocated",
            ["subject_type", "subject_id"]
        )
        
        self.quota_available_total = Gauge(
            "ols_quota_available_total",
            "Available quota remaining", 
            ["subject_type", "subject_id"]
        )
        
        self.quota_utilization_percent = Gauge(
            "ols_quota_utilization_percent",
            "Quota utilization as percentage",
            ["subject_type", "subject_id"]
        )
        
        self.token_usage_total = Gauge(
            "ols_token_usage_total",
            "Total tokens consumed",
            ["user_id", "provider", "model", "token_type"]
        )
        
        self.quota_warning_subjects_total = Gauge(
            "ols_quota_warning_subjects_total",
            "Number of subjects with >80% quota usage",
            ["subject_type"]
        )
        
        self.quota_exceeded_subjects_total = Gauge(
            "ols_quota_exceeded_subjects_total",
            "Number of subjects that exceeded quota",
            ["subject_type"]
        )
        
        logger.info("QuotaMetricsCollector initialized")
    
    def update_quota_metrics(self) -> None:
        """Update quota-related Prometheus metrics."""
        try:
            # Check database health first
            if not self.repository.health_check():
                logger.warning("Database health check failed, skipping quota metrics update")
                return
            
            logger.debug("Updating quota metrics")
            quota_records = self.repository.get_quota_records()
            
            # Track seen metrics to clear stale ones
            seen_quota_metrics: Set[tuple] = set()
            
            # Counters for warning and exceeded subjects
            warning_counts: Dict[str, int] = {}
            exceeded_counts: Dict[str, int] = {}
            
            for record in quota_records:
                subject_type = "user" if record.subject == "u" else "cluster"
                subject_id = record.id if record.id else "cluster"
                
                labels = (subject_type, subject_id)
                seen_quota_metrics.add(labels)
                
                # Update basic quota metrics
                self.quota_limit_total.labels(*labels).set(record.quota_limit)
                self.quota_available_total.labels(*labels).set(record.available)
                self.quota_utilization_percent.labels(*labels).set(record.utilization_percent)
                
                # Track warning and exceeded thresholds
                if record.utilization_percent > 100:
                    exceeded_counts[subject_type] = exceeded_counts.get(subject_type, 0) + 1
                elif record.utilization_percent > 80:
                    warning_counts[subject_type] = warning_counts.get(subject_type, 0) + 1
            
            # Update threshold metrics
            for subject_type in ["user", "cluster"]:
                self.quota_warning_subjects_total.labels(subject_type).set(
                    warning_counts.get(subject_type, 0)
                )
                self.quota_exceeded_subjects_total.labels(subject_type).set(
                    exceeded_counts.get(subject_type, 0)
                )
            
            logger.debug("Updated %d quota records", len(quota_records))
            
        except Exception as e:
            logger.error("Error updating quota metrics: %s", e)
    
    def update_token_usage_metrics(self) -> None:
        """Update token usage Prometheus metrics."""
        try:
            # Check database health first
            if not self.repository.health_check():
                logger.warning("Database health check failed, skipping token usage metrics update")
                return
            
            logger.debug("Updating token usage metrics")
            token_records = self.repository.get_token_usage_records()
            
            # Track seen metrics to clear stale ones
            seen_token_metrics: Set[tuple] = set()
            
            for record in token_records:
                # Update input token metrics
                input_labels = (record.user_id, record.provider, record.model, "input")
                seen_token_metrics.add(input_labels)
                self.token_usage_total.labels(*input_labels).set(record.input_tokens)
                
                # Update output token metrics
                output_labels = (record.user_id, record.provider, record.model, "output")
                seen_token_metrics.add(output_labels)
                self.token_usage_total.labels(*output_labels).set(record.output_tokens)
            
            logger.debug("Updated %d token usage records", len(token_records))
            
        except Exception as e:
            logger.error("Error updating token usage metrics: %s", e)
    
    def update_all_metrics(self) -> None:
        """Update all quota-related metrics."""
        logger.debug("Starting comprehensive quota metrics update")
        
        try:
            self.update_quota_metrics()
        except Exception as e:
            logger.error("Failed to update quota metrics: %s", e)
        
        try:
            self.update_token_usage_metrics()
        except Exception as e:
            logger.error("Failed to update token usage metrics: %s", e)
        
        logger.debug("Completed quota metrics update")