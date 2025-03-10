from shared.utils import LLMInterface, setup_logger
from shared.models import Alert, AlertSeverity
from typing import Dict, List, Any, Optional
import json
import datetime
import os
import time
import threading
from collections import defaultdict


class MonitoringAgent:
    """Agent that monitors system health and model performance"""

    def __init__(self):
        self.logger = setup_logger("MonitoringAgent")
        self.metrics = defaultdict(list)
        self.alerts: List[Alert] = []

        # Initialize LLM interface
        self.llm = LLMInterface(
            system_role="You are a monitoring agent for a federated learning system. "
                        "You analyze metrics and detect anomalies, performance issues, or security concerns."
        )

        self.data_dir = os.environ.get("DATA_DIR", "/data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing data if available
        self._load_data()

        # Start background thread for periodic checks
        self.should_stop = False
        self.check_thread = threading.Thread(target=self._periodic_check)
        self.check_thread.daemon = True
        self.check_thread.start()

        self.logger.info("MonitoringAgent initialized")

    def _load_data(self):
        """Load existing monitoring data if available"""
        metrics_file = os.path.join(self.data_dir, "metrics.json")
        alerts_file = os.path.join(self.data_dir, "alerts.json")

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = defaultdict(list, metrics_data)
                self.logger.info(f"Loaded metrics for {len(self.metrics)} sources")
            except Exception as e:
                self.logger.error(f"Error loading metrics data: {e}")

        if os.path.exists(alerts_file):
            try:
                with open(alerts_file, 'r') as f:
                    alerts_data = json.load(f)
                    self.alerts = [Alert(**alert) for alert in alerts_data]
                self.logger.info(f"Loaded {len(self.alerts)} existing alerts")
            except Exception as e:
                self.logger.error(f"Error loading alerts data: {e}")

    def _save_data(self):
        """Save current monitoring data"""
        metrics_file = os.path.join(self.data_dir, "metrics.json")
        alerts_file = os.path.join(self.data_dir, "alerts.json")

        with open(metrics_file, 'w') as f:
            # Limit the number of metrics we store per source
            limited_metrics = {k: v[-100:] for k, v in self.metrics.items()}
            json.dump(limited_metrics, f, default=str)

        with open(alerts_file, 'w') as f:
            alerts_list = [alert.dict() for alert in self.alerts]
            json.dump(alerts_list, f, default=str)

    def record_metrics(self, source: str, metrics: Dict[str, Any]):
        """Record metrics from a source"""
        timestamp = datetime.datetime.now().isoformat()

        metric_record = {
            "timestamp": timestamp,
            "data": metrics
        }

        self.metrics[source].append(metric_record)

        # Keep only the most recent metrics (last 1000)
        if len(self.metrics[source]) > 1000:
            self.metrics[source] = self.metrics[source][-1000:]

        # Check for anomalies or issues
        self.analyze_metrics(source, metrics)

        # Save updated data
        self._save_data()

        self.logger.info(f"Recorded metrics from {source}")
        return True

    def analyze_metrics(self, source: str, metrics: Dict[str, Any]):
        """Analyze metrics to detect anomalies"""
        # Only analyze if we have enough history
        if len(self.metrics[source]) > 3:
            # Get previous metrics for comparison
            previous_metrics = [m["data"] for m in self.metrics[source][-4:-1]]

            output_structure = {
                "issue_detected": False,
                "severity": "medium",
                "description": "Description of the issue",
                "affected_components": ["list", "of", "affected", "components"]
            }

            prompt = f"""
            Analyze these metrics from {source} and identify any anomalies or issues:
            
            Current metrics:
            {json.dumps(metrics, indent=2)}
            
            Previous metrics (3 most recent):
            {json.dumps(previous_metrics, indent=2)}
            
            Consider issues like:
            1. Performance degradation
            2. Unusual spikes or drops in values
            3. Model drift
            4. Client participation issues
            5. Resource usage concerns
            
            Identify any anomalies, performance degradation, or concerning patterns.
            """

            analysis = self.llm.structured_query(prompt, output_structure)

            if analysis.get("issue_detected", False):
                self.create_alert(
                    source=source,
                    severity=AlertSeverity(analysis.get("severity", "medium")),
                    description=analysis.get("description", "Anomaly detected")
                )

    def create_alert(self, source: str, severity: AlertSeverity, description: str) -> Alert:
        """Create a new alert"""
        alert = Alert(
            source=source,
            severity=severity,
            description=description
        )

        self.alerts.append(alert)
        self.logger.info(f"ALERT [{severity.upper()}]: {description} (source: {source})")

        self._save_data()
        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.datetime.now()

                self.logger.info(f"Resolved alert {alert_id}")
                self._save_data()
                return True

        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        return [alert for alert in self.alerts if not alert.resolved]

    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts, including resolved ones"""
        return self.alerts

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of current system metrics"""
        summary = {
            "sources": len(self.metrics),
            "total_records": sum(len(records) for records in self.metrics.values()),
            "active_alerts": sum(1 for alert in self.alerts if not alert.resolved),
            "last_updated": datetime.datetime.now().isoformat()
        }

        # Add the latest value for each source
        latest = {}
        for source, records in self.metrics.items():
            if records:
                latest[source] = records[-1]["data"]

        summary["latest"] = latest
        return summary

    def _periodic_check(self):
        """Background thread for periodic system checks"""
        while not self.should_stop:
            try:
                # Run a comprehensive system check every 10 minutes
                if self.metrics:
                    self._run_comprehensive_check()
            except Exception as e:
                self.logger.error(f"Error in periodic check: {e}")

            # Sleep for 10 minutes
            for _ in range(600):
                if self.should_stop:
                    break
                time.sleep(1)

    def _run_comprehensive_check(self):
        """Run a comprehensive check of all metrics"""
        # Get all sources with their latest metrics
        latest_metrics = {}
        for source, records in self.metrics.items():
            if records:
                latest_metrics[source] = records[-1]["data"]

        if not latest_metrics:
            return

        output_structure = {
            "overall_health": "good",  # good, concerning, critical
            "issues": [
                {
                    "source": "source_name",
                    "severity": "medium",
                    "description": "Issue description"
                }
            ],
            "recommendations": ["recommendation1", "recommendation2"]
        }

        prompt = f"""
        Perform a comprehensive health check of the federated learning system based on the latest metrics from all sources:
        
        System Metrics:
        {json.dumps(latest_metrics, indent=2)}
        
        Evaluate:
        1. Overall system health
        2. Model performance and convergence
        3. Client participation and reliability
        4. Resource utilization
        5. Data quality and distribution
        6. Any concerning trends or patterns
        
        Provide an assessment of the overall system health and identify any issues that need attention.
        """

        assessment = self.llm.structured_query(prompt, output_structure)

        # Create alerts for any identified issues
        if assessment.get("overall_health") in ["concerning", "critical"]:
            for issue in assessment.get("issues", []):
                if issue.get("source") and issue.get("description"):
                    self.create_alert(
                        source=issue.get("source"),
                        severity=AlertSeverity(issue.get("severity", "medium")),
                        description=issue.get("description")
                    )

        self.logger.info(f"Comprehensive check completed. Overall health: {assessment.get('overall_health', 'unknown')}")

    def shutdown(self):
        """Shutdown the agent properly"""
        self.should_stop = True
        if self.check_thread.is_alive():
            self.check_thread.join(timeout=5)
        self._save_data()