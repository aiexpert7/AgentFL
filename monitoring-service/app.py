from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import Dict, Any, List
from magent import MonitoringAgent
from shared.models import Alert, AlertSeverity
from pydantic import BaseModel
import logging
from prometheus_client import make_asgi_app

# Initialize FastAPI app
app = FastAPI(title="Federated Learning Monitoring Service")

# Add prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
monitoring_agent = MonitoringAgent()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsData(BaseModel):
    source: str
    metrics: Dict[str, Any]


class AlertCreate(BaseModel):
    source: str
    severity: AlertSeverity
    description: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "monitoring"}


@app.post("/metrics")
async def report_metrics(metrics_data: MetricsData):
    """Record metrics from a source"""
    success = monitoring_agent.record_metrics(metrics_data.source, metrics_data.metrics)
    return {"status": "recorded", "success": success}


@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary of current metrics"""
    return monitoring_agent.get_summary_metrics()


@app.get("/alerts")
async def get_alerts(active_only: bool = True):
    """Get system alerts"""
    if active_only:
        alerts = monitoring_agent.get_active_alerts()
    else:
        alerts = monitoring_agent.get_all_alerts()
    return {"alerts": alerts}


@app.post("/alerts")
async def create_alert(alert_data: AlertCreate):
    """Manually create an alert"""
    alert = monitoring_agent.create_alert(
        source=alert_data.source,
        severity=alert_data.severity,
        description=alert_data.description
    )
    return alert
# Add structured metrics collection in monitoring-service/agent.py
def record_round_metrics(self, round_id, global_metrics, client_metrics):
    """Store detailed metrics for each training round"""
    timestamp = datetime.datetime.now().isoformat()
    metrics_record = {
        "timestamp": timestamp,
        "round_id": round_id,
        "global_metrics": global_metrics,
        "client_metrics": client_metrics
    }
    self.db.store_metrics(metrics_record)

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    success = monitoring_agent.resolve_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "resolved", "alert_id": alert_id}

# In monitoring-service/app.py
@app.get("/visualizations/model-performance")
async def get_model_performance():
    """Return model performance metrics in visualization-ready format"""
    metrics = monitoring_agent.get_performance_metrics()
    return format_for_visualization(metrics)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)