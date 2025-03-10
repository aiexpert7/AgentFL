from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import Dict, Any, List, Optional
from oagent import OrchestratorAgent
from pydantic import BaseModel
import logging
from prometheus_client import make_asgi_app
# Initialize FastAPI app
app = FastAPI(title="Federated Learning Orchestrator Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
# Initialize the agent
orchestrator_agent = OrchestratorAgent()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Optional initial model configuration"""
    config: Dict[str, Any]


class ClientUpdate(BaseModel):
    """Client model update submission"""
    client_id: str
    model_update: Dict[str, Any]
    metrics: Dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "orchestrator"}


@app.post("/training/start")
async def start_training(model_config: Optional[ModelConfig] = None):
    """Start the federated training process"""
    config = model_config.config if model_config else None
    success = orchestrator_agent.start_training(config)

    if not success:
        return {"status": "error", "message": "Training already in progress"}

    return {"status": "started"}


@app.post("/training/stop")
async def stop_training():
    """Stop the current training process"""
    success = orchestrator_agent.stop_training()

    if not success:
        return {"status": "error", "message": "No active training to stop"}

    return {"status": "stopped"}


@app.get("/training/status")
async def training_status():
    """Get current training status"""
    return orchestrator_agent.get_training_status()


@app.post("/updates")
async def submit_update(update: ClientUpdate):
    """Submit a model update from a client"""
    success = orchestrator_agent.receive_client_update(
        client_id=update.client_id,
        model_update=update.model_update,
        metrics=update.metrics
    )

    if not success:
        return {"status": "error", "message": "Update rejected"}

    return {"status": "received"}


@app.get("/model")
async def get_global_model():
    """Get the current global model"""
    return {"model": orchestrator_agent.get_global_model()}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8004))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)