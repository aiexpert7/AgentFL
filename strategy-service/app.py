
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import Dict, Any, List, Optional
from sagent import StrategicAgent
from pydantic import BaseModel
import logging
from prometheus_client import make_asgi_app
# Initialize FastAPI app
app = FastAPI(title="Federated Learning Strategy Service")

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
strategic_agent = StrategicAgent()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyRequest(BaseModel):
    model_performance: Dict[str, Any]
    client_metrics: Dict[str, Any]
    current_round: int


class SyntheticDataRequest(BaseModel):
    data_description: Dict[str, Any]
    target_size: int


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "strategy"}


@app.post("/strategies")
async def create_strategy(request: StrategyRequest):
    """Create an improvement strategy"""
    strategy = strategic_agent.create_improvement_strategy(
        model_performance=request.model_performance,
        client_metrics=request.client_metrics,
        current_round=request.current_round
    )
    return strategy


@app.get("/strategies")
async def get_strategies():
    """Get all strategies"""
    strategies = strategic_agent.get_all_strategies()
    return {"strategies": strategies}


@app.get("/strategies/best/{round_number}")
async def get_best_strategy(round_number: int):
    """Get best strategy for a specific round"""
    strategy = strategic_agent.get_best_strategy(round_number)
    if not strategy:
        return {}
    return strategy


@app.post("/synthetic-data")
async def generate_synthetic_data(request: SyntheticDataRequest):
    """Generate synthetic data"""
    dataset = strategic_agent.generate_synthetic_data(
        data_description=request.data_description,
        target_size=request.target_size
    )
    return dataset


@app.get("/synthetic-data/{dataset_id}")
async def get_synthetic_dataset(dataset_id: str):
    """Get information about a synthetic dataset"""
    dataset = strategic_agent.get_synthetic_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)