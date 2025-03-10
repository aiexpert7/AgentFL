# registration-service/app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List, Dict, Any, Optional
from ragent import RegistrationAgent
from shared.models import ClientRegistrationRequest, ClientStatus
from prometheus_client import make_asgi_app
import logging


# Initialize FastAPI app
app = FastAPI(title="Federated Learning Registration Service")

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
registration_agent = RegistrationAgent()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "registration"}


@app.post("/register")
async def register(registration_data: ClientRegistrationRequest, background_tasks: BackgroundTasks):
    """Register a new client"""
    try:
        client_id = registration_agent.initiate_registration(registration_data)
        logger.info(f"New client registration initiated: {client_id}")
        return {"client_id": client_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.get("/registration-status/{client_id}")
async def registration_status(client_id: str):
    """Check registration status for a client"""
    status = registration_agent.get_client_status(client_id)

    if status == ClientStatus.APPROVED:
        return {"status": "approved", "client_id": client_id}
    elif status == ClientStatus.PENDING:
        return {"status": "pending", "client_id": client_id}
    elif status == ClientStatus.REJECTED:
        return {"status": "rejected", "client_id": client_id}
    else:
        raise HTTPException(status_code=404, detail="Client not found")


@app.post("/admin/approve/{client_id}")
async def admin_approve(client_id: str):
    """Admin endpoint to approve a pending registration"""
    success = registration_agent.approve_registration(client_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pending registration not found")
    logger.info(f"Admin approved client: {client_id}")
    return {"status": "approved", "client_id": client_id}


@app.post("/admin/reject/{client_id}")
async def admin_reject(client_id: str, reason: Dict[str, str]):
    """Admin endpoint to reject a pending registration"""
    success = registration_agent.reject_registration(client_id, reason.get("reason", "No reason provided"))
    if not success:
        raise HTTPException(status_code=404, detail="Pending registration not found")
    logger.info(f"Admin rejected client: {client_id}")
    return {"status": "rejected", "client_id": client_id}


@app.get("/admin/clients")
async def admin_get_clients():
    """Admin endpoint to get all registered clients"""
    client_ids = registration_agent.get_registered_clients()
    clients = []
    for client_id in client_ids:
        client_data = registration_agent.get_client_data(client_id)
        if client_data:
            clients.append(client_data)
    return {"clients": clients}


@app.get("/admin/pending")
async def admin_get_pending():
    """Admin endpoint to get all pending registrations"""
    pending = registration_agent.get_pending_registrations()
    return {"pending": pending}


@app.get("/clients/count")
async def get_client_count():
    """Get the number of registered clients"""
    client_count = len(registration_agent.get_registered_clients())
    return {"count": client_count}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting Registration Service on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)