from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid


class ClientStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClientRegistrationRequest(BaseModel):
    """Client registration request data model"""
    name: str
    dataset_description: Dict[str, Any]
    compute_capabilities: Dict[str, Any]
    privacy_level: str
    data_volume: int


class ClientDetails(BaseModel):
    """Registered client details"""
    client_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ClientStatus
    registration_time: datetime = Field(default_factory=datetime.now)
    registration_data: ClientRegistrationRequest
    last_active: Optional[datetime] = None


class ModelUpdate(BaseModel):
    """Client model update submission"""
    client_id: str
    round_id: int
    model_weights: Dict[str, List[float]]
    training_metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class FederatedRound(BaseModel):
    """Federated learning round configuration"""
    round_id: int
    model_configuration: Dict[str, Any]
    client_selection: List[str]
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "in_progress"
    aggregation_strategy: str = "federated_averaging"
    min_clients_required: int = 3


class Alert(BaseModel):
    """System alert"""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    severity: AlertSeverity
    description: str
    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class Strategy(BaseModel):
    """Improvement strategy"""
    strategy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    training_round: int
    weaknesses: List[str]
    aggregation_recommendations: Dict[str, Any]
    client_selection_strategy: Dict[str, Any]
    synthetic_data_recommendations: Optional[Dict[str, Any]] = None
    hyperparameter_recommendations: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class SyntheticDataset(BaseModel):
    """Synthetic dataset metadata"""
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: Dict[str, Any]
    generation_parameters: Dict[str, Any]
    size: int
    status: str = "pending"
    timestamp: datetime = Field(default_factory=datetime.now)