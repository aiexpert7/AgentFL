# registration-service/agent.py
from shared.utils import LLMInterface, setup_logger
from shared.models import ClientDetails, ClientRegistrationRequest, ClientStatus
from typing import Dict, List, Optional, Any
import json
import datetime
import uuid
import os
from prometheus_client import Counter, Gauge

# Prometheus metrics
CLIENT_REGISTRATIONS = Counter('fl_client_registrations_total', 'Total number of client registrations')
APPROVED_CLIENTS = Counter('fl_approved_clients_total', 'Total number of approved clients')
REJECTED_CLIENTS = Counter('fl_rejected_clients_total', 'Total number of rejected clients')
ACTIVE_CLIENTS = Gauge('fl_active_clients', 'Number of active clients')


class RegistrationAgent:
    """Agent that handles client registration for federated learning"""

    def __init__(self):
        self.logger = setup_logger("RegistrationAgent")
        self.clients: Dict[str, ClientDetails] = {}
        self.pending_registrations: Dict[str, ClientDetails] = {}

        # Initialize LLM interface
        self.llm = LLMInterface(
            system_role="You are a registration evaluation agent for a federated learning system. "
                        "You evaluate client registration requests based on data quality, privacy compliance, "
                        "and resource capabilities to determine if they should be automatically approved."
        )

        self.data_dir = os.environ.get("DATA_DIR", "/data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing data if available
        self._load_data()

        # Update metrics
        ACTIVE_CLIENTS.set(len(self.clients))

        self.logger.info("RegistrationAgent initialized")

    def _load_data(self):
        """Load existing registration data if available"""
        clients_file = os.path.join(self.data_dir, "clients.json")
        pending_file = os.path.join(self.data_dir, "pending_registrations.json")

        if os.path.exists(clients_file):
            try:
                with open(clients_file, 'r') as f:
                    clients_data = json.load(f)
                    for client_id, client_data in clients_data.items():
                        # Convert date strings back to datetime objects
                        if 'registration_time' in client_data and isinstance(client_data['registration_time'], str):
                            client_data['registration_time'] = datetime.datetime.fromisoformat(client_data['registration_time'])
                        if 'last_active' in client_data and client_data['last_active'] and isinstance(client_data['last_active'], str):
                            client_data['last_active'] = datetime.datetime.fromisoformat(client_data['last_active'])

                        # Create ClientDetails object
                        self.clients[client_id] = ClientDetails(**client_data)
                self.logger.info(f"Loaded {len(self.clients)} existing clients")
            except Exception as e:
                self.logger.error(f"Error loading clients data: {e}")

        if os.path.exists(pending_file):
            try:
                with open(pending_file, 'r') as f:
                    pending_data = json.load(f)
                    for client_id, client_data in pending_data.items():
                        # Convert date strings back to datetime objects
                        if 'registration_time' in client_data and isinstance(client_data['registration_time'], str):
                            client_data['registration_time'] = datetime.datetime.fromisoformat(client_data['registration_time'])

                        # Create ClientDetails object
                        self.pending_registrations[client_id] = ClientDetails(**client_data)
                self.logger.info(f"Loaded {len(self.pending_registrations)} pending registrations")
            except Exception as e:
                self.logger.error(f"Error loading pending registrations: {e}")

    def _save_data(self):
        """Save current registration data"""
        clients_file = os.path.join(self.data_dir, "clients.json")
        pending_file = os.path.join(self.data_dir, "pending_registrations.json")

        try:
            with open(clients_file, 'w') as f:
                clients_dict = {cid: client.dict() for cid, client in self.clients.items()}
                json.dump(clients_dict, f, default=str)

            with open(pending_file, 'w') as f:
                pending_dict = {cid: client.dict() for cid, client in self.pending_registrations.items()}
                json.dump(pending_dict, f, default=str)
        except Exception as e:
            self.logger.error(f"Error saving registration data: {e}")

    def initiate_registration(self, registration_data: ClientRegistrationRequest) -> str:
        """Process a new client registration request"""
        # Increment registration counter
        CLIENT_REGISTRATIONS.inc()

        # Generate client ID
        client_id = str(uuid.uuid4())

        # Create client details object
        client = ClientDetails(
            client_id=client_id,
            status=ClientStatus.PENDING,
            registration_data=registration_data,
            registration_time=datetime.datetime.now()
        )

        # Store in pending registrations
        self.pending_registrations[client_id] = client

        # Use LLM to evaluate the client
        try:
            evaluation = self._evaluate_client(registration_data)

            if evaluation.get("approved", False):
                self.approve_registration(client_id)
                self.logger.info(f"Automatically approved client {client_id}: {evaluation.get('reason')}")
            else:
                self.logger.info(f"Client {client_id} pending manual review: {evaluation.get('reason')}")
        except Exception as e:
            self.logger.error(f"Error evaluating client {client_id}: {e}")

        # Save updated data
        self._save_data()

        return client_id

    def _evaluate_client(self, registration_data: ClientRegistrationRequest) -> Dict[str, Any]:
        """Use LLM to evaluate if a client should be automatically approved"""
        output_structure = {
            "approved": False,
            "reason": "Explanation for decision",
            "risk_factors": ["list", "of", "risk", "factors"]
        }

        prompt = f"""
        Evaluate this client registration for federated learning:
        
        Client Details:
        {json.dumps(registration_data.dict(), indent=2)}
        
        Determine if this client should be automatically approved based on:
        1. Data quality indicators
        2. Data volume (is it substantial enough to be valuable?)
        3. Client compute capabilities (can they handle training?)
        4. Data privacy compliance
        
        If there are any red flags or concerns, the client should not be automatically approved.
        """

        try:
            return self.llm.structured_query(prompt, output_structure)
        except Exception as e:
            self.logger.error(f"Error evaluating client: {e}")
            return {
                "approved": False,
                "reason": f"Evaluation error: {str(e)}",
                "risk_factors": ["evaluation_failure"]
            }

    def approve_registration(self, client_id: str) -> bool:
        """Approve a pending registration"""
        if client_id not in self.pending_registrations:
            return False

        # Update client status and move to approved clients
        client = self.pending_registrations[client_id]
        client.status = ClientStatus.APPROVED
        client.last_active = datetime.datetime.now()

        self.clients[client_id] = client
        del self.pending_registrations[client_id]

        # Update metrics
        APPROVED_CLIENTS.inc()
        ACTIVE_CLIENTS.set(len(self.clients))

        self.logger.info(f"Approved registration for client {client_id}")
        self._save_data()

        return True

    def reject_registration(self, client_id: str, reason: str) -> bool:
        """Reject a pending registration"""
        if client_id not in self.pending_registrations:
            return False

        # Update client status to rejected
        client = self.pending_registrations[client_id]
        client.status = ClientStatus.REJECTED

        # Could keep a record of rejected clients if needed
        del self.pending_registrations[client_id]

        # Update metrics
        REJECTED_CLIENTS.inc()

        self.logger.info(f"Rejected registration for client {client_id}: {reason}")
        self._save_data()

        return True

    def get_registered_clients(self) -> List[str]:
        """Get IDs of all registered clients"""
        return list(self.clients.keys())

    def get_client_data(self, client_id: str) -> Optional[ClientDetails]:
        """Get data for a specific client"""
        return self.clients.get(client_id)

    def get_pending_registrations(self) -> Dict[str, ClientDetails]:
        """Get all pending registrations"""
        return self.pending_registrations

    def get_client_status(self, client_id: str) -> Optional[ClientStatus]:
        """Get status of a client (approved, pending, or not found)"""
        if client_id in self.clients:
            return ClientStatus.APPROVED
        elif client_id in self.pending_registrations:
            return self.pending_registrations[client_id].status
        else:
            return None

    def update_client_activity(self, client_id: str) -> bool:
        """Update the last active timestamp for a client"""
        if client_id in self.clients:
            self.clients[client_id].last_active = datetime.datetime.now()
            self._save_data()
            return True
        return False