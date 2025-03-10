import os
import json
import time
import datetime
import uuid
import threading
import requests
import numpy as np
from queue import Queue
from typing import Dict, List, Optional, Any, Set
import logging

class OrchestratorAgent:
    """Agent that orchestrates the federated learning process"""

    def __init__(self):
        self.logger = self._setup_logger()

        # Service endpoints
        self.registration_url = os.environ.get("REGISTRATION_SERVICE_URL", "http://registration-service:8001")
        self.monitoring_url = os.environ.get("MONITORING_SERVICE_URL", "http://monitoring-service:8002")
        self.strategy_url = os.environ.get("STRATEGY_SERVICE_URL", "http://strategy-service:8003")

        # Training state
        self.current_round = None
        self.rounds_history = []
        self.client_updates = {}
        self.global_model = {}
        self.active_training = False

        # Configuration
        self.min_clients = int(os.environ.get("MIN_CLIENTS", "3"))
        self.max_waiting_time = int(os.environ.get("MAX_WAITING_TIME", "300"))  # seconds
        self.training_rounds = int(os.environ.get("TRAINING_ROUNDS", "10"))

        # LLM configuration
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.warning("OpenAI API key not found, LLM features will be limited")

        # Data storage
        self.data_dir = os.environ.get("DATA_DIR", "./data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing data if available
        self._load_data()

        # Start background worker for processing training
        self.should_stop = False
        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()

        self.logger.info("OrchestratorAgent initialized")

    def _setup_logger(self):
        """Set up a logger with the given name"""
        logger = logging.getLogger("OrchestratorAgent")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt and return the response"""
        if not self.api_key:
            self.logger.error("Cannot call LLM: API key not set")
            return ""

        try:
            import openai
            openai.api_key = self.api_key

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an orchestrator agent for a federated learning system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"

    def _load_data(self):
        """Load existing training data if available"""
        rounds_file = os.path.join(self.data_dir, "rounds_history.json")
        model_file = os.path.join(self.data_dir, "global_model.json")

        if os.path.exists(rounds_file):
            try:
                with open(rounds_file, 'r') as f:
                    self.rounds_history = json.load(f)
                self.logger.info(f"Loaded {len(self.rounds_history)} training rounds from history")
            except Exception as e:
                self.logger.error(f"Error loading rounds history: {e}")

        if os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    self.global_model = json.load(f)
                self.logger.info(f"Loaded global model with {len(self.global_model)} parameters")
            except Exception as e:
                self.logger.error(f"Error loading global model: {e}")

    def _save_data(self):
        """Save current training data"""
        rounds_file = os.path.join(self.data_dir, "rounds_history.json")
        model_file = os.path.join(self.data_dir, "global_model.json")

        try:
            with open(rounds_file, 'w') as f:
                json.dump(self.rounds_history, f, default=str)

            with open(model_file, 'w') as f:
                json.dump(self.global_model, f, default=str)
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def _training_worker(self):
        """Background worker to process training rounds"""
        while not self.should_stop:
            if self.active_training and self.current_round:
                # Check if we've waited long enough or have enough updates
                if self._should_aggregate():
                    self._aggregate_results()

                    # Report round completion to monitoring
                    self._report_round_metrics()

                    # Check if we've completed all rounds
                    if self.current_round["round_id"] >= self.training_rounds:
                        self.active_training = False
                        self.logger.info(f"Completed all {self.training_rounds} training rounds")
                    else:
                        # Start the next round
                        self._start_next_round()

            # Sleep a bit to avoid busy waiting
            time.sleep(1)

    def _should_aggregate(self) -> bool:
        """Determine if we should aggregate results for the current round"""
        if not self.current_round:
            return False

        # Check if we have enough client updates
        if len(self.client_updates) >= self.current_round.get("min_clients_required", self.min_clients):
            return True

        # Check if we've waited too long
        start_time = datetime.datetime.fromisoformat(self.current_round.get("start_time"))
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        if elapsed_time > self.max_waiting_time:
            self.logger.info(f"Max waiting time reached for round {self.current_round['round_id']}")
            return True

        return False

    def _aggregate_results(self):
        """Aggregate client updates and update the global model"""
        if not self.current_round or not self.client_updates:
            return

        self.logger.info(f"Aggregating results for round {self.current_round['round_id']} "
                         f"with {len(self.client_updates)} client updates")

        # Convert client updates to the format expected by federated_average
        updates = []
        weights = []

        for client_id, update in self.client_updates.items():
            updates.append(update["model_update"])

            # Use data size as weight if available, otherwise equal weights
            client_weight = update.get("metrics", {}).get("data_size", 1.0)
            weights.append(float(client_weight))

        # Perform federated averaging
        if updates:
            try:
                self.global_model = self._federated_average(updates, weights)

                # Update round status and store in history
                self.current_round["end_time"] = datetime.datetime.now().isoformat()
                self.current_round["status"] = "completed"
                self.current_round["participating_clients"] = list(self.client_updates.keys())
                self.rounds_history.append(self.current_round)

                # Save updated data
                self._save_data()

                self.logger.info(f"Successfully aggregated round {self.current_round['round_id']}")
            except Exception as e:
                self.logger.error(f"Error aggregating results: {e}")

                # Report error to monitoring
                self._report_error("aggregation_error", str(e))

    def _federated_average(self, updates: List[Dict[str, Any]], weights: List[float] = None) -> Dict[str, Any]:
        """
        Perform federated averaging on model updates

        Args:
            updates: List of model parameter dictionaries
            weights: Optional weights for each update (e.g., based on data size)

        Returns:
            Averaged model parameters
        """
        if not updates:
            return {}

        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)

        # Normalize weights
        weights = [w / sum(weights) for w in weights]

        result = {}
        for key in updates[0].keys():
            # Initialize with the first update
            if isinstance(updates[0][key], list):
                # Handle list parameters (common for neural network weights)
                result[key] = [w * weights[0] for w in updates[0][key]]

                # Add the rest of the updates
                for i in range(1, len(updates)):
                    result[key] = [r + updates[i][key][j] * weights[i]
                                   for j, r in enumerate(result[key])]
            else:
                # Handle scalar parameters
                result[key] = updates[0][key] * weights[0]

                # Add the rest of the updates
                for i in range(1, len(updates)):
                    result[key] += updates[i][key] * weights[i]

        return result

    def _start_next_round(self):
        """Start the next training round"""
        if not self.active_training:
            return

        next_round_id = (self.current_round["round_id"] + 1) if self.current_round else 1

        # Get strategy recommendation
        try:
            response = requests.get(f"{self.strategy_url}/strategies/best/{next_round_id}")
            if response.status_code == 200:
                strategy = response.json()
            else:
                strategy = None
        except Exception as e:
            self.logger.error(f"Error getting strategy: {e}")
            strategy = None

        # Get registered clients
        try:
            response = requests.get(f"{self.registration_url}/admin/clients")
            if response.status_code == 200:
                clients = response.json().get("clients", [])
            else:
                clients = []
        except Exception as e:
            self.logger.error(f"Error getting clients: {e}")
            clients = []

        if not clients:
            self.logger.error("No clients available, pausing training")
            self.active_training = False
            return

        # Select clients for this round (using strategy recommendations if available)
        selected_clients = self._select_clients(clients, strategy)

        if not selected_clients:
            self.logger.error("No clients selected, pausing training")
            self.active_training = False
            return

        # Create new round configuration
        self.current_round = {
            "round_id": next_round_id,
            "model_config": self.global_model if self.global_model else {"initial": True},
            "client_selection": selected_clients,
            "min_clients_required": min(len(selected_clients), self.min_clients),
            "start_time": datetime.datetime.now().isoformat(),
            "status": "in_progress",
            "aggregation_strategy": "federated_averaging"
        }

        # Reset client updates for new round
        self.client_updates = {}

        self.logger.info(f"Started round {next_round_id} with {len(selected_clients)} selected clients")

        # Notify monitoring of new round
        self._report_round_start()

    def _select_clients(self, clients: List[Dict[str, Any]], strategy: Optional[Dict[str, Any]]) -> List[str]:
        """Select clients for the current round based on strategy"""
        if not clients:
            return []

        client_ids = [client.get("client_id") for client in clients if client.get("client_id")]

        # If we have a strategy with client selection recommendations, use it
        if strategy and "client_selection_strategy" in strategy:
            selection_strategy = strategy["client_selection_strategy"]

            # Use LLM to select clients based on the strategy
            prompt = f"""
            Select clients for federated learning round based on this strategy:
            
            Strategy:
            {json.dumps(selection_strategy, indent=2)}
            
            Available clients:
            {json.dumps(clients, indent=2)}
            
            Select the most appropriate clients that match the strategy criteria.
            Return only client IDs that exist in the available clients list.
            Format your response as a JSON array of client IDs.
            """

            response = self._call_llm(prompt)
            try:
                selected_ids = json.loads(response)
                if isinstance(selected_ids, list):
                    # Ensure all selected clients actually exist
                    verified_ids = [cid for cid in selected_ids if cid in client_ids]

                    if verified_ids:
                        return verified_ids
            except:
                self.logger.error("Could not parse LLM response for client selection")

        # Fallback: select all clients or random subset if too many
        import random
        if len(client_ids) > 10:
            return random.sample(client_ids, 10)
        else:
            return client_ids

    def start_training(self, model_config: Optional[Dict[str, Any]] = None) -> bool:
        """Start the federated training process"""
        if self.active_training:
            return False

        self.active_training = True

        # If model config provided, use it as initial global model
        if model_config:
            self.global_model = model_config

        # Start first round
        self._start_next_round()

        return True

    def stop_training(self) -> bool:
        """Stop the current training process"""
        if not self.active_training:
            return False

        self.active_training = False

        if self.current_round:
            self.current_round["status"] = "stopped"
            self.current_round["end_time"] = datetime.datetime.now().isoformat()
            self.rounds_history.append(self.current_round)
            self.current_round = None

        self._save_data()
        self.logger.info("Training stopped")

        return True

    def receive_client_update(self, client_id: str, model_update: Dict[str, Any],
                              metrics: Dict[str, Any]) -> bool:
        """Process a model update from a client"""
        if not self.active_training or not self.current_round:
            return False

        # Verify client is part of current round
        if client_id not in self.current_round["client_selection"]:
            self.logger.warning(f"Received update from client {client_id} not in current round")
            return False

        # Store update
        self.client_updates[client_id] = {
            "model_update": model_update,
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.logger.info(f"Received update from client {client_id} for round {self.current_round['round_id']}")

        # Report metrics to monitoring
        self._report_client_metrics(client_id, metrics)

        return True

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            "active": self.active_training,
            "current_round": self.current_round["round_id"] if self.current_round else None,
            "total_rounds": self.training_rounds,
            "completed_rounds": len(self.rounds_history),
            "clients_in_round": len(self.current_round["client_selection"]) if self.current_round else 0,
            "clients_reported": len(self.client_updates) if self.current_round else 0
        }
        return status

    def get_global_model(self) -> Dict[str, Any]:
        """Get the current global model"""
        return self.global_model

    def _report_round_start(self):
        """Report round start to monitoring service"""
        if not self.current_round:
            return

        metrics = {
            "event": "round_start",
            "round_id": self.current_round["round_id"],
            "selected_clients": len(self.current_round["client_selection"]),
            "min_clients_required": self.current_round["min_clients_required"],
            "global_model_size": len(self.global_model) if self.global_model else 0
        }

        try:
            requests.post(
                f"{self.monitoring_url}/metrics",
                json={"source": "orchestrator", "metrics": metrics}
            )
        except Exception as e:
            self.logger.error(f"Error reporting to monitoring: {e}")

    def _report_round_metrics(self):
        """Report round metrics to monitoring service"""
        if not self.current_round:
            return

        metrics = {
            "event": "round_complete",
            "round_id": self.current_round["round_id"],
            "participating_clients": len(self.client_updates),
            "duration_seconds": (datetime.datetime.now() -
                                 datetime.datetime.fromisoformat(self.current_round["start_time"])).total_seconds(),
            "client_metrics": {}
        }

        # Aggregate client metrics
        for client_id, update in self.client_updates.items():
            for metric, value in update["metrics"].items():
                if metric not in metrics["client_metrics"]:
                    metrics["client_metrics"][metric] = []
                metrics["client_metrics"][metric].append(value)

        # Calculate statistics on metrics
        for metric, values in metrics["client_metrics"].items():
            try:
                metrics[f"avg_{metric}"] = sum(values) / len(values)

                # Calculate standard deviation
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    metrics[f"std_{metric}"] = variance ** 0.5

                metrics[f"min_{metric}"] = min(values)
                metrics[f"max_{metric}"] = max(values)
            except:
                pass

        try:
            requests.post(
                f"{self.monitoring_url}/metrics",
                json={"source": "orchestrator", "metrics": metrics}
            )
        except Exception as e:
            self.logger.error(f"Error reporting to monitoring: {e}")

    def _report_client_metrics(self, client_id: str, metrics: Dict[str, Any]):
        """Report client metrics to monitoring service"""
        try:
            requests.post(
                f"{self.monitoring_url}/metrics",
                json={"source": f"client_{client_id}", "metrics": metrics}
            )
        except Exception as e:
            self.logger.error(f"Error reporting to monitoring: {e}")

    def _report_error(self, error_type: str, error_message: str):
        """Report error to monitoring service"""
        try:
            requests.post(
                f"{self.monitoring_url}/alerts",
                json={
                    "source": "orchestrator",
                    "severity": "high" if error_type.startswith("critical") else "medium",
                    "description": f"{error_type}: {error_message}"
                }
            )
        except Exception as e:
            self.logger.error(f"Error reporting to monitoring: {e}")

    def shutdown(self):
        """Shutdown the agent properly"""
        self.should_stop = True
        if self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        self._save_data()
