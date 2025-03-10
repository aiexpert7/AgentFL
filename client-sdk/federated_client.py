# client-sdk/federated_client.py
import requests
import json
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class FederatedClient:
    """Client SDK for participating in federated learning"""

    def __init__(self, server_url="http://localhost:8000", client_name=None):
        """
        Initialize the federated learning client

        Args:
            server_url: The URL of the federated learning server
            client_name: A name for this client
        """
        self.server_url = server_url.rstrip('/')
        self.client_name = client_name or f"client-{int(time.time())}"
        self.client_id = None
        self.registered = False
        self.current_round = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up a logger for the client"""
        logger = logging.getLogger(f"FederatedClient.{self.client_name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def register(self, dataset_description: Dict[str, Any],
                 compute_capabilities: Dict[str, Any],
                 privacy_level: str = "moderate",
                 data_volume: int = 1000) -> bool:
        """
        Register this client with the federated learning server

        Args:
            dataset_description: Description of the client's dataset
            compute_capabilities: Description of the client's compute resources
            privacy_level: Privacy level (low, moderate, high)
            data_volume: Number of samples in the client's dataset

        Returns:
            bool: True if registration was successful
        """
        if self.registered:
            self.logger.warning("Client already registered")
            return True

        registration_data = {
            "name": self.client_name,
            "dataset_description": dataset_description,
            "compute_capabilities": compute_capabilities,
            "privacy_level": privacy_level,
            "data_volume": data_volume
        }

        try:
            response = requests.post(
                f"{self.server_url}/register",
                json=registration_data
            )


            response.raise_for_status()
            result = response.json()
            self.client_id = result.get("client_id")
            self.logger.info(f"Registration initiated with ID: {self.client_id}")

            # Wait for registration approval
            return self._wait_for_approval()

        except requests.RequestException as e:
            self.logger.error(f"Error during registration: {e}")
            raise

    def _wait_for_approval(self, max_attempts=10, interval=5) -> bool:
        """
        Wait for registration approval

        Args:
            max_attempts: Maximum number of attempts to check status
            interval: Interval between attempts in seconds

        Returns:
            bool: True if registration was approved
        """
        if not self.client_id:
            return False

        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.server_url}/api/registration/registration-status/{self.client_id}"
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status")

                    if status == "approved":
                        self.registered = True
                        self.logger.info(f"Registration approved for client {self.client_id}")
                        return True
                    elif status == "rejected":
                        self.logger.error("Registration rejected")
                        return False
                    else:
                        self.logger.info(f"Registration status: {status}, waiting... ({attempt+1}/{max_attempts})")
                else:
                    self.logger.warning(f"Failed to check status: {response.text}")

            except Exception as e:
                self.logger.error(f"Error checking registration status: {e}")

            time.sleep(interval)

        self.logger.warning("Registration not approved after maximum attempts")
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def check_for_training_task(self) -> Optional[Dict[str, Any]]:
        """
        Check if there's an active training task for this client

        Returns:
            Optional[Dict]: Training task details or None
        """
        if not self.registered or not self.client_id:
            self.logger.error("Client not registered")
            return None

        try:
            # Check training status
            response = requests.get(f"{self.server_url}/api/orchestrator/training/status")
            response.raise_for_status()
            status = response.json()

            if not status.get("active", False):
                return None

            # Get the global model if active
            model_response = requests.get(f"{self.server_url}/api/orchestrator/model")
            model_response.raise_for_status()
            model_data = model_response.json().get("model", {})

            current_round = status.get("current_round")

            # Only update if it's a new round
            if current_round != self.current_round:
                task = {
                    "round_id": current_round,
                    "model": model_data
                }

                self.current_round = current_round
                self.logger.info(f"Received new training task for round {current_round}")
                return task

            return None

        except requests.RequestException as e:
            self.logger.error(f"Error checking for training task: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error checking for training task: {e}")
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def submit_update(self, model_update: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """
        Submit a model update after local training

        Args:
            model_update: Updated model parameters
            metrics: Training metrics

        Returns:
            bool: True if submission was successful
        """
        if not self.registered or not self.client_id or not self.current_round:
            self.logger.error("Client not ready to submit updates")
            return False

        update_data = {
            "client_id": self.client_id,
            "model_update": model_update,
            "metrics": metrics
        }

        try:
            response = requests.post(
                f"{self.server_url}/api/orchestrator/updates",
                json=update_data
            )

            response.raise_for_status()
            result = response.json()

            if result.get("status") == "received":
                self.logger.info(f"Update for round {self.current_round} submitted successfully")
                return True
            else:
                self.logger.warning(f"Update submission response: {result}")
                return False

        except requests.RequestException as e:
            self.logger.error(f"Error submitting update: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error submitting update: {e}")
            return False

    def train_and_update(self, train_function: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]],
                         interval: int = 60, max_rounds: int = None) -> None:
        """
        Continuously check for training tasks, train locally, and submit updates

        Args:
            train_function: Function that takes model parameters and returns (update, metrics)
            interval: Interval between checks in seconds
            max_rounds: Maximum number of rounds to participate in (None for unlimited)
        """
        rounds_completed = 0

        self.logger.info(f"Starting training loop with check interval {interval}s")

        while max_rounds is None or rounds_completed < max_rounds:
            try:
                # Check for training task
                task = self.check_for_training_task()

                if task:
                    round_id = task.get("round_id")
                    model = task.get("model")

                    self.logger.info(f"Processing training task for round {round_id}")

                    # Train locally
                    try:
                        self.logger.info("Starting local training...")
                        start_time = time.time()
                        update, metrics = train_function(model)
                        training_time = time.time() - start_time

                        # Add training time to metrics
                        metrics["training_time"] = training_time

                        self.logger.info(f"Local training completed in {training_time:.2f}s")

                        # Submit update
                        success = self.submit_update(update, metrics)

                        if success:
                            rounds_completed += 1
                            self.logger.info(f"Completed round {round_id}, total rounds completed: {rounds_completed}")
                    except Exception as e:
                        self.logger.error(f"Error during local training: {e}")

                # Sleep before next check
                time.sleep(interval)

            except KeyboardInterrupt:
                self.logger.info("Training loop interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in training loop: {e}")
                time.sleep(interval)  # Sleep before retry

    def get_latest_global_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest global model

        Returns:
            Optional[Dict]: Global model parameters or None
        """
        if not self.registered:
            self.logger.error("Client not registered")
            return None

        try:
            response = requests.get(f"{self.server_url}/api/orchestrator/model")

            if response.status_code == 200:
                return response.json().get("model", {})
            else:
                self.logger.error(f"Failed to get global model: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting global model: {e}")
            return None

