import os
import logging
import json
import numpy as np
from openai import OpenAI  # Updated import
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential


def setup_logger(name):
    """Set up a logger with the given name"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class LLMInterface:
    """Interface for LLM interactions"""
    def __init__(self, system_role):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set OPENAI_API_KEY environment variable.")

        # Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.api_key)
        self.system_role = system_role
        self.logger = setup_logger("LLMInterface")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(self, prompt, temperature=0.2, max_tokens=1000):
        """Query the LLM with a prompt"""
        try:
            # Updated API call for chat completions
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            # Updated response structure
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    def structured_query(self, prompt, output_structure, temperature=0.2):
        """Query the LLM and expect structured JSON output"""
        enhanced_prompt = prompt + f"\n\nPlease format your response as JSON with this structure:\n{json.dumps(output_structure, indent=2)}"

        try:
            response = self.query(enhanced_prompt, temperature)
            # Extract JSON if response contains other text
            response = self._extract_json(response)
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Could not parse LLM response as JSON")
            return {"error": "Failed to get structured response"}

    def _extract_json(self, text):
        """Extract JSON from text that might contain markdown or other formatting"""
        if text.find("```json") >= 0 and text.find("```", text.find("```json") + 7) >= 0:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif text.find("{") >= 0 and text.find("}") >= 0:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        return text


# The rest of the code remains unchanged
def federated_average(updates: List[Dict[str, np.ndarray]], weights: List[float] = None) -> Dict[str, np.ndarray]:
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
        result[key] = updates[0][key] * weights[0]

        # Add the rest of the updates
        for i in range(1, len(updates)):
            result[key] += updates[i][key] * weights[i]

    return result


# Metrics for Prometheus
def setup_metrics():
    """Set up Prometheus metrics"""
    try:
        from prometheus_client import Counter, Gauge, Histogram

        metrics = {
            "client_registrations": Counter("fl_client_registrations_total", "Total number of client registrations"),
            "active_clients": Gauge("fl_active_clients", "Number of active clients"),
            "training_rounds": Counter("fl_training_rounds_total", "Total number of training rounds"),
            "model_updates": Counter("fl_model_updates_total", "Total number of model updates"),
            "aggregation_time": Histogram("fl_aggregation_time_seconds", "Time taken for model aggregation"),
            "alerts": Counter("fl_alerts_total", "Total number of alerts", ["severity"]),
            "model_performance": Gauge("fl_model_performance", "Model performance metrics", ["metric"])
        }

        return metrics
    except ImportError:
        return {}