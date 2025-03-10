# client-sdk/example_client.py
import numpy as np
import time
import logging
import argparse
from federated_client import FederatedClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ExampleClient")

def train_on_local_data(model):
    """
    Example implementation of local training function

    Args:
        model: Global model parameters

    Returns:
        Tuple[Dict, Dict]: (model_update, metrics)
    """
    # In a real implementation, this would train the model on local data
    # For this example, we just add some noise to simulate training

    logger.info("Training on local data...")
    time.sleep(5)  # Simulate training time

    # Create model update (in real implementation, this would be from actual training)
    update = {}
    for key, value in model.items():
        if isinstance(value, list):
            # Add small random updates to parameters
            update[key] = [v + np.random.normal(0, 0.01) for v in value]
        else:
            update[key] = value

    # Simulate metrics from training
    metrics = {
        "loss": np.random.uniform(0.1, 0.5),
        "accuracy": np.random.uniform(0.7, 0.95),
        "data_size": 1000,
        "epochs": 5,
        "batch_size": 32
    }

    return update, metrics


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                        help='URL of the federated learning server')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for this client')
    parser.add_argument('--interval', type=int, default=10,
                        help='Interval between checking for new tasks (in seconds)')
    parser.add_argument('--max-rounds', type=int, default=None,
                        help='Maximum number of rounds to participate in')
    parser.add_argument('--data-size', type=int, default=1000,
                        help='Size of the local dataset')

    args = parser.parse_args()

    # Create federated learning client
    client = FederatedClient(
        server_url=args.server,
        client_name=args.name
    )

    # Register with the server
    registration_successful = client.register(
        dataset_description={
            "type": "image",
            "classes": 10,
            "distribution": "balanced"
        },
        compute_capabilities={
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu": False
        },
        privacy_level="moderate",
        data_volume=args.data_size
    )

    if not registration_successful:
        logger.error("Registration failed, exiting")
        return

    logger.info(f"Registration successful! Client ID: {client.client_id}")

    # Participate in training rounds
    client.train_and_update(
        train_function=train_on_local_data,
        interval=args.interval,
        max_rounds=args.max_rounds
    )


if __name__ == "__main__":
    main()