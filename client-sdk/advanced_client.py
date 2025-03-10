

# client-sdk/advanced_client.py
import numpy as np
import time
import logging
import argparse
import tensorflow as tf
import os
import json
from federated_client import FederatedClient
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedClient")

class FederatedMNISTClient:
    """
    Advanced client implementation that uses TensorFlow for MNIST training
    """

    def __init__(self, data_size=1000, batch_size=32, epochs=5):
        """
        Initialize the MNIST client

        Args:
            data_size: Number of samples to use from MNIST
            batch_size: Batch size for training
            epochs: Number of epochs to train for
        """
        self.data_size = data_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

        # Load and prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Load and prepare MNIST data"""
        logger.info("Loading MNIST dataset...")

        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Reshape for CNN
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Limit to data_size
        if self.data_size and self.data_size < len(x_train):
            indices = np.random.choice(len(x_train), self.data_size, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        logger.info(f"Data prepared: {len(self.x_train)} training samples, {len(self.x_test)} test samples")

    def _create_model(self):
        """Create a simple CNN model for MNIST"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _model_to_weights(self, model) -> Dict[str, Any]:
        """Convert model weights to dictionary format"""
        weights = {}
        for i, layer in enumerate(model.weights):
            weights[f"layer_{i}"] = layer.numpy().tolist()
        return weights

    def _weights_to_model(self, weights: Dict[str, Any], model=None):
        """Apply weights from dictionary to model"""
        if model is None:
            model = self._create_model()

        weight_values = []
        for i in range(len(model.weights)):
            weight_key = f"layer_{i}"
            if weight_key in weights:
                weight_values.append(np.array(weights[weight_key]))

        if weight_values:
            model.set_weights(weight_values)

        return model

    def train(self, global_model: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Train the model on local data

        Args:
            global_model: Global model parameters

        Returns:
            Tuple[Dict, Dict]: (model_update, metrics)
        """
        # Initialize or update the model with global weights
        if not self.model or global_model:
            self.model = self._weights_to_model(global_model) if global_model else self._create_model()

        logger.info(f"Starting training for {self.epochs} epochs with batch size {self.batch_size}")

        # Train the model
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(self.x_test, self.y_test)
        )

        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        # Prepare metrics
        metrics = {
            "train_loss": float(history.history['loss'][-1]),
            "train_accuracy": float(history.history['accuracy'][-1]),
            "val_loss": float(history.history['val_loss'][-1]),
            "val_accuracy": float(history.history['val_accuracy'][-1]),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "data_size": len(self.x_train),
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

        # Get updated weights
        model_update = self._model_to_weights(self.model)

        return model_update, metrics

    def save_model(self, filename: str):
        """Save the model to a file"""
        if self.model:
            self.model.save(filename)
            logger.info(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """Load the model from a file"""
        if os.path.exists(filename):
            self.model = tf.keras.models.load_model(filename)
            logger.info(f"Model loaded from {filename}")
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description='Advanced Federated Learning Client (MNIST)')
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
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs per round')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory to save models')

    args = parser.parse_args()

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Create advanced client
    mnist_client = FederatedMNISTClient(
        data_size=args.data_size,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Create federated learning client
    client = FederatedClient(
        server_url=args.server,
        client_name=args.name or f"mnist-client-{int(time.time())}"
    )

    # Register with the server
    registration_successful = client.register(
        dataset_description={
            "type": "image",
            "dataset": "mnist",
            "classes": 10,
            "distribution": "balanced"
        },
        compute_capabilities={
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu": tf.test.is_gpu_available(),
            "framework": "tensorflow"
        },
        privacy_level="moderate",
        data_volume=args.data_size
    )

    if not registration_successful:
        logger.error("Registration failed, exiting")
        return

    logger.info(f"Registration successful! Client ID: {client.client_id}")

    # Define training function that will be called by the client
    def train_function(global_model):
        # Save model before training if it exists
        if global_model:
            model_file = os.path.join(args.model_dir, f"global_model_round_{client.current_round}.h5")
            with open(os.path.join(args.model_dir, f"global_model_round_{client.current_round}.json"), 'w') as f:
                json.dump(global_model, f)

        # Train the model
        model_update, metrics = mnist_client.train(global_model)

        # Save updated model
        model_file = os.path.join(args.model_dir, f"local_model_round_{client.current_round}.h5")
        mnist_client.save_model(model_file)

        return model_update, metrics

    # Participate in training rounds
    client.train_and_update(
        train_function=train_function,
        interval=args.interval,
        max_rounds=args.max_rounds
    )


if __name__ == "__main__":
    main()