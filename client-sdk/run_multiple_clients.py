#!/usr/bin/env python
# run_multiple_clients.py
import argparse
import subprocess
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClientLauncher")

# Global list to keep track of running processes
processes = []

def run_client(client_config):
    """Run a single client with the provided configuration"""
    cmd = [
        "python", "mnist_federated_client.py",
        "--server", client_config["server"],
        "--name", client_config["name"],
        "--heterogeneity", client_config["heterogeneity"],
        "--privacy", client_config["privacy"],
        "--split", str(client_config["split"]),
        "--interval", str(client_config["interval"]),
        "--model-dir", os.path.join(client_config["base_model_dir"], client_config["name"])
    ]

    if client_config.get("rounds") is not None:
        cmd.extend(["--rounds", str(client_config["rounds"])])

    # Create model directory
    os.makedirs(os.path.join(client_config["base_model_dir"], client_config["name"]), exist_ok=True)

    # Create log file
    log_file = open(os.path.join(client_config["log_dir"], f"{client_config['name']}.log"), 'w')

    logger.info(f"Starting client: {client_config['name']}")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Add to global processes list
    processes.append((process, log_file, client_config["name"]))

    return process

def create_client_configs(args):
    """Create configurations for all clients to run"""
    client_configs = []

    heterogeneity_levels = args.heterogeneity.split(',')
    privacy_levels = args.privacy.split(',')

    # Create configs for all specified clients
    for i in range(1, args.num_clients + 1):
        # Distribute heterogeneity and privacy levels
        heterogeneity = heterogeneity_levels[i % len(heterogeneity_levels)]
        privacy = privacy_levels[i % len(privacy_levels)]

        # Create client config
        client_config = {
            "server": args.server,
            "name": f"client-{i:02d}-{heterogeneity[0]}{privacy[0]}",
            "heterogeneity": heterogeneity,
            "privacy": privacy,
            "split": i % 10 + 1,  # Cycle through splits 1-10
            "rounds": args.rounds,
            "interval": args.interval,
            "base_model_dir": args.model_dir,
            "log_dir": args.log_dir
        }

        client_configs.append(client_config)

    return client_configs

def signal_handler(sig, frame):
    """Handle Ctrl+C by terminating all client processes"""
    logger.info("Terminating all clients...")

    for process, log_file, name in processes:
        logger.info(f"Terminating client: {name}")
        process.terminate()

    # Wait for all processes to terminate
    for process, log_file, name in processes:
        process.wait()
        log_file.close()

    logger.info("All clients terminated")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Run multiple MNIST federated learning clients')
    parser.add_argument('--server', type=str, default='http://localhost:8001',
                        help='URL of the federated learning server')
    parser.add_argument('--num-clients', type=int, default=5,
                        help='Number of clients to run')
    parser.add_argument('--heterogeneity', type=str, default='low,medium,high',
                        help='Comma-separated list of heterogeneity levels to use')
    parser.add_argument('--privacy', type=str, default='low,moderate,high',
                        help='Comma-separated list of privacy levels to use')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Maximum number of rounds for each client')
    parser.add_argument('--interval', type=int, default=30,
                        help='Interval in seconds between checking for new tasks')
    parser.add_argument('--model-dir', type=str, default='./client_models',
                        help='Base directory to save models')
    parser.add_argument('--log-dir', type=str, default='./client_logs',
                        help='Directory to save client logs')
    parser.add_argument('--max-parallel', type=int, default=None,
                        help='Maximum number of clients to run in parallel (default: all)')

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create client configurations
    client_configs = create_client_configs(args)

    # Set up signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)

    # Set maximum parallel clients
    max_workers = args.max_parallel or args.num_clients

    logger.info(f"Starting {args.num_clients} clients with max {max_workers} in parallel")

    # Start clients in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_client, config) for config in client_configs]

        # Wait for all clients to finish
        for future in futures:
            future.result()

    # Close all log files
    for _, log_file, _ in processes:
        log_file.close()

    logger.info("All clients completed")

if __name__ == "__main__":
    main()