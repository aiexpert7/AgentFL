import os
import json
import datetime
import uuid
import logging
from typing import Dict, List, Optional, Any


class StrategicAgent:
    """Agent that provides strategic planning and synthetic data generation"""

    def __init__(self):
        self.logger = self._setup_logger()
        self.strategies = []
        self.synthetic_datasets = {}

        # LLM configuration
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.warning("OpenAI API key not found, LLM features will be limited")

        # Data storage
        self.data_dir = os.environ.get("DATA_DIR", "./data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing data if available
        self._load_data()

        self.logger.info("StrategicAgent initialized")

    def _setup_logger(self):
        """Set up a logger with the given name"""
        logger = logging.getLogger("StrategicAgent")
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
                    {"role": "system", "content": "You are a strategic planning agent for a federated learning system."},
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
        """Load existing strategy data if available"""
        strategies_file = os.path.join(self.data_dir, "strategies.json")
        datasets_file = os.path.join(self.data_dir, "synthetic_datasets.json")

        if os.path.exists(strategies_file):
            try:
                with open(strategies_file, 'r') as f:
                    self.strategies = json.load(f)
                self.logger.info(f"Loaded {len(self.strategies)} existing strategies")
            except Exception as e:
                self.logger.error(f"Error loading strategies data: {e}")

        if os.path.exists(datasets_file):
            try:
                with open(datasets_file, 'r') as f:
                    self.synthetic_datasets = json.load(f)
                self.logger.info(f"Loaded {len(self.synthetic_datasets)} synthetic datasets")
            except Exception as e:
                self.logger.error(f"Error loading synthetic datasets data: {e}")

    def _save_data(self):
        """Save current strategy data"""
        strategies_file = os.path.join(self.data_dir, "strategies.json")
        datasets_file = os.path.join(self.data_dir, "synthetic_datasets.json")

        try:
            with open(strategies_file, 'w') as f:
                json.dump(self.strategies, f, default=str)

            with open(datasets_file, 'w') as f:
                json.dump(self.synthetic_datasets, f, default=str)
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def create_improvement_strategy(self, model_performance: Dict[str, Any],
                                    client_metrics: Dict[str, Any],
                                    current_round: int) -> Dict[str, Any]:
        """Create a strategy to improve model performance"""
        prompt = f"""
        Create a strategic plan to improve federated learning model performance.
        
        Current state:
        - Training round: {current_round}
        - Model performance: {json.dumps(model_performance, indent=2)}
        - Client metrics: {json.dumps(client_metrics, indent=2)}
        
        Provide a strategic plan that includes:
        1. Identified weaknesses or bottlenecks
        2. Recommended adjustments to aggregation algorithm
        3. Client selection strategy for next round
        4. Potential areas for synthetic data augmentation
        5. Hyperparameter tuning recommendations
        
        Return your plan as structured JSON with these fields:
        - weaknesses: array of strings
        - aggregation_recommendations: object with key-value pairs
        - client_selection_strategy: object with key-value pairs
        - synthetic_data_recommendations: object with key-value pairs
        - hyperparameter_recommendations: object with key-value pairs
        """

        response = self._call_llm(prompt)

        try:
            strategy_data = json.loads(response)
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            strategy_data = {
                "weaknesses": ["Unable to parse strategy"],
                "aggregation_recommendations": {},
                "client_selection_strategy": {},
                "synthetic_data_recommendations": {},
                "hyperparameter_recommendations": {}
            }

        # Create Strategy object
        strategy = {
            "strategy_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "training_round": current_round,
            "weaknesses": strategy_data.get("weaknesses", []),
            "aggregation_recommendations": strategy_data.get("aggregation_recommendations", {}),
            "client_selection_strategy": strategy_data.get("client_selection_strategy", {}),
            "synthetic_data_recommendations": strategy_data.get("synthetic_data_recommendations", {}),
            "hyperparameter_recommendations": strategy_data.get("hyperparameter_recommendations", {})
        }

        self.strategies.append(strategy)
        self.logger.info(f"Created improvement strategy {strategy['strategy_id']} for round {current_round}")

        # Save updated data
        self._save_data()

        return strategy

    def generate_synthetic_data(self, data_description: Dict[str, Any], target_size: int) -> Dict[str, Any]:
        """Generate synthetic data based on description"""
        prompt = f"""
        Generate a blueprint for synthetic data generation:
        
        Data description:
        {json.dumps(data_description, indent=2)}
        
        Target size: {target_size} samples
        
        Provide a detailed blueprint for generating this synthetic data, including:
        1. Distribution parameters for each feature
        2. Correlation structure between features
        3. Class balance considerations
        4. Potential noise injection approaches
        5. Quality verification methods
        
        Return your blueprint as structured JSON with a field called 'generation_parameters' that contains:
        - distributions: object with parameters for each feature
        - correlations: object describing correlations between features
        - class_balance: object with class balance parameters
        - noise_parameters: object with noise injection parameters
        - quality_verification: array of verification methods
        """

        response = self._call_llm(prompt)

        try:
            blueprint = json.loads(response)
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            blueprint = {
                "generation_parameters": {
                    "distributions": {},
                    "correlations": {},
                    "class_balance": {},
                    "noise_parameters": {}
                }
            }

        # Create SyntheticDataset object
        dataset = {
            "dataset_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "description": data_description,
            "generation_parameters": blueprint.get("generation_parameters", {}),
            "size": target_size,
            "status": "created"
        }

        # Store the dataset
        self.synthetic_datasets[dataset["dataset_id"]] = dataset

        self.logger.info(f"Generated synthetic dataset {dataset['dataset_id']} with {target_size} samples")

        # Save updated data
        self._save_data()

        return dataset

    def get_synthetic_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve information about a synthetic dataset"""
        return self.synthetic_datasets.get(dataset_id)

    def get_best_strategy(self, current_round: int) -> Optional[Dict[str, Any]]:
        """Get the best strategy for the current training round"""
        relevant_strategies = [s for s in self.strategies if s.get("training_round", 0) <= current_round]
        if not relevant_strategies:
            return None

        # In a real implementation, this would evaluate strategies more carefully
        return max(relevant_strategies, key=lambda s: s.get("training_round", 0))

    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all strategies"""
        return self.strategies


