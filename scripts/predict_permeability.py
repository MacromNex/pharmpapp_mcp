#!/usr/bin/env python3
"""
Script: predict_permeability.py
Description: Predict cyclic peptide permeability using trained models

Original Use Case: examples/use_case_1_predict_permeability.py
Dependencies Removed: Simplified model configurations

Usage:
    python scripts/predict_permeability.py --input <input_file> --model <model_name> --output <output_file>

Example:
    python scripts/predict_permeability.py --input examples/data/descriptors.csv --model caco2_c --output results/predictions.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import logging
import warnings

# Essential scientific packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_jobs": -1,
    "cross_validation": True,
    "scale_features": True
}

# Model configurations based on PharmPapp documentation
MODEL_CONFIGS = {
    'rrck_c': {
        'name': 'RRCK-C',
        'algorithm': 'SVR',
        'descriptors': ['apol', 'ast_violation', 'a_acc', 'a_aro', 'a_don', 'a_donacc', 'a_heavy', 'a_hyd', 'a_IC', 'a_nC', 'a_nF', 'a_nH', 'a_nN', 'a_nO', 'a_nS', 'balabanJ', 'BCUT_PEOE_1', 'BCUT_PEOE_2', 'BCUT_SLOGP_1', 'BCUT_SLOGP_2', 'BCUT_SMR_1', 'bpol', 'b_1rotN', 'b_double', 'b_max1len', 'chi1v', 'chi1v_C', 'chi1_C', 'chiral', 'chiral_u', 'diameter', 'h_ema', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstrain', 'Kier2', 'Kier3', 'KierA2', 'KierFlex', 'lip_acc', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_nring', 'opr_nrot', 'opr_violation', 'PEOE_PC+', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-5', 'PEOE_VSA-6', 'PEOE_VSA_HYD', 'PEOE_VSA_NEG', 'PEOE_VSA_PNEG', 'PEOE_VSA_POL', 'PEOE_VSA_POS', 'PEOE_VSA_PPOS', 'petitjeanSC', 'radius', 'reactive', 'rsynth', 'SlogP', 'SlogP_VSA0', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'TPSA', 'VDistEq', 'vsa_don', 'weinerPath', 'weinerPol'],
        'example_file': 'examples/data/example for RRCK-C.csv'
    },
    'pampa_c': {
        'name': 'PAMPA-C',
        'algorithm': 'LightGBM',
        'descriptors': ['a_acc', 'a_aro', 'a_don', 'a_donacc', 'a_heavy', 'a_hyd', 'a_IC', 'a_nC', 'a_nF', 'a_nH', 'a_nN', 'a_nO', 'a_nS', 'apol', 'ast_violation', 'balabanJ', 'BCUT_PEOE_1', 'BCUT_PEOE_2', 'BCUT_SLOGP_1', 'BCUT_SLOGP_2', 'bpol', 'b_1rotN', 'b_double', 'chi1v', 'chi1v_C', 'chi1_C', 'chiral', 'chiral_u', 'diameter', 'GCUT_PEOE_3', 'h_ema', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstates', 'Kier2', 'Kier3', 'KierA2', 'KierFlex', 'lip_acc', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_nring', 'opr_nrot', 'opr_violation', 'PEOE_PC+', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-5', 'PEOE_VSA-6', 'petitjeanSC', 'radius', 'reactive', 'rsynth', 'VAdjMa', 'VDistEq', 'vsa_don', 'weinerPath', 'weinerPol'],
        'example_file': 'examples/data/example for PAMPA-C.csv'
    },
    'caco2_l': {
        'name': 'Caco2-L',
        'algorithm': 'RandomForest',
        'descriptors': ['a_acc', 'a_aro', 'a_don', 'a_donacc', 'a_heavy', 'a_hyd', 'a_IC', 'a_nC', 'a_nF', 'a_nH', 'a_nN', 'a_nO', 'a_nS', 'apol', 'ast_violation', 'balabanJ', 'BCUT_PEOE_1', 'BCUT_PEOE_2', 'BCUT_SLOGP_1', 'BCUT_SLOGP_2', 'bpol', 'b_1rotN', 'b_double', 'b_max1len', 'chi1v', 'chi1v_C', 'chi1_C', 'chiral', 'chiral_u', 'diameter', 'h_ema', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstrain', 'Kier2', 'Kier3', 'KierA2', 'KierFlex', 'lip_acc', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_nring', 'opr_nrot', 'opr_violation', 'PEOE_PC+', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-5', 'PEOE_VSA-6', 'PEOE_VSA_HYD', 'PEOE_VSA_NEG', 'PEOE_VSA_PNEG', 'PEOE_VSA_POL', 'PEOE_VSA_POS', 'PEOE_VSA_PPOS', 'petitjeanSC', 'radius', 'reactive', 'rsynth', 'SlogP', 'SlogP_VSA0', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'VDistEq', 'vsa_don', 'weinerPath', 'weinerPol'],
        'example_file': 'examples/data/example for Caco2_L.csv'
    },
    'caco2_c': {
        'name': 'Caco2-C',
        'algorithm': 'RandomForest',
        'descriptors': ['a_acc', 'a_aro', 'a_don', 'a_donacc', 'a_heavy', 'a_hyd', 'a_IC', 'a_nC', 'a_nF', 'a_nH', 'a_nN', 'a_nO', 'a_nS', 'apol', 'ast_violation', 'balabanJ', 'BCUT_PEOE_1', 'BCUT_PEOE_2', 'BCUT_SLOGP_1', 'BCUT_SLOGP_2', 'bpol', 'b_1rotN', 'b_double', 'b_max1len', 'chi1v', 'chi1v_C', 'chi1_C', 'chiral', 'chiral_u', 'diameter', 'h_ema', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstrain', 'Kier2', 'Kier3', 'KierA2', 'KierFlex', 'lip_acc', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_nring', 'opr_nrot', 'opr_violation', 'PEOE_PC+', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-5', 'PEOE_VSA-6', 'PEOE_VSA_HYD', 'PEOE_VSA_NEG', 'PEOE_VSA_PNEG', 'PEOE_VSA_POL', 'PEOE_VSA_POS', 'PEOE_VSA_PPOS', 'petitjeanSC', 'radius', 'reactive', 'rsynth', 'SlogP', 'SlogP_VSA0', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'VDistEq', 'vsa_don', 'weinerPath', 'weinerPol'],
        'example_file': 'examples/data/example for Caco2_C.csv'
    },
    'caco2_a': {
        'name': 'Caco2-A',
        'algorithm': 'RandomForest',
        'descriptors': ['a_acc', 'a_aro', 'a_don', 'a_donacc', 'a_heavy', 'a_hyd', 'a_IC', 'a_nC', 'a_nF', 'a_nH', 'a_nN', 'a_nO', 'a_nS', 'apol', 'ast_violation', 'balabanJ', 'BCUT_PEOE_1', 'BCUT_PEOE_2', 'BCUT_SLOGP_1', 'BCUT_SLOGP_2', 'BCUT_SMR_1', 'bpol', 'b_1rotN', 'b_double', 'b_max1len', 'chi1v', 'chi1v_C', 'chi1_C', 'chiral', 'chiral_u', 'diameter', 'h_ema', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstrain', 'Kier2', 'Kier3', 'KierA2', 'KierFlex', 'lip_acc', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_nring', 'opr_nrot', 'opr_violation', 'PEOE_PC+', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-5', 'PEOE_VSA-6', 'PEOE_VSA_HYD', 'PEOE_VSA_NEG', 'PEOE_VSA_PNEG', 'PEOE_VSA_POL', 'PEOE_VSA_POS', 'PEOE_VSA_PPOS', 'petitjeanSC', 'radius', 'reactive', 'rsynth', 'SlogP', 'SlogP_VSA0', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'TPSA', 'VDistEq', 'vsa_don', 'weinerPath', 'weinerPol'],
        'example_file': 'examples/data/example for Caco2_A.csv'
    }
}

# ==============================================================================
# Core Class (extracted and simplified from use case)
# ==============================================================================
class PharmPappPermeabilityPredictor:
    """
    Predict cyclic peptide permeability using PharmPapp models.

    Simplified from examples/use_case_1_predict_permeability.py
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize predictor for specified model."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not supported. Available: {list(MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.scaler = None
        self.feature_names = None

        self._create_model()

    def _create_model(self):
        """Create model based on algorithm type."""
        algorithm = self.model_config['algorithm']

        if algorithm == 'RandomForest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs']
            )
        elif algorithm == 'SVR':
            self.model = SVR(
                kernel='rbf',
                gamma='scale'
            )
        elif algorithm == 'LightGBM':
            # Use RandomForest as fallback since LightGBM requires additional installation
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs']
            )
            self.logger.info("Using RandomForest instead of LightGBM for compatibility")
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Initialize scaler if needed
        if self.config['scale_features']:
            self.scaler = StandardScaler()

        self.logger.info(f"Initialized {self.model_config['name']} model with {algorithm}")

    def load_data(self, file_path: Union[str, Path, pd.DataFrame], has_target: bool = True) -> Dict[str, Any]:
        """Load data from file or DataFrame."""
        if isinstance(file_path, pd.DataFrame):
            data = file_path.copy()
            self.logger.info(f"Using provided DataFrame with {len(data)} samples")
        else:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Read data
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            self.logger.info(f"Loaded {len(data)} samples from {file_path}")

        # Identify target column
        target_col = None
        if has_target:
            target_columns = ['Experimental', 'target', 'y', 'permeability', 'logPe']
            for col in target_columns:
                if col in data.columns:
                    target_col = col
                    break

            if target_col is None:
                self.logger.warning("No target column found. Assuming prediction mode.")
                has_target = False

        # Get model-specific descriptors
        required_descriptors = self.model_config['descriptors']
        available_descriptors = [col for col in required_descriptors if col in data.columns]
        missing_descriptors = [col for col in required_descriptors if col not in data.columns]

        if missing_descriptors:
            self.logger.warning(f"Missing {len(missing_descriptors)} descriptors: {missing_descriptors[:5]}{'...' if len(missing_descriptors) > 5 else ''}")

        if len(available_descriptors) < len(required_descriptors) * 0.8:
            raise ValueError(f"Too many missing descriptors. Available: {len(available_descriptors)}/{len(required_descriptors)}")

        # Use available descriptors
        self.feature_names = available_descriptors

        result_data = {
            'features': data[available_descriptors],
            'target': data[target_col] if has_target and target_col else None,
            'metadata': data[[col for col in data.columns if col not in available_descriptors + ([target_col] if target_col else [])]]
        }

        return result_data

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for training/prediction."""
        # Handle missing values
        X = data.fillna(data.mean())

        # Scale features if configured
        if self.scaler is not None:
            if hasattr(self.scaler, 'scale_'):  # Already fitted
                X = self.scaler.transform(X)
            else:  # Not fitted yet
                X = self.scaler.fit_transform(X)

        return X.values

    def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model."""
        if data['target'] is None:
            raise ValueError("Training data must include target values")

        X = self.prepare_features(data['features'])
        y = data['target'].values

        # Split data if cross-validation is enabled
        if self.config['cross_validation'] and len(X) > 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        self.logger.info(f"Training completed - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")

        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = self.prepare_features(data)
        predictions = self.model.predict(X)

        self.logger.info(f"Generated {len(predictions)} predictions")

        return predictions

    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save trained model."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_config': self.model_config,
            'config': self.config
        }

        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load trained model."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        self.logger.info(f"Model loaded from {model_path}")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_permeability(
    input_file: Union[str, Path],
    model_name: str,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    demo: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict cyclic peptide permeability using PharmPapp models.

    Args:
        input_file: Path to input file (CSV with descriptors)
        model_name: Model to use ('rrck_c', 'pampa_c', 'caco2_l', 'caco2_c', 'caco2_a')
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        demo: Use demo mode with example data
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: DataFrame with predictions
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata including model performance

    Example:
        >>> result = run_predict_permeability("descriptors.csv", "caco2_c", "predictions.csv")
        >>> print(f"Predicted {len(result['result'])} samples")
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Handle demo mode
    if demo:
        example_file = MODEL_CONFIGS[model_name]['example_file']
        input_file = Path(example_file)
        if not output_file:
            output_file = f"demo_predictions_{model_name}.csv"

    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Initialize predictor
    predictor = PharmPappPermeabilityPredictor(model_name, config)

    # Load data
    data = predictor.load_data(input_file, has_target=True)

    # Train model if target data is available
    training_metrics = None
    if data['target'] is not None:
        training_metrics = predictor.train(data)
    else:
        logger.warning("No target data found. Using pre-configured model parameters.")

    # Make predictions
    predictions = predictor.predict(data['features'])

    # Create results DataFrame
    results_df = data['metadata'].copy()
    results_df[f'{model_name}_prediction'] = predictions

    if data['target'] is not None:
        results_df['experimental'] = data['target']
        results_df['residual'] = data['target'] - predictions

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    return {
        "result": results_df,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "model_name": model_name,
            "model_config": predictor.model_config,
            "samples_predicted": len(results_df),
            "features_used": len(predictor.feature_names) if predictor.feature_names else 0,
            "training_metrics": training_metrics,
            "config": config
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i',
                       help='Input file path (CSV with molecular descriptors)')
    parser.add_argument('--model', '-m',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to use for prediction')
    parser.add_argument('--output', '-o',
                       help='Output file path (CSV)')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo data')

    args = parser.parse_args()

    # Validate arguments
    if not args.demo:
        if not args.input:
            parser.error("--input is required unless --demo is used")
        if not args.model:
            parser.error("--model is required unless --demo is used")

    # Set defaults for demo mode
    if args.demo and not args.model:
        args.model = 'caco2_c'  # Default model for demo

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    try:
        # Run prediction
        result = run_predict_permeability(
            input_file=args.input,
            model_name=args.model,
            output_file=args.output,
            config=config,
            demo=args.demo
        )

        # Print summary
        print(f"✅ Success: Model {result['metadata']['model_name']} ({MODEL_CONFIGS[args.model]['name']})")
        print(f"🔬 Predicted {result['metadata']['samples_predicted']} samples")
        print(f"📊 Used {result['metadata']['features_used']} features")

        if result['metadata']['training_metrics']:
            metrics = result['metadata']['training_metrics']
            print(f"📈 Training R²: {metrics['train_r2']:.3f}, Test R²: {metrics['test_r2']:.3f}")
            print(f"📏 RMSE: {metrics['test_rmse']:.3f}")

        if result['output_file']:
            print(f"💾 Output saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == '__main__':
    main()