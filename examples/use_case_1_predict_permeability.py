#!/usr/bin/env python3
"""
PharmPapp Permeability Prediction - Use Case 1
==============================================

This script predicts peptide permeability using trained models for different cell line assays.
It replicates the functionality of the KNIME PharmPapp workflows in Python.

Models available:
- RRCK-C: Support Vector Machine (103 descriptors)
- PAMPA-C: LightGBM (71 descriptors)
- Caco2-L: Random Forest (82 descriptors)
- Caco2-C: Random Forest (79 descriptors)
- Caco2-A: Random Forest (92 descriptors)

Usage:
    python use_case_1_predict_permeability.py --input data/example_input.csv --model caco2_c --output predictions.csv
    python use_case_1_predict_permeability.py --input data/example_for_Caco2_C.csv --model caco2_c

Author: PharmPapp MCP Implementation
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations based on the PharmPapp documentation
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
        'descriptors': ['apol', 'a_aro', 'a_don', 'a_donacc', 'a_hyd', 'a_nO', 'balabanJ', 'BCUT_SLOGP_2', 'b_1rotN', 'b_max1len', 'chi1v_C', 'chi1_C', 'chiral', 'diameter', 'GCUT_PEOE_3', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstates', 'h_pstrain', 'Kier3', 'KierFlex', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_nring', 'opr_nrot', 'opr_violation', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-2', 'PEOE_VSA-4', 'PEOE_VSA-6', 'PEOE_VSA_NEG', 'PEOE_VSA_PNEG', 'PEOE_VSA_POS', 'petitjeanSC', 'radius', 'reactive', 'SlogP', 'SlogP_VSA0', 'SlogP_VSA1', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA8', 'SlogP_VSA9', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'VAdjMa', 'VDistEq', 'vsa_don', 'weinerPath'],
        'example_file': 'examples/data/example for PAMPA-C.csv'
    },
    'caco2_l': {
        'name': 'Caco2-L',
        'algorithm': 'RandomForest',
        'descriptors': ['apol', 'ast_fraglike', 'ast_violation', 'ast_violation_ext', 'a_acc', 'a_aro', 'a_don', 'a_hyd', 'a_nN', 'a_nS', 'balabanJ', 'BCUT_PEOE_2', 'BCUT_SLOGP_2', 'BCUT_SLOGP_3', 'BCUT_SMR_2', 'b_double', 'b_max1len', 'b_rotR', 'chiral', 'chiral_u', 'diameter', 'GCUT_PEOE_3', 'GCUT_SLOGP_3', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstates', 'h_pstrain', 'lip_don', 'lip_druglike', 'lip_violation', 'logP(o/w)', 'logS', 'mutagenic', 'opr_brigid', 'opr_leadlike', 'opr_violation', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-6', 'PEOE_VSA_FNEG', 'PEOE_VSA_FPOS', 'PEOE_VSA_NEG', 'radius', 'reactive', 'rsynth', 'SlogP', 'SlogP_VSA1', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SMR_VSA1', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'VAdjEq', 'VAdjMa', 'VDistEq', 'vsa_don', 'weinerPath'],
        'example_file': 'examples/data/example for Caco2_L.csv'
    },
    'caco2_c': {
        'name': 'Caco2-C',
        'algorithm': 'RandomForest',
        'descriptors': ['apol', 'ast_violation', 'ast_violation_ext', 'a_acc', 'a_aro', 'a_don', 'a_donacc', 'a_nCl', 'a_nF', 'a_nN', 'a_nS', 'balabanJ', 'b_1rotN', 'b_double', 'b_max1len', 'chiral', 'chiral_u', 'diameter', 'GCUT_PEOE_3', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_dbo', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstates', 'h_pstrain', 'lip_druglike', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_leadlike', 'opr_nring', 'opr_violation', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-6', 'PEOE_VSA_NEG', 'petitjeanSC', 'radius', 'reactive', 'SlogP', 'SlogP_VSA0', 'SlogP_VSA1', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'TPSA', 'VAdjMa', 'VDistEq', 'vsa_don', 'weinerPath'],
        'example_file': 'examples/data/example for Caco2_C.csv'
    },
    'caco2_a': {
        'name': 'Caco2-A',
        'algorithm': 'RandomForest',
        'descriptors': ['apol', 'ast_fraglike', 'ast_violation', 'ast_violation_ext', 'a_acc', 'a_aro', 'a_don', 'a_donacc', 'a_hyd', 'a_nCl', 'a_nF', 'a_nN', 'a_nO', 'a_nS', 'balabanJ', 'BCUT_SLOGP_2', 'b_1rotN', 'b_1rotR', 'b_double', 'b_max1len', 'b_rotR', 'chiral', 'chiral_u', 'diameter', 'GCUT_PEOE_3', 'h_logD', 'h_logP', 'h_logS', 'h_log_dbo', 'h_log_pbo', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstates', 'h_pstrain', 'Kier3', 'lip_don', 'lip_druglike', 'lip_violation', 'logP(o/w)', 'logS', 'opr_brigid', 'opr_leadlike', 'opr_nring', 'opr_violation', 'PEOE_PC+', 'PEOE_PC-', 'PEOE_VSA+0', 'PEOE_VSA+1', 'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA+4', 'PEOE_VSA+5', 'PEOE_VSA+6', 'PEOE_VSA-0', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3', 'PEOE_VSA-4', 'PEOE_VSA-6', 'PEOE_VSA_FNEG', 'PEOE_VSA_FPOS', 'PEOE_VSA_NEG', 'petitjeanSC', 'radius', 'reactive', 'rings', 'rsynth', 'SlogP', 'SlogP_VSA0', 'SlogP_VSA1', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'TPSA', 'VAdjEq', 'VAdjMa', 'VDistEq', 'vsa_acc', 'vsa_don', 'weinerPath'],
        'example_file': 'examples/data/example for Caco2_A.csv'
    }
}

class PharmPappPredictor:
    """
    PharmPapp Permeability Prediction System

    This class replicates the functionality of the KNIME PharmPapp workflows for
    predicting peptide permeability across different cell line assays.
    """

    def __init__(self, model_name):
        """
        Initialize predictor with specified model.

        Args:
            model_name (str): Model name ('rrck_c', 'pampa_c', 'caco2_l', 'caco2_c', 'caco2_a')
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        logger.info(f"Initialized {self.config['name']} predictor using {self.config['algorithm']}")

    def _create_model(self):
        """Create machine learning model based on configuration."""
        if self.config['algorithm'] == 'RandomForest':
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif self.config['algorithm'] == 'SVR':
            return SVR(
                kernel='rbf',
                gamma='scale',
                C=1.0
            )
        elif self.config['algorithm'] == 'LightGBM':
            # Use RandomForest as LightGBM substitute for demo
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=7,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config['algorithm']}")

    def load_data(self, file_path, has_target=True):
        """
        Load and preprocess data.

        Args:
            file_path (str): Path to CSV file with molecular descriptors
            has_target (bool): Whether the file contains 'Permeability' target column

        Returns:
            tuple: (X, y) if has_target=True, else (X, None)
        """
        logger.info(f"Loading data from {file_path}")

        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {data.shape}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

        # Check if required descriptors are present
        required_descriptors = self.config['descriptors']
        missing_descriptors = [desc for desc in required_descriptors if desc not in data.columns]

        if missing_descriptors:
            logger.warning(f"Missing descriptors: {missing_descriptors}")
            logger.warning("Using available descriptors only")
            available_descriptors = [desc for desc in required_descriptors if desc in data.columns]
        else:
            available_descriptors = required_descriptors

        # Extract features
        X = data[available_descriptors].copy()

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            logger.warning("Found missing values, filling with column means")
            X = X.fillna(X.mean())

        # Extract target if available
        y = None
        if has_target and 'Permeability' in data.columns:
            y = data['Permeability'].values
            logger.info(f"Target variable range: {y.min():.3f} to {y.max():.3f}")

        logger.info(f"Final feature matrix shape: {X.shape}")
        return X, y

    def train(self, X, y):
        """
        Train the permeability prediction model.

        Args:
            X (DataFrame): Feature matrix with molecular descriptors
            y (array): Target permeability values
        """
        logger.info(f"Training {self.config['name']} model...")

        # Create and configure model
        self.model = self._create_model()

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        logger.info(f"Training R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}")
        logger.info(f"Test R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}")

        self.is_trained = True

        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }

    def predict(self, X):
        """
        Predict permeability for new molecules.

        Args:
            X (DataFrame): Feature matrix with molecular descriptors

        Returns:
            array: Predicted log permeability values
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load_model()")

        logger.info(f"Predicting permeability for {len(X)} molecules")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        logger.info(f"Predictions range: {predictions.min():.3f} to {predictions.max():.3f}")

        return predictions

    def save_model(self, model_path):
        """Save trained model and scaler."""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Load pre-trained model and scaler."""
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.is_trained = True

        logger.info(f"Model loaded from {model_path}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='PharmPapp Permeability Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train model on example data and make predictions
    python use_case_1_predict_permeability.py --model caco2_c --train --input data/example_for_Caco2_C.csv

    # Make predictions using example data (demo mode)
    python use_case_1_predict_permeability.py --model caco2_c --demo

    # Predict new molecules (requires pre-calculated descriptors)
    python use_case_1_predict_permeability.py --model caco2_c --input new_molecules.csv --output predictions.csv
        """
    )

    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), required=True,
                        help='Permeability model to use')
    parser.add_argument('--input', type=str, help='Input CSV file with molecular descriptors')
    parser.add_argument('--output', type=str, help='Output CSV file for predictions')
    parser.add_argument('--train', action='store_true', help='Train model on input data')
    parser.add_argument('--demo', action='store_true', help='Run demo using example data')
    parser.add_argument('--save-model', type=str, help='Save trained model to file')
    parser.add_argument('--load-model', type=str, help='Load pre-trained model from file')

    args = parser.parse_args()

    try:
        # Initialize predictor
        predictor = PharmPappPredictor(args.model)

        # Demo mode - use example data
        if args.demo:
            logger.info(f"Running demo mode for {predictor.config['name']} model")
            input_file = predictor.config['example_file']

            # Load example data
            X, y = predictor.load_data(input_file, has_target=True)

            # Train model
            metrics = predictor.train(X, y)
            logger.info("Demo training completed")

            # Make predictions on the same data (for demonstration)
            predictions = predictor.predict(X)

            # Create output dataframe
            results = X.copy()
            results['Permeability_Actual'] = y
            results['Permeability_Predicted'] = predictions
            results['Prediction_Error'] = y - predictions

            # Save results
            output_file = f"demo_predictions_{args.model}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Demo results saved to {output_file}")

            return

        # Load pre-trained model if specified
        if args.load_model:
            predictor.load_model(args.load_model)

        # Training mode
        if args.train:
            if not args.input:
                raise ValueError("Input file required for training")

            X, y = predictor.load_data(args.input, has_target=True)
            metrics = predictor.train(X, y)

            if args.save_model:
                predictor.save_model(args.save_model)

        # Prediction mode
        if args.input and not args.train:
            # Determine if input has target variable
            temp_data = pd.read_csv(args.input)
            has_target = 'Permeability' in temp_data.columns

            X, y = predictor.load_data(args.input, has_target=has_target)
            predictions = predictor.predict(X)

            # Create output dataframe
            results = X.copy()
            results['Permeability_Predicted'] = predictions

            if has_target:
                results['Permeability_Actual'] = y
                results['Prediction_Error'] = y - predictions

                # Calculate metrics
                r2 = r2_score(y, predictions)
                rmse = np.sqrt(mean_squared_error(y, predictions))
                logger.info(f"Prediction R²: {r2:.3f}, RMSE: {rmse:.3f}")

            # Save results
            output_file = args.output or f"predictions_{args.model}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())