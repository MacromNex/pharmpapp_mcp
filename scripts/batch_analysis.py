#!/usr/bin/env python3
"""
Script: batch_analysis.py
Description: Batch analysis and comparison of multiple permeability models for cyclic peptides

Original Use Case: examples/use_case_3_batch_analysis.py
Dependencies Removed: Matplotlib dependencies made optional

Usage:
    python scripts/batch_analysis.py --input <input_file> --output <output_dir>

Example:
    python scripts/batch_analysis.py --input examples/data/descriptors.csv --output results/batch_analysis/
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

# Import our predictor
from predict_permeability import PharmPappPermeabilityPredictor, MODEL_CONFIGS

# Optional plotting (graceful fallback)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "models_to_test": ['rrck_c', 'pampa_c', 'caco2_l', 'caco2_c', 'caco2_a'],
    "create_plots": True,
    "calculate_correlations": True,
    "save_individual_predictions": True,
    "test_size": 0.2,
    "random_state": 42
}

# ==============================================================================
# Core Class (extracted and simplified from use case)
# ==============================================================================
class CyclicPeptideBatchAnalyzer:
    """
    Batch analysis of multiple permeability models for cyclic peptides.

    Simplified from examples/use_case_3_batch_analysis.py
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize batch analyzer."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.models = {}

        if self.config['create_plots'] and not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available. Plots will be skipped.")
            self.config['create_plots'] = False

    def load_data(self, input_file: Union[str, Path]) -> pd.DataFrame:
        """Load input data for batch analysis."""
        input_file = Path(input_file)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read data
        if input_file.suffix.lower() == '.csv':
            data = pd.read_csv(input_file)
        elif input_file.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")

        self.logger.info(f"Loaded {len(data)} samples from {input_file}")
        return data

    def run_model_analysis(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Run analysis for a single model."""
        try:
            self.logger.info(f"Analyzing model: {model_name}")

            # Initialize predictor
            predictor = PharmPappPermeabilityPredictor(model_name, self.config)

            # Load and prepare data for this model
            model_data = predictor.load_data(data, has_target=True)

            results = {
                'model_name': model_name,
                'model_config': MODEL_CONFIGS[model_name],
                'success': True,
                'error': None,
                'predictions': None,
                'training_metrics': None,
                'features_used': None
            }

            # Train model if target data is available
            if model_data['target'] is not None:
                try:
                    training_metrics = predictor.train(model_data)
                    results['training_metrics'] = training_metrics
                    self.logger.info(f"Training metrics for {model_name}: R² = {training_metrics['test_r2']:.3f}")
                except Exception as e:
                    self.logger.warning(f"Training failed for {model_name}: {str(e)}")
                    results['training_metrics'] = {'error': str(e)}

            # Make predictions
            try:
                predictions = predictor.predict(model_data['features'])
                results['predictions'] = predictions
                results['features_used'] = len(predictor.feature_names) if predictor.feature_names else 0

                # Create results DataFrame
                pred_df = model_data['metadata'].copy()
                pred_df[f'{model_name}_prediction'] = predictions
                if model_data['target'] is not None:
                    pred_df['experimental'] = model_data['target']
                    pred_df['residual'] = model_data['target'] - predictions

                results['results_df'] = pred_df

            except Exception as e:
                self.logger.error(f"Prediction failed for {model_name}: {str(e)}")
                results['success'] = False
                results['error'] = str(e)

            return results

        except Exception as e:
            self.logger.error(f"Model analysis failed for {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e),
                'predictions': None,
                'training_metrics': None,
                'features_used': None
            }

    def analyze_all_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run analysis for all configured models."""
        models_to_test = self.config['models_to_test']
        all_results = {}

        for model_name in models_to_test:
            results = self.run_model_analysis(model_name, data)
            all_results[model_name] = results

        # Create summary statistics
        successful_models = [name for name, res in all_results.items() if res['success']]
        failed_models = [name for name, res in all_results.items() if not res['success']]

        summary = {
            'total_models': len(models_to_test),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'successful_model_names': successful_models,
            'failed_model_names': failed_models,
            'model_results': all_results
        }

        self.logger.info(f"Batch analysis complete: {len(successful_models)}/{len(models_to_test)} models successful")

        return summary

    def create_summary_statistics(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary statistics table."""
        summary_data = []

        for model_name, model_results in results['model_results'].items():
            if model_results['success'] and model_results['training_metrics']:
                metrics = model_results['training_metrics']
                if isinstance(metrics, dict) and 'error' not in metrics:
                    summary_data.append({
                        'Model': MODEL_CONFIGS[model_name]['name'],
                        'Algorithm': MODEL_CONFIGS[model_name]['algorithm'],
                        'Features_Used': model_results.get('features_used', 'N/A'),
                        'Train_R2': metrics.get('train_r2', 'N/A'),
                        'Test_R2': metrics.get('test_r2', 'N/A'),
                        'RMSE': metrics.get('test_rmse', 'N/A'),
                        'Samples': metrics.get('n_samples', 'N/A'),
                        'Status': 'Success'
                    })
                else:
                    summary_data.append({
                        'Model': MODEL_CONFIGS[model_name]['name'],
                        'Algorithm': MODEL_CONFIGS[model_name]['algorithm'],
                        'Features_Used': 'N/A',
                        'Train_R2': 'N/A',
                        'Test_R2': 'N/A',
                        'RMSE': 'N/A',
                        'Samples': 'N/A',
                        'Status': f"Training Failed: {metrics.get('error', 'Unknown')}"
                    })
            else:
                summary_data.append({
                    'Model': MODEL_CONFIGS[model_name]['name'],
                    'Algorithm': MODEL_CONFIGS[model_name]['algorithm'],
                    'Features_Used': 'N/A',
                    'Train_R2': 'N/A',
                    'Test_R2': 'N/A',
                    'RMSE': 'N/A',
                    'Samples': 'N/A',
                    'Status': f"Failed: {model_results.get('error', 'Unknown')}"
                })

        return pd.DataFrame(summary_data)

    def create_plots(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Create analysis plots."""
        if not self.config['create_plots'] or not PLOTTING_AVAILABLE:
            return []

        plot_files = []

        try:
            # Collect successful predictions for correlation analysis
            prediction_data = {}
            for model_name, model_results in results['model_results'].items():
                if model_results['success'] and model_results['predictions'] is not None:
                    prediction_data[MODEL_CONFIGS[model_name]['name']] = model_results['predictions']

            if len(prediction_data) < 2:
                self.logger.warning("Not enough successful models for correlation plots")
                return plot_files

            # Create correlation matrix
            pred_df = pd.DataFrame(prediction_data)
            correlation_matrix = pred_df.corr()

            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
            plt.title('Model Prediction Correlations')
            plt.tight_layout()

            corr_plot_path = output_dir / 'model_correlations.png'
            plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(corr_plot_path)

            # Plot prediction distributions
            plt.figure(figsize=(12, 8))
            for i, (model_name, predictions) in enumerate(prediction_data.items()):
                plt.subplot(2, 3, i + 1)
                plt.hist(predictions, bins=20, alpha=0.7, label=model_name)
                plt.xlabel('Predicted Permeability')
                plt.ylabel('Frequency')
                plt.title(f'{model_name} Predictions')
                plt.legend()

            plt.tight_layout()

            dist_plot_path = output_dir / 'prediction_distributions.png'
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(dist_plot_path)

            self.logger.info(f"Created {len(plot_files)} plots")

        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")

        return plot_files

    def save_results(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Save all analysis results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        # Save summary statistics
        summary_df = self.create_summary_statistics(results)
        summary_path = output_dir / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False)
        saved_files.append(summary_path)

        # Save individual model predictions
        if self.config['save_individual_predictions']:
            for model_name, model_results in results['model_results'].items():
                if model_results['success'] and 'results_df' in model_results:
                    pred_path = output_dir / f'{model_name}_predictions.csv'
                    model_results['results_df'].to_csv(pred_path, index=False)
                    saved_files.append(pred_path)

        # Save analysis metadata
        metadata = {
            'analysis_config': self.config,
            'model_summary': {
                'total_models': results['total_models'],
                'successful_models': results['successful_models'],
                'failed_models': results['failed_models'],
                'successful_model_names': results['successful_model_names'],
                'failed_model_names': results['failed_model_names']
            }
        }

        metadata_path = output_dir / 'analysis_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files.append(metadata_path)

        # Create plots
        plot_files = self.create_plots(results, output_dir)
        saved_files.extend(plot_files)

        self.logger.info(f"Saved {len(saved_files)} files to {output_dir}")

        return saved_files

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_analysis(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run batch analysis of multiple permeability models for cyclic peptides.

    Args:
        input_file: Path to input file (CSV with molecular descriptors)
        output_dir: Directory to save output files
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: Analysis results for all models
            - output_files: List of saved file paths
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_analysis("descriptors.csv", "results/batch/")
        >>> print(f"Analyzed {result['results']['total_models']} models")
    """
    # Setup
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize analyzer
    analyzer = CyclicPeptideBatchAnalyzer(config)

    # Load data
    data = analyzer.load_data(input_file)

    # Run analysis for all models
    analysis_results = analyzer.analyze_all_models(data)

    # Save results
    saved_files = analyzer.save_results(analysis_results, output_dir)

    return {
        "results": analysis_results,
        "output_files": [str(f) for f in saved_files],
        "metadata": {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "samples_analyzed": len(data),
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
    parser.add_argument('--input', '-i', required=True,
                       help='Input file path (CSV with molecular descriptors)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory path')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--models', nargs='+',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Models to analyze (default: all)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    if args.models:
        config['models_to_test'] = args.models
    if args.no_plots:
        config['create_plots'] = False

    try:
        # Run batch analysis
        result = run_batch_analysis(
            input_file=args.input,
            output_dir=args.output,
            config=config
        )

        # Print summary
        results = result['results']
        print(f"✅ Batch Analysis Complete")
        print(f"📊 Models analyzed: {results['total_models']}")
        print(f"✅ Successful: {results['successful_models']} ({results['successful_model_names']})")
        print(f"❌ Failed: {results['failed_models']} ({results['failed_model_names']})")
        print(f"📁 Output files: {len(result['output_files'])}")
        print(f"📂 Output directory: {result['metadata']['output_dir']}")

        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == '__main__':
    main()