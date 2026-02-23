#!/usr/bin/env python3
"""
PharmPapp Batch Analysis - Use Case 3
=====================================

This script performs batch analysis across multiple PharmPapp models for comprehensive
permeability prediction and model comparison.

It allows users to:
1. Compare predictions across all 5 PharmPapp models
2. Identify consensus predictions
3. Analyze prediction uncertainty
4. Generate comprehensive reports

Usage:
    python use_case_3_batch_analysis.py --input data/molecules.csv --output results/
    python use_case_3_batch_analysis.py --demo

Author: PharmPapp MCP Implementation
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from use_case_1_predict_permeability import PharmPappPredictor, MODEL_CONFIGS
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PharmPappBatchAnalyzer:
    """
    Batch analysis tool for PharmPapp permeability predictions.

    This class provides comprehensive analysis across all PharmPapp models,
    enabling comparison of predictions and identification of consensus results.
    """

    def __init__(self, output_dir="."):
        """
        Initialize batch analyzer.

        Args:
            output_dir (str): Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.predictors = {}
        self.models_trained = {}

        logger.info("Initialized PharmPapp Batch Analyzer")

    def train_all_models(self):
        """Train all PharmPapp models using their respective example data."""
        logger.info("Training all PharmPapp models...")

        for model_name, config in MODEL_CONFIGS.items():
            try:
                logger.info(f"Training {config['name']} model...")

                # Initialize predictor
                predictor = PharmPappPredictor(model_name)

                # Load training data
                X, y = predictor.load_data(config['example_file'], has_target=True)

                # Train model
                metrics = predictor.train(X, y)

                # Store predictor and metrics
                self.predictors[model_name] = predictor
                self.models_trained[model_name] = metrics

                logger.info(f"  {config['name']} - R²: {metrics['test_r2']:.3f}, RMSE: {metrics['test_rmse']:.3f}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")

        logger.info(f"Successfully trained {len(self.predictors)} models")

    def predict_all_models(self, input_data):
        """
        Make predictions using all trained models.

        Args:
            input_data (DataFrame): Input data with molecular descriptors

        Returns:
            DataFrame: Predictions from all models
        """
        logger.info("Making predictions with all models...")

        results = input_data.copy() if input_data is not None else pd.DataFrame()
        predictions = {}

        for model_name, predictor in self.predictors.items():
            try:
                # Get model-specific descriptors
                model_descriptors = MODEL_CONFIGS[model_name]['descriptors']
                available_descriptors = [desc for desc in model_descriptors if desc in input_data.columns]

                if len(available_descriptors) < len(model_descriptors) * 0.5:
                    logger.warning(f"Insufficient descriptors for {model_name} (have {len(available_descriptors)}/{len(model_descriptors)})")
                    continue

                # Extract relevant features
                X_model = input_data[available_descriptors].fillna(input_data[available_descriptors].mean())

                # Make predictions
                pred = predictor.predict(X_model)
                predictions[model_name] = pred

                logger.info(f"  {MODEL_CONFIGS[model_name]['name']}: {len(pred)} predictions")

            except Exception as e:
                logger.error(f"Failed prediction with {model_name}: {str(e)}")

        # Combine predictions
        pred_df = pd.DataFrame(predictions)

        return results, pred_df

    def analyze_consensus(self, predictions_df):
        """
        Analyze consensus across model predictions.

        Args:
            predictions_df (DataFrame): Predictions from multiple models

        Returns:
            DataFrame: Consensus analysis results
        """
        logger.info("Analyzing prediction consensus...")

        results = pd.DataFrame()

        if predictions_df.empty:
            return results

        # Calculate consensus metrics
        results['mean_prediction'] = predictions_df.mean(axis=1)
        results['median_prediction'] = predictions_df.median(axis=1)
        results['std_prediction'] = predictions_df.std(axis=1)
        results['min_prediction'] = predictions_df.min(axis=1)
        results['max_prediction'] = predictions_df.max(axis=1)
        results['range_prediction'] = results['max_prediction'] - results['min_prediction']

        # Calculate agreement metrics
        results['cv_prediction'] = results['std_prediction'] / abs(results['mean_prediction'] + 1e-6)  # Coefficient of variation
        results['agreement_level'] = pd.cut(results['cv_prediction'],
                                          bins=[0, 0.1, 0.2, 0.5, float('inf')],
                                          labels=['High', 'Medium', 'Low', 'Very Low'])

        # Count available predictions per molecule
        results['n_predictions'] = predictions_df.notna().sum(axis=1)

        # Confidence score (inverse of CV, scaled 0-1)
        results['confidence_score'] = 1 / (1 + results['cv_prediction'])

        logger.info(f"Consensus analysis completed for {len(results)} molecules")

        return results

    def generate_report(self, input_data, predictions_df, consensus_df, actual_values=None):
        """
        Generate comprehensive analysis report.

        Args:
            input_data (DataFrame): Original input data
            predictions_df (DataFrame): Model predictions
            consensus_df (DataFrame): Consensus analysis
            actual_values (array, optional): Actual permeability values for validation
        """
        logger.info("Generating comprehensive report...")

        # Create report directory
        report_dir = self.output_dir / "report"
        report_dir.mkdir(exist_ok=True)

        # 1. Summary statistics
        self._generate_summary_stats(predictions_df, consensus_df, report_dir)

        # 2. Model comparison plots
        self._generate_comparison_plots(predictions_df, consensus_df, report_dir)

        # 3. Consensus analysis plots
        self._generate_consensus_plots(consensus_df, report_dir)

        # 4. Model performance comparison (if actual values available)
        if actual_values is not None:
            self._generate_performance_plots(predictions_df, actual_values, report_dir)

        # 5. Generate text report
        self._generate_text_report(input_data, predictions_df, consensus_df, actual_values, report_dir)

        logger.info(f"Report generated in {report_dir}")

    def _generate_summary_stats(self, predictions_df, consensus_df, report_dir):
        """Generate summary statistics."""
        summary = {}

        # Model availability
        summary['n_molecules'] = len(predictions_df)
        summary['models_available'] = list(predictions_df.columns)
        summary['n_models'] = len(summary['models_available'])

        # Prediction statistics
        for model in predictions_df.columns:
            model_name = MODEL_CONFIGS[model]['name']
            summary[f'{model_name}_mean'] = predictions_df[model].mean()
            summary[f'{model_name}_std'] = predictions_df[model].std()
            summary[f'{model_name}_range'] = predictions_df[model].max() - predictions_df[model].min()

        # Consensus statistics
        summary['consensus_mean'] = consensus_df['mean_prediction'].mean()
        summary['consensus_std'] = consensus_df['mean_prediction'].std()
        summary['avg_uncertainty'] = consensus_df['std_prediction'].mean()
        summary['high_agreement_pct'] = (consensus_df['agreement_level'] == 'High').mean() * 100

        # Save summary
        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ['Value']
        summary_df.to_csv(report_dir / "summary_statistics.csv")

    def _generate_comparison_plots(self, predictions_df, consensus_df, report_dir):
        """Generate model comparison plots."""

        # 1. Correlation matrix
        plt.figure(figsize=(10, 8))
        corr_matrix = predictions_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Model Prediction Correlations')
        plt.tight_layout()
        plt.savefig(report_dir / "model_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Prediction distributions
        plt.figure(figsize=(12, 8))
        predictions_df.boxplot()
        plt.title('Prediction Distributions by Model')
        plt.ylabel('Log Permeability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(report_dir / "prediction_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Pairwise scatter plots
        if len(predictions_df.columns) >= 2:
            from itertools import combinations

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, (model1, model2) in enumerate(combinations(predictions_df.columns, 2)):
                if i >= 6:  # Limit to 6 comparisons
                    break

                ax = axes[i]
                ax.scatter(predictions_df[model1], predictions_df[model2], alpha=0.6)
                ax.plot([predictions_df[model1].min(), predictions_df[model1].max()],
                       [predictions_df[model1].min(), predictions_df[model1].max()], 'r--')
                ax.set_xlabel(MODEL_CONFIGS[model1]['name'])
                ax.set_ylabel(MODEL_CONFIGS[model2]['name'])

                # Calculate R²
                corr = predictions_df[model1].corr(predictions_df[model2])
                ax.set_title(f'R² = {corr**2:.3f}')

            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.savefig(report_dir / "pairwise_comparisons.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _generate_consensus_plots(self, consensus_df, report_dir):
        """Generate consensus analysis plots."""

        # 1. Uncertainty distribution
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(consensus_df['std_prediction'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Standard Deviation')
        plt.ylabel('Count')
        plt.title('Uncertainty Distribution')

        plt.subplot(1, 2, 2)
        agreement_counts = consensus_df['agreement_level'].value_counts()
        plt.pie(agreement_counts.values, labels=agreement_counts.index, autopct='%1.1f%%')
        plt.title('Agreement Levels')

        plt.tight_layout()
        plt.savefig(report_dir / "consensus_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Consensus vs uncertainty
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(consensus_df['mean_prediction'], consensus_df['std_prediction'],
                            c=consensus_df['confidence_score'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Confidence Score')
        plt.xlabel('Mean Prediction')
        plt.ylabel('Prediction Standard Deviation')
        plt.title('Consensus Prediction vs Uncertainty')
        plt.tight_layout()
        plt.savefig(report_dir / "consensus_vs_uncertainty.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_performance_plots(self, predictions_df, actual_values, report_dir):
        """Generate model performance comparison plots."""
        from sklearn.metrics import r2_score, mean_squared_error

        # Calculate performance metrics
        performance = {}
        for model in predictions_df.columns:
            valid_mask = ~predictions_df[model].isna()
            if valid_mask.sum() > 0:
                y_true = actual_values[valid_mask]
                y_pred = predictions_df[model][valid_mask]

                performance[model] = {
                    'R2': r2_score(y_true, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'MAE': np.mean(np.abs(y_true - y_pred))
                }

        # Performance comparison plot
        if performance:
            metrics_df = pd.DataFrame(performance).T

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i, metric in enumerate(['R2', 'RMSE', 'MAE']):
                ax = axes[i]
                bars = ax.bar(metrics_df.index, metrics_df[metric])
                ax.set_title(f'{metric} by Model')
                ax.set_ylabel(metric)
                plt.setp(ax.get_xticklabels(), rotation=45)

                # Add value labels on bars
                for bar, value in zip(bars, metrics_df[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(report_dir / "model_performance.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Save performance metrics
            metrics_df.to_csv(report_dir / "performance_metrics.csv")

    def _generate_text_report(self, input_data, predictions_df, consensus_df, actual_values, report_dir):
        """Generate text-based summary report."""

        report_lines = []
        report_lines.append("PharmPapp Batch Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Summary
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of Molecules: {len(predictions_df)}")
        report_lines.append(f"Models Used: {', '.join([MODEL_CONFIGS[m]['name'] for m in predictions_df.columns])}")
        report_lines.append("")

        # Model training summary
        report_lines.append("Model Training Performance:")
        report_lines.append("-" * 30)
        for model_name, metrics in self.models_trained.items():
            model_display = MODEL_CONFIGS[model_name]['name']
            report_lines.append(f"{model_display:15} - R²: {metrics['test_r2']:.3f}, RMSE: {metrics['test_rmse']:.3f}")
        report_lines.append("")

        # Consensus summary
        report_lines.append("Consensus Analysis:")
        report_lines.append("-" * 20)
        report_lines.append(f"Mean Prediction: {consensus_df['mean_prediction'].mean():.3f} ± {consensus_df['mean_prediction'].std():.3f}")
        report_lines.append(f"Average Uncertainty: {consensus_df['std_prediction'].mean():.3f}")
        report_lines.append(f"High Agreement: {(consensus_df['agreement_level'] == 'High').mean() * 100:.1f}%")
        report_lines.append("")

        # Top/bottom predictions
        sorted_pred = consensus_df.sort_values('mean_prediction')

        report_lines.append("Highest Permeability Predictions:")
        report_lines.append("-" * 35)
        for i in range(min(5, len(sorted_pred))):
            idx = sorted_pred.index[-i-1]
            report_lines.append(f"  Molecule {idx}: {sorted_pred.loc[idx, 'mean_prediction']:.3f} ± {sorted_pred.loc[idx, 'std_prediction']:.3f}")

        report_lines.append("")
        report_lines.append("Lowest Permeability Predictions:")
        report_lines.append("-" * 34)
        for i in range(min(5, len(sorted_pred))):
            idx = sorted_pred.index[i]
            report_lines.append(f"  Molecule {idx}: {sorted_pred.loc[idx, 'mean_prediction']:.3f} ± {sorted_pred.loc[idx, 'std_prediction']:.3f}")

        # Save report
        with open(report_dir / "analysis_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))

    def run_batch_analysis(self, input_file=None, has_target=False):
        """
        Run complete batch analysis.

        Args:
            input_file (str, optional): Input file with molecular descriptors
            has_target (bool): Whether input file contains target values
        """
        logger.info("Starting batch analysis...")

        # Train all models
        self.train_all_models()

        if not self.predictors:
            raise ValueError("No models successfully trained")

        # Load input data or use demo data
        if input_file:
            logger.info(f"Loading input data from {input_file}")
            # Load first available model's example to get format
            first_model = list(self.predictors.keys())[0]
            predictor = self.predictors[first_model]
            input_data, actual_values = predictor.load_data(input_file, has_target=has_target)
        else:
            logger.info("Using demo data from all models")
            # Combine all example datasets
            all_data = []
            all_targets = []

            for model_name, predictor in self.predictors.items():
                X, y = predictor.load_data(MODEL_CONFIGS[model_name]['example_file'], has_target=True)
                all_data.append(X)
                all_targets.extend(y)

            # Use the largest dataset as reference
            input_data = max(all_data, key=len)
            actual_values = np.array(all_targets[:len(input_data)]) if has_target else None

        # Make predictions with all models
        results_data, predictions_df = self.predict_all_models(input_data)

        if predictions_df.empty:
            raise ValueError("No successful predictions made")

        # Analyze consensus
        consensus_df = self.analyze_consensus(predictions_df)

        # Generate comprehensive report
        self.generate_report(input_data, predictions_df, consensus_df, actual_values)

        # Save results
        combined_results = pd.concat([results_data, predictions_df, consensus_df], axis=1)
        combined_results.to_csv(self.output_dir / "batch_predictions.csv", index=False)

        logger.info("Batch analysis completed successfully")

        return combined_results, predictions_df, consensus_df

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='PharmPapp Batch Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run batch analysis with custom data
    python use_case_3_batch_analysis.py --input molecules.csv --output results/

    # Run demo analysis
    python use_case_3_batch_analysis.py --demo --output demo_results/

    # Analyze data with known permeability values
    python use_case_3_batch_analysis.py --input test_data.csv --has-target --output validation/

Output:
    The tool generates:
    - batch_predictions.csv: All predictions and consensus analysis
    - report/: Directory with plots and detailed analysis
    - summary_statistics.csv: Summary metrics
        """
    )

    parser.add_argument('--input', type=str, help='Input CSV file with molecular descriptors')
    parser.add_argument('--output', type=str, default='batch_results', help='Output directory')
    parser.add_argument('--has-target', action='store_true', help='Input contains target permeability values')
    parser.add_argument('--demo', action='store_true', help='Run demo using example data')

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = PharmPappBatchAnalyzer(args.output)

        # Run analysis
        if args.demo:
            logger.info("Running demo batch analysis")
            results, predictions, consensus = analyzer.run_batch_analysis()
        else:
            if not args.input:
                raise ValueError("Input file required (use --demo for demo mode)")
            results, predictions, consensus = analyzer.run_batch_analysis(args.input, args.has_target)

        # Print summary
        logger.info("\nBatch Analysis Summary:")
        logger.info(f"  Molecules analyzed: {len(results)}")
        logger.info(f"  Models used: {len(predictions.columns)}")
        logger.info(f"  Mean prediction: {consensus['mean_prediction'].mean():.3f}")
        logger.info(f"  Average uncertainty: {consensus['std_prediction'].mean():.3f}")
        logger.info(f"  High agreement: {(consensus['agreement_level'] == 'High').mean() * 100:.1f}%")
        logger.info(f"\nResults saved to: {analyzer.output_dir}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())