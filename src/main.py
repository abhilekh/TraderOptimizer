import argparse
from pathlib import Path
import pandas as pd
from src.constant import AssetType
from src.data_manager import DataManager
from src.runner import apply_strategy, generate_param_combinations, load_configuration
from src.strategy import Strategy
from src.backtester import Backtester
from src.utils import Utils, UtilsPath


def setup_args() -> argparse.Namespace:
    """
    Set up argument parser to accept a run ID.
    """
    parser = argparse.ArgumentParser(
        description="Run a backtest for a given configuration.")
    parser.add_argument(
        "-c",
        "--config_id",
        type=str,
        required=True,
        help="The ID of the run configuration to execute (e.g., 'BEL_RSI_Aggressive_Daily').",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        action="store_true",  # This makes it a flag
        help="Run in optimizer mode to test multiple parameter combinations."
    )
    return parser.parse_args()


def run_single_backtest(
        ticker_data: pd.DataFrame, output_path: Path,
        config: dict, optimization_run: bool = False) -> None:
    """
    Runs a single backtest and returns the performance metrics as a dictionary.
    This function is now separated to be callable by both single runs and the optimizer.
    """
    print("-" * 50)
    print(f"Running with params: {config.get('strategy_params')}")

    # 1. Apply Strategy
    signals_df: pd.DataFrame = apply_strategy(ticker_data, config)
    if signals_df.empty:
        print("Strategy generated no signals. Skipping backtest.")
        return {}

    # 2. Run Backtest
    initial_capital = config.get("initial_capital", 10000.0)
    backtester = Backtester(
        initial_capital=initial_capital,
        broker_key="zerodha",
        asset_type=AssetType.STOCKS
    )
    backtester.run(ticker_data, signals_df)

    # 3. Analyze Performance
    performance_summary = backtester.analyze_performance()

    # 4. Save detailed results for this specific run
    strategy_params_str = '_'.join(
        [f'{k}{v}' for k, v in config.get('strategy_params', {}).items()])
    filename_prefix = f"backtest_{
        config['ticker']}_{
        config['strategy_id']}_{
            config['new_timeframe']}_{strategy_params_str}"
    output_filepath = output_path / f"{filename_prefix}.txt"

    # Save metrics text file
    metrics_str = "\n".join(
        [f"{k}: {v}" for k, v in performance_summary.items()])
    UtilsPath.write_text_file(
        file_path=output_filepath,
        data=metrics_str,
        createPath=True)

    # Save trades csv
    trades_df = backtester.get_trades()
    if not trades_df.empty:
        trades_filepath = output_path / f"{filename_prefix}_trades.csv"
        trades_df.to_csv(trades_filepath, index=False)

    print(f"Individual run results saved to {output_path}")
    return performance_summary


def main():
    """
    Main function to run the backtesting process.
    """
    args = setup_args()
    run_id: str = args.config_id
    run_optimizer: bool = args.optimizer

    # Define base paths
    base_path = Path(__file__).resolve().parent.parent
    config_path = base_path / "config"
    output_path = base_path / "output" / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load Configuration
        config = load_configuration(config_path, run_id)
        print(f"Successfully loaded configuration for run ID: {run_id}")

        # 2. Fetch Data
        data_manager = DataManager(
            cache_dir=base_path / "data"
        )
        ticker_data = data_manager.fetch_data(
            ticker=config["ticker"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            interval=config["new_timeframe"]
        )
        print(
            f"Successfully fetched data for {
                config["ticker"]}.",
            ticker_data.columns)

        if not run_optimizer:
            print("--- Running in Single Strategy Mode ---")
            run_single_backtest(
                ticker_data=ticker_data,
                output_path=output_path,
                config=config)
            return

        # This is case of Optimizer.
        print("--- Running in Optimizer Mode ---")
        param_generator = generate_param_combinations(config)
        all_results = []

        for i, params_config in enumerate(param_generator):
            # Run backtest for one combination of parameters
            performance = run_single_backtest(
                ticker_data=ticker_data,
                output_path=output_path,
                config=params_config)

            # Collect results for final comparison
            if performance:
                result_row = {
                    **params_config.get('strategy_params', {}), **performance}
                all_results.append(result_row)

        if not all_results:
            print("Optimizer run completed, but no valid results were generated.")
            return

        # Create a summary DataFrame and save it
        results_df = pd.DataFrame(all_results)
        # Sort by a key metric, e.g., Sharpe Ratio. Handle cases where it might
        # be NaN.
        results_df.sort_values(
            by="Sharpe Ratio",
            ascending=False,
            inplace=True,
            na_position='last')

        summary_filepath = output_path / \
            f"optimizer_summary_{config['strategy_id']}.csv"
        results_df.to_csv(summary_filepath, index=False)
        print("-" * 50)
        print("OPTIMIZATION COMPLETE. Best results:")
        print(results_df.head())
        print(f"\nFull optimization summary saved to: {summary_filepath}")

    except (ValueError, FileNotFoundError) as e:
        print(f"Value Error/FileNotFoundError seen: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
