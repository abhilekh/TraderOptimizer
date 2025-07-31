import argparse
import json
from pathlib import Path
import pdb
import pandas as pd
from src.config_manager import ConfigManager
from src.data_manager import DataManager
from src.runner import apply_strategy, generate_param_combinations,  load_configuration, save_results
from src.strategy import Strategy
from src.backtester import Backtester
from src.utils import Utils, UtilsPath


def setupargs() -> argparse.Namespace:
     # Set up argument parser to accept a run ID
    parser = argparse.ArgumentParser(description="Run a backtest for a given configuration.")
    parser.add_argument(
        "-c",
        "--config_id",
        type=str,
        required=True,
        help="The ID of the run configuration to execute (e.g., 'BEL_RSI_Aggressive_Daily').",
    )
    parser.add_argument(
        "-o",
        "--run_optimizer",
        action="store_true",  # This makes it a flag
        help="Used to run optimizer"
    )
    args = parser.parse_args()
    return args


def run_strategy_capture_result(
          ticker_data: pd.DataFrame, output_path: Path, config: dict, optimization_run:bool = False) -> None:
        
        # 1. Apply Strategy
        signals: pd.DataFrame = apply_strategy(ticker_data, config)
        print(f"Successfully applied strategy: {config['strategy_id']}", signals.columns)

        # 2. Run Backtest
        initial_capital = config.get("initial_capital", 10000.0)
        backtester = Backtester(
            df_ticker_data=ticker_data,
            df_with_signals=signals,
            initial_capital=initial_capital,
            commission=config.get("commission", 0.001)
        )
        portfolio, trades = backtester.run_backtest()
        print("Backtest complete. Analyzing performance...", portfolio.columns)
        pdb.set_trace()
        
        # 3. Analyze Performance and Save Results
        # Construct a descriptive output filename
        filename_prefix = f"backtest_{config['ticker']}_{config['strategy_id']}_{config['new_timeframe']}"
        if optimization_run:
            filename_prefix += f"_optimzer"
        
        output_filepath = output_path / f"{filename_prefix}.txt"
        print(f"Analyzing performance...{output_filepath}")
        (metrics,  completed_trades) = backtester.analyze_performance(output_filepath = Path(output_filepath))
        metrics_str = "\n".join(metrics)
        print(metrics_str)
        
        if output_filepath is not None:
            UtilsPath.write_file(file_path=output_filepath, data=metrics_str, createPath=True, extension=".txt")
            
        # Save only completed trades to the final CSV
        if not completed_trades.empty:
            completed_trades.to_csv(str(output_filepath).replace('.txt', '_trades.csv'), index=False)
        print(f"Performance metrics: {metrics}")
        save_results(portfolio, signals, output_path, config)
        print(f"Backtest results saved to dir: {output_filepath}")


def main():
    """
    Main function to run the backtesting process.
    """
    args = setupargs()
    run_id: str = args.config_id
    run_optimizer: bool = args.run_optimizer
    print(run_optimizer)

    # Define base paths
    base_path = Path(__file__).resolve().parent.parent
    config_path = base_path / "config"
    output_path = base_path / "output"
    output_path.mkdir(exist_ok=True)

    try:
        # 1. Load Configuration
        config = load_configuration(config_path, run_id, run_optimizer)
        print(f"Successfully loaded configuration for run ID: {run_id}")

        # 2. Fetch Data
        data_manager = DataManager(
            cache_dir= base_path / "data"
        )
        ticker = config["ticker"]
        
        ticker_data = data_manager.fetch_data(ticker=ticker, start_date=config["start_date"], end_date=config["end_date"], interval=config["new_timeframe"])
        print(f"Successfully fetched data for {ticker}.", ticker_data.columns)

        if not run_optimizer:
            run_strategy_capture_result(ticker_data=ticker_data, output_path=output_path, config=config)
        else:
            print("Config is ", json.dumps(config, indent=2))
            # Run optimizer logic here if needed
            print("Running optimizer...")
            param_generator = generate_param_combinations(config)
            for i, params in enumerate(param_generator):
                print("Config is ", json.dumps(params, indent=2))
                run_strategy_capture_result(ticker_data=ticker_data, output_path=output_path, config=params, optimization_run=True)
            print("Optimizer run completed. No specific logic implemented yet.")

    except (ValueError, FileNotFoundError) as e:
        print(f"Value Error/FileNotFoundError seen: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()