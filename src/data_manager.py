"""
Manages fetching, caching, and loading of financial market data.
"""

import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from typing import Union


class DataManager:
    """
    Handles fetching data from Yahoo Finance and caching it locally.
    """

    def __init__(self, cache_dir: Union[Path, str]):
        """
        Initializes the DataManager.

        Args:
            cache_dir (str or Path): The directory to use for caching data files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_filepath(
            self,
            ticker: str,
            interval: str,
            start_date: str,
            end_date: str) -> Path:
        """Constructs a standardized filepath for caching."""
        filename = f"{ticker}_{interval}_{start_date}_{end_date}.csv"
        return self.cache_dir / filename

    def fetch_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetches data for a given ticker, utilizing a local cache.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date for the data (YYYY-MM-DD).
            end_date (str): The end date for the data (YYYY-MM-DD).
            interval (str): The data interval (e.g., '1d' for daily).

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data.
        """
        cache_filepath = self._get_cache_filepath(
            ticker, interval, start_date, end_date)
        print(
            f"Checking cache for {ticker} data from {start_date} to {end_date}...at {cache_filepath}")

        if cache_filepath.exists():
            print(f"Loading data for {ticker} from cache...")
            try:
                data = pd.read_csv(
                    cache_filepath,
                    index_col='Date',
                    parse_dates=True)
                return data
            except (IOError, pd.errors.EmptyDataError) as e:
                print(
                    f"Cache file for {ticker} is invalid ({e}). Redownloading...")

        print(f"Downloading data for {ticker} from yfinance...")
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval)
            if data is None or data.empty:
                print(f"No data returned for {ticker} from yfinance.")
                return pd.DataFrame()

            data.to_csv(cache_filepath)
            print(f"Data for {ticker} cached successfully.")
            return data
        except Exception as e:
            print(
                f"An error occurred while downloading data for {ticker}: {e}")
            return pd.DataFrame()
