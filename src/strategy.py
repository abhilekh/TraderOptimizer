"""
Defines various trading strategies, applying buy/sell/exit logic based on pre-calculated indicators.
"""
from typing import Any, Dict
import pandas as pd
import sys
from src.indicators import CalculateIndicator
class Strategy:
    """
    Implements trading strategies with four-state signal logic (Buy, Sell, Buy Close, Sell Close).
    """

    @classmethod
    def _apply_four_state_strategy_logic(
        cls,
        df: pd.DataFrame,
        buy_entry: pd.Series,
        sell_entry: pd.Series,
        long_exit: pd.Series,
        short_exit: pd.Series
    ) -> pd.DataFrame:
        """
        Applies four-state signal logic to a DataFrame based on entry and exit conditions.

        Args:
            df (pd.DataFrame): Input DataFrame with market data.
            buy_entry (pd.Series): Boolean Series indicating buy entry points.
            sell_entry (pd.Series): Boolean Series indicating sell entry points.
            long_exit (pd.Series): Boolean Series indicating long exit points.
            short_exit (pd.Series): Boolean Series indicating short exit points.

        Returns:
            pd.DataFrame: DataFrame with an added 'Signal' column representing trade actions.
        """
        signals = [0] * len(df)
        position = 0  # 0: No position, 1: Long, -1: Short

        for i in range(1, len(df)):
            is_buy_entry = buy_entry.iloc[i]
            is_sell_entry = sell_entry.iloc[i]
            is_long_exit = long_exit.iloc[i]
            is_short_exit = short_exit.iloc[i]

            if position == 0:
                if is_buy_entry:
                    signals[i] = 1  # Buy
                    position = 1
                elif is_sell_entry:
                    signals[i] = -1  # Sell
                    position = -1
            elif position == 1:
                if is_long_exit:
                    signals[i] = 2  # Buy Close
                    position = 0
            elif position == -1:
                if is_short_exit:
                    signals[i] = -2  # Sell Close
                    position = 0

        df['Signal'] = signals
        return df

    @classmethod
    def apply_rsi_ema_crossover(
        cls,
        data_frame: pd.DataFrame,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Applies a four-state RSI strategy using an EMA (or other MA) of RSI crossover.

        Args:
            data_frame (pd.DataFrame): Input DataFrame with market data.
            params (Dict[str, Any]): Parameters for RSI and MA calculation:
                - "RSI_Period": int, period for RSI calculation.
                - "MA_Period": int, period for MA calculation.
                - "MA_Type": str, type of MA ("EMA", "SMA", "WMA").
                - "rsi_threshold": float, threshold for RSI.

        Returns:
            pd.DataFrame: DataFrame with an added 'Signal' column representing trade actions.
        """
        df_copy = data_frame.copy()
        rsi_period = params.get("RSI_Period", 14)
        ma_period = params.get("MA_Period", 9)
        ma_type = params.get("MA_Type", "EMA").upper()
        rsi_threshold = params.get("rsi_threshold", 50)

        rsi_col = f'RSI_{rsi_period}'
        if ma_type not in ["EMA", "SMA", "WMA"]:
            print(f"Unsupported MA_Type '{ma_type}'. Defaulting to EMA.")
            ma_type = "EMA"
        ma_of_rsi_col = f'{ma_type}_of_{rsi_col}'

        # Calculate RSI if missing
        if rsi_col not in df_copy.columns:
            df_copy[rsi_col] = CalculateIndicator.calculate_rsi(df_copy, period=rsi_period)

        # Calculate MA of RSI if missing
        if ma_of_rsi_col not in df_copy.columns:
            df_copy[ma_of_rsi_col] = CalculateIndicator.calculate_ema(df_copy, period=ma_period, column=rsi_col)

        df_copy.dropna(subset=[rsi_col, ma_of_rsi_col], inplace=True)
        if df_copy.empty:
            return pd.DataFrame()

        buy_entry = (df_copy[rsi_col] > df_copy[ma_of_rsi_col]) & (df_copy[rsi_col] > rsi_threshold)
        sell_entry = (df_copy[rsi_col] < df_copy[ma_of_rsi_col]) & (df_copy[rsi_col] < rsi_threshold)
        long_exit = ~buy_entry
        short_exit = ~sell_entry
        # Ensure long_exit only follows a buy_entry (long position), and short_exit only follows a sell_entry (short position)
        long_exit = (buy_entry.shift(1, fill_value=False)) & (~buy_entry)
        short_exit = (sell_entry.shift(1, fill_value=False)) & (~sell_entry)

        return cls._apply_four_state_strategy_logic(df_copy, buy_entry, sell_entry, long_exit, short_exit)

    @classmethod
    def apply_moving_average_crossover(
        cls,
        df: pd.DataFrame,
        short_ma_col: str,
        long_ma_col: str
    ) -> pd.DataFrame:
        """
        Applies a four-state Moving Average Crossover strategy.

        Args:
            df (pd.DataFrame): Input DataFrame with market data.
            short_ma_col (str): Column name for short period moving average.
            long_ma_col (str): Column name for long period moving average.

        Returns:
            pd.DataFrame: DataFrame with an added 'Signal' column representing trade actions.

        Raises:
            ValueError: If required columns are not found in the DataFrame.
        """
        if short_ma_col not in df.columns or long_ma_col not in df.columns:
            raise ValueError(f"Required columns '{short_ma_col}' and '{long_ma_col}' not found.")

        df_copy = df.copy()

        bullish_crossover = (df_copy[short_ma_col] > df_copy[long_ma_col]) & (df_copy[short_ma_col].shift(1) <= df_copy[long_ma_col].shift(1))
        bearish_crossover = (df_copy[short_ma_col] < df_copy[long_ma_col]) & (df_copy[short_ma_col].shift(1) >= df_copy[long_ma_col].shift(1))

        buy_entry = bullish_crossover
        sell_entry = bearish_crossover
        long_exit = bearish_crossover
        short_exit = bullish_crossover

        return cls._apply_four_state_strategy_logic(df_copy, buy_entry, sell_entry, long_exit, short_exit)