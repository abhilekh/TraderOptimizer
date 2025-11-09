"""
Defines various trading strategies, applying buy/sell/exit logic based on pre-calculated indicators.
"""
from typing import Dict, Any
import pandas as pd
from src.utils import UtilsDict
from src.constant import POSITION, SIGNAL, PRICE_TYPE
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
        signals = pd.Series(0, index=df.index)
        position = POSITION.NOT_HELD  # 0: No position, 1: Long, -1: Short

        for i in range(1, len(df)):
            if position == POSITION.NOT_HELD:
                if buy_entry.iloc[i]:
                    signals.iloc[i] = SIGNAL.BUY_ENTRY  # Buy
                    position = POSITION.BUY
                elif sell_entry.iloc[i]:
                    signals.iloc[i] = SIGNAL.SELL_ENTRY  # Sell
                    position = POSITION.SELL
            elif position == POSITION.BUY and long_exit.iloc[i]:
                signals.iloc[i] = SIGNAL.BUY_EXIT  # Buy Close
                position = POSITION.NOT_HELD
            elif position == POSITION.SELL and short_exit.iloc[i]:
                signals.iloc[i] = SIGNAL.SELL_EXIT  # Sell Close
                position = POSITION.NOT_HELD

        df['Signal'] = signals
        return df

    @classmethod
    def apply_rsi_ema_crossover(
        cls,
        data_frame: pd.DataFrame,
        params: Dict[str, Any],
        price_type: PRICE_TYPE = PRICE_TYPE.AT_MARKET_CLOSE,
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
        essential_keys: list[str] = [
            "RSI_Period",
            "MA_Period",
            "MA_Type",
            "rsi_threshold"]
        df_copy = data_frame.copy()
        if not set(essential_keys).issubset(params.keys()):
            print(params)
            raise ValueError("Necessary keys are absent.")
        rsi_period, ma_period, ma_type, rsi_threshold = UtilsDict.get_val_list(
            params, essential_keys)

        ma_calculators = {
            "EMA": CalculateIndicator.calculate_ema,
            "SMA": CalculateIndicator.calculate_sma,
            "WMA": CalculateIndicator.calculate_wma,
            "DEMA": CalculateIndicator.calculate_dema,
            "TEMA": CalculateIndicator.calculate_tema,
        }

        ma_func = ma_calculators.get(ma_type)
        if not ma_func:
            raise ValueError(
                f"Unsupported MA_Type '{ma_type}' specified in config.")

        rsi_col = f'RSI_{rsi_period}'
        ma_of_rsi_col = f'{ma_type}_{ma_period}_of_{rsi_col}'

        # Calculate RSI if missing
        if rsi_col not in df_copy.columns:
            df_copy[rsi_col] = CalculateIndicator.calculate_rsi(
                df_copy, period=rsi_period, price_type=price_type)
        # Calculate MA of RSI if missing
        if ma_of_rsi_col not in df_copy.columns:
            df_copy[ma_of_rsi_col] = ma_func(
                df_copy, period=ma_period, column=rsi_col)
        df_copy.dropna(subset=[rsi_col, ma_of_rsi_col], inplace=True)

        if df_copy.empty:
            return pd.DataFrame()

        # Define signals
        rsi_above_ma = df_copy[rsi_col] > df_copy[ma_of_rsi_col]
        rsi_below_ma = df_copy[rsi_col] < df_copy[ma_of_rsi_col]

        buy_entry = rsi_above_ma & (df_copy[rsi_col] > rsi_threshold)
        sell_entry = rsi_below_ma & (df_copy[rsi_col] < rsi_threshold)
        # Exit when the crossover condition reverses or RSI with threshold
        # changes
        long_exit = ~buy_entry
        short_exit = ~sell_entry
        # Ensure long_exit only follows a buy_entry (long position), and
        # short_exit only follows a sell_entry (short position)
        long_exit = (buy_entry.shift(1, fill_value=False)) & (~buy_entry)
        short_exit = (sell_entry.shift(1, fill_value=False)) & (~sell_entry)

        return cls._apply_four_state_strategy_logic(
            df_copy, buy_entry, sell_entry, long_exit, short_exit)

    @classmethod
    def apply_moving_average_crossover(
        cls,
        data_frame: pd.DataFrame,
        params: Dict[str, Any]
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
        df_copy = data_frame.copy()
        short_ma_period = params.get("Short_MA_Period", 50)
        long_ma_period = params.get("Long_MA_Period", 200)
        short_ma_type = params.get("Short_MA_Type", "SMA").upper()
        long_ma_type = params.get("Long_MA_Type", "SMA").upper()

        ma_calculators = {
            "EMA": CalculateIndicator.calculate_ema,
            "SMA": CalculateIndicator.calculate_sma,
            "WMA": CalculateIndicator.calculate_wma,
            "DEMA": CalculateIndicator.calculate_dema,
            "TEMA": CalculateIndicator.calculate_tema
        }

        short_ma_col = f"{short_ma_type}_{short_ma_period}"
        long_ma_col = f"{long_ma_type}_{long_ma_period}"

        def cal_ma(ma_col_str, ma_type, ma_period):
            if ma_col_str in df_copy.columns:
                return
            ma_func = ma_calculators.get(ma_type)
            df_copy[ma_col_str] = ma_func(
                df_copy['Close'],
                period=ma_period,
                price_type=PRICE_TYPE.AT_MARKET_CLOSE)

        cal_ma(short_ma_col, short_ma_type, short_ma_period)
        cal_ma(long_ma_col, long_ma_type, long_ma_period)
        bullish_crossover = (df_copy[short_ma_col] > df_copy[long_ma_col]) & (
            df_copy[short_ma_col].shift(1) <= df_copy[long_ma_col].shift(1))
        bearish_crossover = (df_copy[short_ma_col] < df_copy[long_ma_col]) & (
            df_copy[short_ma_col].shift(1) >= df_copy[long_ma_col].shift(1))

        buy_entry = bullish_crossover
        sell_entry = bearish_crossover
        long_exit = bearish_crossover
        short_exit = bullish_crossover

        return cls._apply_four_state_strategy_logic(
            df_copy, buy_entry, sell_entry, long_exit, short_exit)
