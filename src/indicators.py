"""
This module defines the `CalculateIndicator` class, which provides class methods to compute popular technical analysis indicators such as RSI, EMA, SMA, WMA, DEMA, and TEMA. The calculations leverage the `pandas_ta` library and are designed to work with both pandas DataFrame and Series inputs. Each method validates input types and columns, and provides clear error messages for incorrect usage.
Classes:
    CalculateIndicator: 
        Provides class methods to calculate various technical indicators using pandas_ta.
        Methods:
            _inner_calculate_indicator(data, indicator_func_name, indicator_display_name, period, column):
                Internal helper to calculate a specified indicator, handling both DataFrame and Series inputs.
            calculate_rsi(data, period):
                Calculates the Relative Strength Index (RSI) for the 'Close' column of a DataFrame.
            calculate_ema(data, period, column='Close'):
                Calculates the Exponential Moving Average (EMA) for a specified column or Series.
            calculate_sma(data, period, column='Close'):
                Calculates the Simple Moving Average (SMA) for a specified column or Series.
            calculate_wma(data, period, column='Close'):
                Calculates the Weighted Moving Average (WMA) for a specified column or Series.
            calculate_dema(data, period, column='Close'):
                Calculates the Double Exponential Moving Average (DEMA) for a specified column or Series.
            calculate_tema(data, period, column='Close'):
                Calculates the Triple Exponential Moving Average (TEMA) for a specified column or Series.
            TypeError: If input data is neither a DataFrame nor a Series, or if the indicator function cannot be called correctly.
            RuntimeError: For any other unexpected errors during indicator calculation.
"""

from typing import Union
import pandas as pd
import pandas_ta as ta # Import pandas_ta as a module (ta)

class CalculateIndicator:

    @classmethod
    def _inner_calculate_indicator(
        cls,
        data: Union[pd.DataFrame, pd.Series],
        indicator_func_name: str, # 'rsi', 'ema', 'sma' etc.
        indicator_display_name: str, # 'RSI', 'EMA', 'SMA' for error messages
        period: int,
        column: str
    ) -> pd.Series:
        """
        Calculates a specified technical indicator using pandas_ta.
        Handles calls for both DataFrame (with column) and Series inputs.

        Args:
            data (Union[pd.DataFrame, pd.Series]): Input financial data.
            indicator_func_name (str): The name of the pandas_ta function to call
                                       (e.g., 'rsi', 'ema', 'sma').
            indicator_display_name (str): User-friendly name for the indicator
                                          (used in error messages).
            period (int): The period for indicator calculation.
            column (str): Column name to use if 'data' is a DataFrame. Defaults to 'Close'.

        Returns:
            pd.Series: A Series containing indicator values.

        Raises:
            ValueError: If a specified column is not found in the DataFrame.
            TypeError: If 'data' is neither a DataFrame nor a Series,
                       or if the indicator function cannot be called correctly.
            AttributeError: If the pandas_ta method name is incorrect.
        """
        indicator_result: pd.Series
        
        # Try block for direct pandas_ta calls and input data checks
        try:
            if isinstance(data, pd.DataFrame):
                if column not in data.columns:
                    raise ValueError( # This ValueError should propagate directly
                        f"Column '{column}' not found in DataFrame for "
                        f"'{indicator_display_name}' calculation."
                    )
                method = getattr(data.ta, indicator_func_name)
                indicator_result = method(close=data[column], length=period)
                
            elif isinstance(data, pd.Series):
                method = getattr(ta, indicator_func_name)
                indicator_result = method(close=data, length=period)
            else:
                raise TypeError( # This TypeError should propagate directly
                    f"Input 'data' must be a pandas DataFrame or Series for "
                    f"'{indicator_display_name}' calculation."
                )
        
        except (ValueError, TypeError) as err:
            raise err
       
        # Catch specific pandas_ta related errors
        except AttributeError as err:
            # This catches errors if indicator_func_name is not found on .ta or ta module
            raise TypeError( # Re-raise as TypeError for incorrect method access
                f"Method name '{indicator_func_name}' not found for "
                f"'{indicator_display_name}' calculation. Error: {err}"
            ) from err
        
        # Do NOT catch ValueError or TypeError here, allow them to propagate
        # Catch any other truly unexpected errors and wrap them in RuntimeError
        except Exception as err:
            raise RuntimeError(
                f"An unexpected error occurred during '{indicator_display_name}' calculation: {err}"
            ) from err
                
        return indicator_result

    @classmethod
    def calculate_rsi(cls, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculates Relative Strength Index (RSI).
        Expects 'Close' column in the DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing financial data.
            period (int): The period for RSI calculation.

        Returns:
            pd.Series: A Series containing RSI values.
        """
        return cls._inner_calculate_indicator(
            data=data,
            indicator_func_name='rsi',
            indicator_display_name='RSI',
            period=period,
            column='Close' # RSI is always calculated on Close price of a DataFrame
        )

    @classmethod
    def calculate_ema(cls,
        data: Union[pd.DataFrame, pd.Series],
        period: int,
        column: str = 'Close'
    ) -> pd.Series:
        """
        Calculates Exponential Moving Average (EMA).
        Can accept a DataFrame (and a column name) or a Series directly.

        Args:
            data (Union[pd.DataFrame, pd.Series]): Input data for EMA calculation.
            period (int): The period for EMA calculation.
            column (str): Column name to use if 'data' is a DataFrame. Defaults to 'Close'.

        Returns:
            pd.Series: A Series containing EMA values.
        """
        return cls._inner_calculate_indicator(
            data=data,
            indicator_func_name='ema',
            indicator_display_name='EMA',
            period=period,
            column=column
        )

    @classmethod
    def calculate_sma(
        cls,
        data: Union[pd.DataFrame, pd.Series],
        period: int,
        column: str = 'Close'
    ) -> pd.Series:
        """
        Calculates Simple Moving Average (SMA).
        Can accept a DataFrame (and a column name) or a Series directly.

        Args:
            data (Union[pd.DataFrame, pd.Series]): Input data for SMA calculation.
            period (int): The period for SMA calculation.
            column (str): Column name to use if 'data' is a DataFrame. Defaults to 'Close'.

        Returns:
            pd.Series: A Series containing SMA values.
        """
        return cls._inner_calculate_indicator(
            data=data,
            indicator_func_name='sma',
            indicator_display_name='SMA',
            period=period,
            column=column
        )

    @classmethod
    def calculate_wma(
        cls, data: Union[pd.DataFrame, pd.Series],
        period: int, column: str = 'Close') -> pd.Series:
        """
        Calculates the Weighted Moving Average (WMA) for a given data series.

        Parameters:
            data (Union[pd.DataFrame, pd.Series]): The input data containing price values.
            period (int): The number of periods to use for calculating the WMA.
            column (str, optional): The column name to use from the DataFrame if `data` is a DataFrame. Defaults to 'Close'.

        Returns:
            pd.Series: A pandas Series containing the calculated WMA values.

        Raises:
            ValueError: If the specified column does not exist in the DataFrame.
            TypeError: If the input data is not a pandas DataFrame or Series.

        Example:
            >>> calculate_wma(df, period=10, column='Close')
        """
        return cls._inner_calculate_indicator(
            data=data,
            indicator_func_name='wma',
            indicator_display_name='WMA',
            period=period,
            column=column
        )

    @classmethod
    def calculate_dema(
        cls, data: Union[pd.DataFrame, pd.Series],
        period: int, column: str = 'Close') -> pd.Series:
        """
        Calculates the Double Exponential Moving Average (DEMA) for a given data series.

        Parameters:
            data (Union[pd.DataFrame, pd.Series]): Input data containing price information.
            period (int): The number of periods to use for the DEMA calculation.
            column (str, optional): The column name to use from the DataFrame if `data` is a DataFrame. Defaults to 'Close'.

        Returns:
            pd.Series: A pandas Series containing the DEMA values.

        Raises:
            ValueError: If the specified column does not exist in the DataFrame.

        Notes:
            DEMA is a technical indicator that aims to reduce the lag of traditional moving averages, providing a more responsive trend-following indicator.
        
        """
        return cls._inner_calculate_indicator(
            data=data,
            indicator_func_name='dema',
            indicator_display_name='DEMA',
            period=period,
            column=column
        )

    @classmethod
    def calculate_tema(
        cls, data: Union[pd.DataFrame, pd.Series],
        period: int, column: str = 'Close') -> pd.Series:
        """
        Calculates the Triple Exponential Moving Average (TEMA).
        """
        return cls._inner_calculate_indicator(
            data=data,
            indicator_func_name='tema',
            indicator_display_name='TEMA',
            period=period,
            column=column
        )


    # Add more indicator calculation functions here as needed (e.g., calculate_macd, calculate_stoch)
    # Example:
    # def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    #     # MACD often returns multiple series (MACD, Histogram, Signal Line), so it's a DataFrame
    #     return _inner_calculate_indicator_multi_output( # This would need a separate helper for multi-output
    #         data=data,
    #         indicator_func_name='macd',
    #         indicator_display_name='MACD',
    #         params={'fast': fast, 'slow': slow, 'signal': signal},
    #         column='Close'
    #     )