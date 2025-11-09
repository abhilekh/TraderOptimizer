"""
Contains the core backtesting engine.

This module provides the Backtester class, which simulates trading strategies
based on signals, calculates equity over time, and generates detailed
performance metrics. It supports both long and short positions and features a
flexible commission model.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np

from src.constant import POSITION, SIGNAL, AssetType, TradeSide
from src.commission import CommissionCalculator

# from src.utils import UtilsPath


class Backtester:
    """
    Simulates trades based on a four-state signal system and analyzes performance,
    using the CommissionCalculator for realistic Indian brokerage fees.

    NOTE: Requires 'pandas' and 'numpy' to be installed.

    Attributes:
        initial_capital (float): The starting capital for the backtest.
        calculator (CommissionCalculator): Instance used for fee calculations.
        broker_key (str): Broker schedule key for the calculator.
        asset_type (AssetType): Asset type for the calculator.
        cash (float): The current cash balance.
        shares (float): The number of shares held (positive for long, negative for short).
        position (POSITION): The current market position (NOT_HELD, BUY, SELL).
        trades (list): A list of dictionaries, where each dict records a single trade
                       and includes entry/exit fees.
        equity_curve (pd.Series): A time series of the portfolio's total equity.
    """

    def __init__(self, initial_capital: float, broker_key: str, asset_type: AssetType):
        """
        Initializes the Backtester.

        Args:
            initial_capital (float): The starting capital.
            broker_key (str): The broker key (e.g., 'zerodha') for fee calculation.
            asset_type (AssetType): The asset type (e.g., AssetType.STOCKS).
        """
        # Commission setup
        self.calculator = CommissionCalculator(MASTER_FEE_SCHEDULE)
        self.broker_key = broker_key
        self.asset_type = asset_type

        # Initial state setup
        self.initial_capital = initial_capital
        self._reset_state()

    def _reset_state(self):
        """Resets the state for a new backtest run."""
        self.cash = self.initial_capital
        self.shares = 0.0
        self.position = POSITION.NOT_HELD
        self.trades = []
        self.equity_curve = None
        self.final_equity = 0

    def _get_trade_fees(
        self,
        principal_value: float,
        trade_side: TradeSide,
        buy_date: str | None = None,
        sell_date: str | None = None
    ) -> float:
        """Helper to calculate the total transaction fee using the CommissionCalculator."""
        try:
            result = self.calculator.calculate_commission_fees(
                broker_key=self.broker_key,
                principal_value=principal_value,
                trade_side=trade_side,
                asset_type=self.asset_type,
                buy_datetime=buy_date,
                sell_datetime=sell_date
            )
            # Check for error in calculation result
            if 'error' in result:
                # Log error and return 0 fees, allowing simulation to continue
                print(f"Warning: Fee calculation failed for trade: {result['error']}")
                return 0.0
                
            return result.get("TOTAL TRANSACTION FEE (₹)", 0.0)
        except Exception as e:
            print(f"Error calculating fees: {e}")
            return 0.0

    def run(
        self, df_with_signals: pd.DataFrame
    ):
        """
        Runs the backtest simulation based on a four-state signal system.
        Fees are calculated dynamically using the CommissionCalculator.
        """
        self._reset_state()
        signals_copy = df_with_signals.copy()
        equity = pd.Series(index=signals_copy.index, dtype=float)

        # Store the current trade's entry date for correct STT calculation
        current_entry_date = None

        for i in range(1, len(signals_copy)):
            signal = signals_copy['Signal'].iloc[i]
            price = signals_copy['Close'].iloc[i]
            current_date_str = signals_copy.index[i].isoformat()

            # --- Equity Calculation ---
            holdings_value = self.shares * price
            if self.position == POSITION.SELL:
                # Equity = Cash - Liability (Liability is absolute shares * current price)
                liability = abs(self.shares) * price
                equity.iloc[i] = self.cash - liability
            else:
                # Equity = Cash + Holdings
                equity.iloc[i] = self.cash + holdings_value
            # --- End Equity Calculation ---


            # --- Trade Logic ---
            if self.position == POSITION.NOT_HELD:
                if signal == SIGNAL.BUY_ENTRY:  # Long Entry
                    # 1. Total cash is the available principal for the trade
                    principal_value = self.cash
                    
                    # 2. Calculate fees (Stamp Duty applies on the Buy side)
                    fees = self._get_trade_fees(
                        principal_value=principal_value,
                        trade_side=TradeSide.BUY,
                        buy_date=current_date_str
                    )
                    
                    # 3. Calculate shares and update cash/shares
                    shares_capital = principal_value - fees
                    if shares_capital > 0:
                        self.shares = shares_capital / price
                        self.cash = 0.0 # Full deployment
                        self.position = POSITION.BUY
                        current_entry_date = current_date_str
                        self.trades.append(
                            {
                                'Entry_Date': signals_copy.index[i],
                                'Entry_Price': price,
                                'Type': 'Long',
                                'Shares': self.shares,
                                'Fees_Entry': fees
                            }
                        )
                        
                elif signal == SIGNAL.SELL_ENTRY:  # Short Entry
                    # 1. Determine size (shorting value = current cash used as collateral)
                    shares_to_short = self.cash / price
                    principal_value = shares_to_short * price
                    
                    # 2. Calculate fees on the short sale principal (Sell side fees)
                    fees = self._get_trade_fees(
                        principal_value=principal_value,
                        trade_side=TradeSide.SELL,
                        sell_date=current_date_str
                    )
                    
                    # 3. Update cash/shares
                    # Cash increases by the short sale value, minus the selling fees
                    self.cash += principal_value - fees
                    self.shares = -shares_to_short
                    self.position = POSITION.SELL
                    current_entry_date = current_date_str
                    self.trades.append(
                        {
                            'Entry_Date': signals_copy.index[i],
                            'Entry_Price': price,
                            'Type': 'Short',
                            'Shares': shares_to_short,
                            'Fees_Entry': fees
                        }
                    )

            elif self.position == POSITION.BUY:  # In a long position
                if signal == SIGNAL.BUY_EXIT:  # Long Close (Sell)
                    if not self.trades or 'Exit_Date' in self.trades[-1]: continue
                    
                    # 1. Calculate principal value of sale
                    principal_value = self.shares * price
                    
                    # 2. Calculate fees (requires buy and sell date for delivery/intraday STT)
                    entry_date_str = self.trades[-1]['Entry_Date'].isoformat()
                    fees = self._get_trade_fees(
                        principal_value=principal_value,
                        trade_side=TradeSide.SELL,
                        buy_date=entry_date_str,
                        sell_date=current_date_str
                    )
                    
                    # 3. Update cash/shares
                    # Cash received is principal minus all selling fees
                    self.cash += principal_value - fees
                    
                    # Update trade record
                    self.trades[-1].update(
                        {
                            'Exit_Date': signals_copy.index[i], 
                            'Exit_Price': price,
                            'Fees_Exit': fees
                        }
                    )
                    
                    self.shares = 0
                    self.position = POSITION.NOT_HELD
                    current_entry_date = None


            elif self.position == POSITION.SELL:  # In a short position
                if signal == SIGNAL.SELL_EXIT:  # Short Close (Buy to Cover)
                    if not self.trades or 'Exit_Date' in self.trades[-1]: continue
                    
                    shares_to_cover = abs(self.shares)
                    
                    # 1. Calculate principal value of cover purchase
                    principal_value = shares_to_cover * price
                    
                    # 2. Calculate fees (requires buy and sell date for delivery/intraday STT)
                    entry_date_str = self.trades[-1]['Entry_Date'].isoformat()
                    fees = self._get_trade_fees(
                        principal_value=principal_value,
                        trade_side=TradeSide.BUY,
                        buy_date=entry_date_str,
                        sell_date=current_date_str
                    )
                    
                    # 3. Update cash/shares
                    # Cost to cover is principal + fees
                    self.cash -= (principal_value + fees) 
                    
                    # Update trade record
                    self.trades[-1].update(
                        {
                            'Exit_Date': signals_copy.index[i], 
                            'Exit_Price': price,
                            'Fees_Exit': fees
                        }
                    )

                    self.shares = 0
                    self.position = POSITION.NOT_HELD
                    current_entry_date = None

        self.equity_curve = equity.dropna()
        if not self.equity_curve.empty:
            self.final_equity = self.equity_curve.iloc[-1]
        else:
            self.final_equity = self.initial_capital
            
        # If still in a position, update final equity with current market value
        if self.position != POSITION.NOT_HELD and not self.equity_curve.empty:
             self.final_equity = equity.iloc[-1]


    def get_trades(self) -> pd.DataFrame:
        """
        Calculates Gross P/L, Total Fees, and Net P/L for each completed trade.
        Fees are retrieved from the trade record (recorded during run).
        """
        if not self.trades:
            return pd.DataFrame()

        trades_df = pd.DataFrame(self.trades)
        
        # Filter for completed trades 
        completed_trades = trades_df.dropna(subset=['Exit_Price', 'Fees_Entry', 'Fees_Exit']).copy()
        completed_trades['Shares'] = completed_trades['Shares'].abs() # Use absolute shares for P/L calculation

        if not completed_trades.empty:
            
            # 1. Gross Profit/Loss (P/L) before fees
            long_gross_pl = (
                completed_trades['Exit_Price'] - completed_trades['Entry_Price']) * completed_trades['Shares']
            short_gross_pl = (
                completed_trades['Entry_Price'] - completed_trades['Exit_Price']) * completed_trades['Shares']

            completed_trades['Gross P/L'] = np.where(
                completed_trades['Type'] == 'Long',
                long_gross_pl,
                short_gross_pl)
            
            # 2. Total Fees
            completed_trades['Total Fees'] = completed_trades['Fees_Entry'] + completed_trades['Fees_Exit']
            
            # 3. Net P/L
            completed_trades['Net P/L'] = completed_trades['Gross P/L'] - completed_trades['Total Fees']
            
            # Rename for compatibility with analyze_performance 
            completed_trades['P/L'] = completed_trades['Net P/L']

        return completed_trades


    def _calculate_max_drawdown(self) -> float:
        """Calculates the maximum drawdown."""
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        return drawdown.min() * 100  # Return as a percentage

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculates the Sharpe ratio."""
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        # Calculate daily returns from the equity curve
        daily_returns = self.equity_curve.pct_change().dropna()
        
        # Only proceed if there are sufficient returns to calculate std dev
        if daily_returns.std() == 0 or len(daily_returns) == 0:
            return 0.0
        
        # Calculate Sharpe ratio (annualized using 252 trading days)
        sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std()
        return sharpe_ratio * np.sqrt(252)

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of performance metrics.
        Robustly handles cases with no trades.
        """
        if self.equity_curve is None or self.equity_curve.empty:
            return {
                "Error": "Backtest has not been run or generated no equity curve."}

        total_profit_loss = self.final_equity - self.initial_capital
        total_return_percentage = (
            total_profit_loss / self.initial_capital) * 100

        trades_df: pd.DataFrame = self.get_trades()
        # Use the 'P/L' column created in get_trades (which is Net P/L)
        num_trades = len(trades_df)

        win_rate, avg_win, avg_loss, winning_trades_count, losing_trades_count = 0, 0, 0, 0, 0
        if num_trades > 0:
            winning_trades = trades_df[trades_df['P/L'] > 0]
            losing_trades = trades_df[trades_df['P/L'] <= 0]
            winning_trades_count = len(winning_trades)
            losing_trades_count = len(losing_trades)
            win_rate = (winning_trades_count / num_trades) * \
                100 if num_trades > 0 else 0
            avg_win = winning_trades['P/L'].mean(
            ) if winning_trades_count > 0 else 0
            # Note: average loss is typically reported as a negative number
            avg_loss = losing_trades['P/L'].mean() if losing_trades_count > 0 else 0

        # FIX: Removed hardcoded '₹' from f-strings for currency fields.
        metrics: Dict[str, Any] = {
            "Initial Capital": f"{self.initial_capital:,.2f}",
            "Final Equity": f"{self.final_equity:,.2f}",
            "Total Net P/L": f"{total_profit_loss:,.2f}",
            "Total Return (%)": f"{total_return_percentage:.2f}%",
            "Sharpe Ratio": f"{self._calculate_sharpe_ratio():.2f}",
            "Max Drawdown (%)": f"{self._calculate_max_drawdown():.2f}%",
            "Number of Trades": num_trades,
            "Winning Trades": winning_trades_count,
            "Losing Trades": losing_trades_count,
            "Win Rate (%)": f"{win_rate:.2f}%",
            "Average Winning Trade": f"{avg_win:,.2f}",
            "Average Losing Trade": f"{avg_loss:,.2f}"
        }

        return metrics

# --- Example Usage ---


if __name__ == '__main__':
    principal_value = 50000.00
    calculator = CommissionCalculator(MASTER_FEE_SCHEDULE)

    # --- Scenario 1: Zerodha - Delivery Buy Trade (Zero Brokerage) ---
    print(f"--- Zerodha (Delivery) Buy Trade of ₹{principal_value:,.2f} ---")
    zerodha_delivery_buy = calculator.calculate_commission_fees(
        broker_key="zerodha",
        principal_value=principal_value,
        trade_side=TradeSide.BUY,
        asset_type=AssetType.STOCKS,
        buy_datetime="2025-10-09T10:00:00",
        sell_datetime="2025-10-10T10:00:00"  # Held overnight (Delivery)
    )
    for key, value in zerodha_delivery_buy.items():
        print(f"{key}: {value}")
    print("-" * 40)

    # --- Scenario 2: Zerodha - Delivery Sell Trade (STT Applied) ---
    print(f"--- Zerodha (Delivery) Sell Trade of ₹{principal_value:,.2f} ---")
    zerodha_delivery_sell = calculator.calculate_commission_fees(
        broker_key="zerodha",
        principal_value=principal_value,
        trade_side=TradeSide.SELL,
        asset_type=AssetType.STOCKS,
        buy_datetime="2025-10-09T10:00:00",
        sell_datetime="2025-10-10T10:00:00"  # Held overnight (Delivery)
    )
    for key, value in zerodha_delivery_sell.items():
        print(f"{key}: {value}")
    print("-" * 40)

    # --- Scenario 3: Base Broker - Intraday Sell Trade (High Brokerage + Intraday STT) ---
    intraday_value = 10000.00
    print(
        f"--- Base Broker (0.5% Brokerage) Intraday Sell Trade of ₹{intraday_value:,.2f} ---")
    base_intraday_sell = calculator.calculate_commission_fees(
        broker_key="base",
        principal_value=intraday_value,
        trade_side=TradeSide.SELL,
        asset_type=AssetType.STOCKS,
        buy_datetime="2025-10-09T10:00:00",
        sell_datetime="2025-10-09T14:30:00"  # Same day (Intraday)
    )
    for key, value in base_intraday_sell.items():
        print(f"{key}: {value}")
    print("-" * 40)

    # --- Scenario 4: Backtester Example ---
    try:
        # Mocking a basic DataFrame for a runnable example
        dates = pd.to_datetime([
            '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04',
            '2025-10-07', '2025-10-08', '2025-10-09'
        ])

        # Simple up-down price movement
        mock_data = {
            'Close': [100.0, 101.0, 102.0, 100.0, 105.0, 103.0, 106.0],
            # Signal: Buy, Wait, Close, Buy, Wait, Close, Wait
            'Signal': [SIGNAL.NO_ACTION, SIGNAL.BUY_ENTRY, SIGNAL.NO_ACTION, SIGNAL.BUY_EXIT,
                       SIGNAL.BUY_ENTRY, SIGNAL.NO_ACTION, SIGNAL.BUY_EXIT]
        }
        df_signals = pd.DataFrame(mock_data, index=dates)

        print("--- Backtester Simulation (Zerodha, Stocks) ---")
        backtester = Backtester(
            initial_capital=100000.0,
            broker_key="zerodha",
            asset_type=AssetType.STOCKS
        )
        backtester.run(df_signals)

        # Display performance metrics
        metrics = backtester.analyze_performance()
        for key, value in metrics.items():
            print(f"{key}: {value}")

        print("\nCompleted Trades (Net P/L includes all fees):")
        print(backtester.get_trades()[['Entry_Date',
                                       'Exit_Date',
                                       'Type',
                                       'Shares',
                                       'Gross P/L',
                                       'Total Fees',
                                       'Net P/L']])

    except NameError:
        print("\nNote: Backtester example requires pandas and numpy to run.")
