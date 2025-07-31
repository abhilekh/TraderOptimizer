"""
Core backtesting logic to simulate trades, calculate equity, and analyze performance.
"""
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from .utils import UtilsPath

class Backtester:
    """
    Backtester class to simulate trades based on a four-state signal system.
    It handles long and short positions, calculates equity, and analyzes performance.
    """
    
    def __init__(self, df_ticker_data: pd.DataFrame, df_with_signals: pd.DataFrame, initial_capital: float, commission: float = 0.001):
        """
        Initializes the Backtester with a DataFrame containing signals and an initial capital amount.
        
        Args:
            df_with_signals (pd.DataFrame): DataFrame with 'Close' prices and 'Signal' column.
            initial_capital (float): Initial capital for the backtest.
        """
        self.df_data = df_ticker_data
        self.df_with_signals = df_with_signals
        self.initial_capital = initial_capital
        self.commission = commission
        self.run_backtest(True, True)


    def run_backtest(
        self, calculate_for_closed_trades: bool = True,
        calculate_for_sells: bool = True
    ) -> Tuple[pd.DataFrame, float]:
        """
        Runs the backtest simulation based on a four-state signal system.
        Handles long and short positions.
        """
        signals_copy = self.df_with_signals.copy()
        cash = self.initial_capital
        shares = 0.0
        position = 0  # -1 for short, 0 for none, 1 for long
        trades = []
        
        signals_copy['Equity'] = self.initial_capital

        for i in range(1, len(signals_copy)):
            signal = signals_copy['Signal'].iloc[i]
            price = signals_copy['Close'].iloc[i]
            
            # Update equity at each step based on the current value of holdings
            signals_copy.loc[signals_copy.index[i], 'Equity'] = cash + (shares * price if position != -1 else -shares * price)

            if position == 0:  # Not in a position
                if signal == 1:  # Buy Entry
                    shares_to_buy = cash / price
                    cash = 0
                    shares = shares_to_buy
                    position = 1
                    trades.append({'Entry_Date': signals_copy.index[i], 'Entry_Price': price, 'Type': 'Long'})
                elif signal == -1:  # Sell Entry (Short)
                    shares_to_short = cash / price
                    # For shorting, cash increases by the value of the shorted stock
                    cash += shares_to_short * price
                    shares = -shares_to_short  # Use negative shares to represent a short position
                    position = -1
                    trades.append({'Entry_Date': signals_copy.index[i], 'Entry_Price': price, 'Type': 'Short'})

            elif position == 1:  # In a long position
                if signal == 2:  # Buy Close
                    cash += shares * price
                    shares = 0
                    position = 0
                    if trades and 'Exit_Date' not in trades[-1]:
                        trades[-1].update({'Exit_Date': signals_copy.index[i], 'Exit_Price': price})

            elif position == -1:  # In a short position
                if signal == -2:  # Sell Close
                    # Cost to buy back the shares
                    cost_to_cover = abs(shares) * price
                    cash -= cost_to_cover
                    shares = 0
                    position = 0
                    if trades and 'Exit_Date' not in trades[-1]:
                        trades[-1].update({'Exit_Date': signals_copy.index[i], 'Exit_Price': price})
                    
        self.trades_df = pd.DataFrame(trades)
        
        # --- Final Performance Calculation ---
        self.final_equity = signals_copy['Equity'].iloc[-1]

        # Calculate P/L only for completed trades
        if not self.trades_df.empty and 'Exit_Price' in self.trades_df.columns:
            completed_trades = self.trades_df.dropna(subset=['Exit_Price']).copy()
            
            if not completed_trades.empty:
                # Correct P/L calculation for both long and short trades
                long_pl = (completed_trades['Exit_Price'] - completed_trades['Entry_Price']) * (self.initial_capital / completed_trades['Entry_Price'])
                short_pl = (completed_trades['Entry_Price'] - completed_trades['Exit_Price']) * (self.initial_capital / completed_trades['Entry_Price'])

                completed_trades['Equity'] = long_pl.where(completed_trades['Type'] == 'Long', short_pl)
                self.trades_df = self.trades_df.merge(completed_trades[['Entry_Date', 'Equity']], on='Entry_Date', how='left')

        return self.trades_df, self.final_equity

    def analyze_performance(
        self,
        output_filepath: 'Optional[Path]' = None
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Calculates and prints performance metrics. Robustly handles cases with no trades.
        """
        print(f"Analyzing performance...{output_filepath}")
        final_equity: int = self.final_equity
        trades_df = self.trades_df.copy()

        total_profit_loss = final_equity - self.initial_capital
        total_return_percentage = (total_profit_loss / self.initial_capital) * 100
        
        completed_trades = trades_df.dropna(subset=['Exit_Price', 'Equity'])
        num_trades = len(completed_trades)
        
        win_rate, avg_win, avg_loss, winning_trades_count, losing_trades_count = 0, 0, 0, 0, 0
        if num_trades > 0:
            winning_trades = completed_trades[completed_trades['Equity'] > 0]
            losing_trades = completed_trades[completed_trades['Equity'] <= 0]
            winning_trades_count = len(winning_trades)
            losing_trades_count = len(losing_trades)
            win_rate = (winning_trades_count / num_trades) * 100 if num_trades > 0 else 0
            avg_win = winning_trades['Equity'].mean() if winning_trades_count > 0 else 0
            avg_loss = losing_trades['Equity'].mean() if losing_trades_count > 0 else 0

        metrics: List = [
            "--- Performance Metrics ---",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Equity: ${final_equity:,.2f}",
            f"Total Profit/Loss: ${total_profit_loss:,.2f}",
            f"Total Return (%): {total_return_percentage:.2f}%",
            f"Number of Trades: {num_trades}",
        ]
        if num_trades > 0:
            metrics.extend([
                f"Winning Trades: {winning_trades_count}",
                f"Losing Trades: {losing_trades_count}",
                f"Win Rate (%): {win_rate:.2f}%",
                f"Average Win: ${avg_win:,.2f}",
                f"Average Loss: ${avg_loss:,.2f}"
            ])

        
        
        return metrics, completed_trades