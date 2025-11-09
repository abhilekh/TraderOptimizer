import datetime
from typing import Dict, Any

from src.constant import AssetType, TradeSide, HoldingType
from src.utils import UtilsJson


class CommissionCalculator:
    """
    Calculates transaction fees based on a hierarchical fee schedule 
    and determines trade type (Delivery/Intraday) for regulatory fee application.
    """

    def __init__(self, master_schedule: Dict[str, Any]):
        """Initializes the calculator with the master fee schedule."""
        self.master_schedule = master_schedule

    def _get_effective_fees(self, broker_key: str, asset_type: AssetType) -> Dict[str, Any]:
        """
        Retrieves and merges the effective fee schedule for a specific broker and asset.

        Uses UtilsJson for broker-level inheritance and dictionary merging.
        """
        asset_key = asset_type.value

        # 1. Use UtilsJson.get_merged_section to handle broker inheritance (Shallow Merge).
        # This merges top-level keys (like 'stocks', 'options') from the base.
        try:
            merged_broker_config = UtilsJson.get_merged_section(
                config_data=self.master_schedule,
                derived_key_id=broker_key,
                base_key_identifier='inherits_from'
            )
        except ValueError as e:
            # Handle error where broker key is not found or inheritance fails
            if "not found" in str(e):
                raise ValueError(
                    f"Broker key '{broker_key}' not found in master schedule.") from e
            raise e

        # 2. Extract and deep-merge the configuration for the specific asset.

        # Start with the base asset config as the foundation (use a shallow copy)
        base_asset_config = self.master_schedule["base"].get(asset_key, {})
        effective_config = base_asset_config.copy()

        # Apply the merged broker's asset-specific overrides
        merged_asset_config = merged_broker_config.get(asset_key, {})

        # Use the new UtilsJson.deep_merge to recursively merge the asset-specific
        # overrides (which contain nested 'broker' and 'regulatory' dicts) into the base config.
        # This ensures immutability of the master schedule.
        effective_config = UtilsJson.deep_merge(
            effective_config, merged_asset_config)

        # 3. Final validation
        if not effective_config.get("broker") and not effective_config.get("regulatory"):
            raise ValueError(
                f"Asset type '{asset_key}' configuration not found in schedule for broker '{broker_key}'.")

        return effective_config

    def _determine_holding_type(self, buy_datetime: str | None, sell_datetime: str | None) -> HoldingType:
        """Determines if a stock trade is 'Delivery' (held > 1 day) or 'Intraday' (held < 1 day)."""
        if not buy_datetime or not sell_datetime:
            return HoldingType.UNKNOWN

        try:
            # Assuming input date strings are in ISO format (e.g., '2025-10-09T10:00:00')
            buy_dt = datetime.datetime.fromisoformat(buy_datetime)
            sell_dt = datetime.datetime.fromisoformat(sell_datetime)
        except ValueError:
            return HoldingType.UNKNOWN

        if sell_dt <= buy_dt:
            return HoldingType.INTRADAY  # Or immediate

        time_difference = sell_dt - buy_dt

        # Check for same-day trade or held less than 24 hours (Intraday)
        if buy_dt.date() == sell_dt.date() or time_difference < datetime.timedelta(days=1):
            return HoldingType.INTRADAY

        # Held overnight or longer
        return HoldingType.DELIVERY

    def _calculate_stocks_fees(self, principal_value: float, trade_side: TradeSide, holding_type: HoldingType, rates: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates fees specifically for Stock trades (Delivery/Intraday)."""

        broker_rates = rates.get("broker", {})
        regulatory_rates = rates.get("regulatory", {})

        # --- 4.1. Brokerage Calculation (Max(Rate-based, Constant) capped by MaxCap) ---

        rate_key = f"rate_{trade_side.value}"
        const_key = f"const_{trade_side.value}"
        cap_key = f"cap_{trade_side.value}"

        # 1. Calculate percentage brokerage
        percentage_brokerage = principal_value * \
            broker_rates.get(rate_key, 0.0)

        # 2. Apply constant floor (if any)
        constant_fee = broker_rates.get(const_key, 0.0)

        # Brokerage is Max(Percentage, Constant)
        brokerage = max(percentage_brokerage, constant_fee)

        # 3. Apply cap (if any)
        brokerage_cap = broker_rates.get(cap_key)
        if brokerage_cap is not None and brokerage > brokerage_cap:
            brokerage = brokerage_cap

        # --- 4.2. Regulatory Fees ---

        # Exchange Transaction Charges (on both sides)
        etc_charges = principal_value * regulatory_rates.get("etc_rate", 0.0)

        # SEBI Turnover Fee (on both sides)
        sebi_fee = principal_value * regulatory_rates.get("sebi_rate", 0.0)

        # Stamp Duty (only on Buy side)
        stamp_duty = principal_value * \
            regulatory_rates.get(
                "stamp_duty_rate", 0.0) if trade_side == TradeSide.BUY else 0.00

        # GST (18% applied on the total of commissions and exchange fees)
        # GST Base = Brokerage + ETC + SEBI Fee
        gst_base = brokerage + etc_charges + sebi_fee
        gst_tax = gst_base * regulatory_rates.get("gst_rate", 0.0)

        # STT (Securities Transaction Tax) - Varies by holding type
        stt_tax = 0.0
        if holding_type == HoldingType.DELIVERY and trade_side == TradeSide.SELL:
            stt_tax = principal_value * \
                regulatory_rates.get("stt_delivery", 0.0)
        elif holding_type == HoldingType.INTRADAY:
            # For intraday, STT is applied on both sides in this model's logic
            stt_tax = principal_value * \
                regulatory_rates.get("stt_intraday", 0.0)

        # --- 4.3. Total Fees ---
        total_fees = brokerage + etc_charges + sebi_fee + stamp_duty + gst_tax + stt_tax

        # --- 5. Return Results ---

        fees_breakdown = {
            "Asset Type": AssetType.STOCKS.value.capitalize(),
            "Trade Side": trade_side.name,
            # Using .name (DELIVERY, INTRADAY) for display
            "Holding Type": holding_type.name,
            "Principal Value (₹)": round(principal_value, 2),
            "Primary Brokerage (₹)": round(brokerage, 2),
            "Exchange Transaction Charges (₹)": round(etc_charges, 2),
            "SEBI Turnover Fee (₹)": round(sebi_fee, 2),
            "Stamp Duty (₹)": round(stamp_duty, 2),
            "GST (₹)": round(gst_tax, 2),
            "STT (Securities Transaction Tax) (₹)": round(stt_tax, 2),
            "TOTAL TRANSACTION FEE (₹)": round(total_fees, 2)
        }

        return fees_breakdown

    def calculate_commission_fees(
        self,
        broker_key: str,
        principal_value: float,
        trade_side: TradeSide,
        asset_type: AssetType,
        buy_datetime: str | None = None,
        sell_datetime: str | None = None
    ) -> Dict[str, Any]:
        """
        Calculates the total transaction fees for a trade, considering broker, 
        asset class, and holding period.

        Args:
            broker_key: Key for the broker schedule (e.g., 'zerodha').
            principal_value: The total value of the trade (Price * Quantity).
            trade_side: The side of the transaction (TradeSide.BUY or TradeSide.SELL).
            asset_type: The type of asset (AssetType.STOCKS, .OPTIONS, etc.).
            buy_datetime: Optional ISO-formatted string (YYYY-MM-DDTHH:MM) of the buy time.
            sell_datetime: Optional ISO-formatted string (YYYY-MM-DDTHH:MM) of the sell time.

        Returns:
            A dictionary containing the breakdown of all calculated fees and the total fee.
        """

        if principal_value <= 0:
            return {"error": "Principal value must be positive."}

        # 1. Get Effective Fee Structure for this broker/asset combo
        try:
            effective_rates = self._get_effective_fees(broker_key, asset_type)
        except ValueError as e:
            return {"error": str(e)}

        # 2. Determine Holding Type (relevant for STOCKS)
        holding_type: HoldingType = self._determine_holding_type(
            buy_datetime, sell_datetime)

        # 3. Call specific calculation logic based on asset type
        if asset_type == AssetType.STOCKS:
            return self._calculate_stocks_fees(principal_value, trade_side, holding_type, effective_rates)

        # Placeholder for other assets (only implemented for Stocks)
        elif asset_type in [AssetType.OPTIONS, AssetType.CURRENCY, AssetType.CRYPTO]:
            return {
                "Asset Type": asset_type.value.capitalize(),
                "error": f"Fee calculation for {asset_type.value.capitalize()} is not yet implemented. Only Stocks are currently supported."
            }

        return {"error": f"Unsupported asset type: {asset_type.value}"}

# --- Example Usage ---


if __name__ == '__main__':
    principal_value = 50000.00
    COMMISSION_DATA = UtilsJson.read_json_file("config/commission.json")
    calculator = CommissionCalculator(COMMISSION_DATA)

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
