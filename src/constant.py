from enum import Enum, StrEnum


# class Position
class POSITION(Enum):
    NOT_HELD = 0
    BUY = 1
    SELL = 2


class TradeSide(Enum):
    """Defines the possible sides of a transaction."""
    NOT_HELD = 0
    BUY = 1
    SELL = 2


class SIGNAL(Enum):
    NO_ACTION = 0
    BUY_ENTRY = 1
    BUY_EXIT = 2
    SELL_ENTRY = 3
    SELL_EXIT = 4


class PRICE_TYPE(StrEnum):
    AT_MARKET_CLOSE = "close"
    AT_MARKET_OPEN = "open"
    MARKET_HIGH = "high"
    MARKET_LOW = "low"
    CUSTOM = "na"


class AssetType(Enum):
    """Defines the supported asset types."""
    STOCKS = 'stocks'
    OPTIONS = 'options'
    CURRENCY = 'currency'
    CRYPTO = 'crypto'


class HoldingType(Enum):
    """Defines the holding period for a stock transaction."""
    DELIVERY = 0
    INTRADAY = 1
    UNKNOWN = 2  # For cases where dates are missing or invalid
