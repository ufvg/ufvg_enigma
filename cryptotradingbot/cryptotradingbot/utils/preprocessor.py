from config import LOB_LEVELS
import numpy as np


def normalize_lob(lob):
    bids = np.array(lob['bids'], dtype=float)[:LOB_LEVELS]
    asks = np.array(lob['asks'], dtype=float)[:LOB_LEVELS]
    mid_price = (bids[0, 0] + asks[0, 0]) / 2
    bids[:, 0] = (bids[:, 0] - mid_price) / mid_price  # Price deviation
    asks[:, 0] = (asks[:, 0] - mid_price) / mid_price
    return np.concatenate([bids.flatten(), asks.flatten()])  # Shape: (20 * LOB_LEVELS,)