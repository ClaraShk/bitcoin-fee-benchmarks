"""
Constants for Bitcoin fee prediction benchmark.
"""

# Prediction horizons for fee prediction targets
# - '3h': Will the transaction be included within 3 hours?
# - '1d': Will the transaction be included within 1 day (24 hours)?
PREDICTION_HORIZONS = ['3h', '1d']

# Horizon durations in seconds
HORIZON_SECONDS = {
    '3h': 3 * 60 * 60,      # 10,800 seconds
    '1d': 24 * 60 * 60,     # 86,400 seconds
}

# Feature columns available in snapshot data
# These are the columns that can be used as inputs for prediction
FEATURE_COLUMNS = [
    'fee_rate',       # Fee rate in sat/vbyte
    'virtual_size',   # Virtual size in vbytes
    'size',           # Raw size in bytes
    'fee',            # Total fee in satoshis
    'num_of_inputs',  # Number of transaction inputs
    'output_value',   # Total output value in satoshis
]

# Timestamp columns for temporal ordering
TIMESTAMP_COLUMNS = [
    'first_seen_timestamp',  # When tx first entered mempool
    'block_timestamp',       # When tx was confirmed (None if unconfirmed)
    'snapshot_start',        # Start of snapshot window
    'snapshot_end',          # End of snapshot window
    'mempool_exit',          # When tx left mempool
]

# Target-related columns
TARGET_COLUMNS = [
    'block_timestamp',  # Primary: when tx was included in a block
    'block_height',     # Block number where tx was confirmed
]

# Transaction identifier
TX_ID_COLUMN = 'txid'

# Columns to load for a minimal working dataset
MINIMAL_COLUMNS = [
    'txid',
    'first_seen_timestamp',
    'block_timestamp',
    'fee_rate',
    'virtual_size',
    'fee',
]
