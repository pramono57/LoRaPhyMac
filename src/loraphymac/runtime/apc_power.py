# ============================================================
# runtime/apc_power.py
#
# R-APC TxPowerIndex <-> conducted PTx mapping
#
# Operational principles:
# - TxPowerIndex 0 = highest permitted operational PTx
# - Larger TxPowerIndex = lower conducted PTx
# - PTx interval = 2 dB per TxPowerIndex step
# - Operational PTx range follows entity.ptx_min / entity.ptx_max
# - entity.ptx_max is expected to have been resolved previously
#   from hardware capability and regulatory EIRP constraints
#
# This module does NOT define:
# - hardware maximum TX power
# - antenna gain
# - feeder loss
# - regulatory EIRP limits
#
# Those constraints are resolved before APC initialization.
# ============================================================


APC_TX_INDEX_STEP_DBM = 2


# ============================================================
# 1. OPERATIONAL PTx TABLE
# ============================================================
def _get_supported_ptx_values(entity):
    """
    Return ordered operational conducted-PTx values for an entity.

    The mapping is generated from:
        1. hardware-supported PTx levels in hw.tx_current;
        2. operational lower bound entity.ptx_min;
        3. operational upper bound entity.ptx_max;
        4. APC_TX_INDEX_STEP_DBM.

    Important:
        entity.ptx_max must already include any regulatory
        EIRP constraint before this function is called.

    Mapping principle:
        TxPowerIndex 0 = highest permitted operational PTx
        TxPowerIndex increases as PTx decreases.

    Example:
        Node:
            entity.ptx_min = 5
            entity.ptx_max = 17

            index 0 -> 17 dBm
            index 1 -> 15 dBm
            index 2 -> 13 dBm
            index 3 -> 11 dBm
            index 4 ->  9 dBm
            index 5 ->  7 dBm
            index 6 ->  5 dBm

        Relay:
            entity.ptx_min = 12
            entity.ptx_max = 20

            index 0 -> 20 dBm
            index 1 -> 18 dBm
            index 2 -> 16 dBm
            index 3 -> 14 dBm
            index 4 -> 12 dBm
    """

    if entity is None:
        raise ValueError("entity is None")

    entity_id = getattr(entity, "id", "ENTITY")

    if not hasattr(entity, "hw"):
        raise ValueError(
            f"{entity_id}: missing hw attribute"
        )

    if not hasattr(entity.hw, "tx_current"):
        raise ValueError(
            f"{entity_id}: hw.tx_current is not defined"
        )

    tx_current = entity.hw.tx_current

    if tx_current is None or len(tx_current) == 0:
        raise ValueError(
            f"{entity_id}: hw.tx_current is empty"
        )

    # --------------------------------------------------------
    # Operational PTx bounds must already be initialized.
    # Normally these are produced by the EIRP limiter.
    # --------------------------------------------------------
    if not hasattr(entity, "ptx_min"):
        raise ValueError(
            f"{entity_id}: entity.ptx_min is not initialized. "
            f"Apply operational PTx bounds before APC initialization."
        )

    if not hasattr(entity, "ptx_max"):
        raise ValueError(
            f"{entity_id}: entity.ptx_max is not initialized. "
            f"Apply regulatory EIRP / hardware PTx bounds "
            f"before APC initialization."
        )

    ptx_min = int(entity.ptx_min)
    ptx_max = int(entity.ptx_max)

    if ptx_min > ptx_max:
        raise ValueError(
            f"{entity_id}: invalid operational PTx range "
            f"[{ptx_min}, {ptx_max}] dBm"
        )

    # --------------------------------------------------------
    # Hardware-supported physical PTx levels
    # --------------------------------------------------------
    all_values = sorted(
        {int(v) for v in tx_current.keys()},
        reverse=True,
    )

    if not all_values:
        raise ValueError(
            f"{entity_id}: supported PTx list is empty"
        )

    hw_min = min(all_values)
    hw_max = max(all_values)

    if ptx_min < hw_min or ptx_max > hw_max:
        raise ValueError(
            f"{entity_id}: operational PTx range "
            f"[{ptx_min}, {ptx_max}] dBm exceeds "
            f"hardware-supported range "
            f"[{hw_min}, {hw_max}] dBm"
        )

    # ptx_max should already have been snapped to a
    # hardware-supported PTx value by the EIRP limiter.
    if ptx_max not in all_values:
        raise ValueError(
            f"{entity_id}: operational ptx_max={ptx_max} dBm "
            f"is not available in hw.tx_current. "
            f"EIRP limiter must select a hardware-supported "
            f"conducted PTx level."
        )

    step_dbm = int(APC_TX_INDEX_STEP_DBM)

    if step_dbm <= 0:
        raise ValueError(
            f"APC_TX_INDEX_STEP_DBM must be > 0, "
            f"got {step_dbm}"
        )

    # --------------------------------------------------------
    # Build APC operational PTx ladder.
    #
    # Anchor the 2-dB interval at entity.ptx_max, not at the
    # absolute hardware capability.
    # --------------------------------------------------------
    values = [
        ptx
        for ptx in all_values
        if (
            ptx_min <= ptx <= ptx_max
            and (ptx_max - ptx) % step_dbm == 0
        )
    ]

    if not values:
        raise ValueError(
            f"{entity_id}: no valid operational PTx values "
            f"for range [{ptx_min}, {ptx_max}] dBm "
            f"with {step_dbm} dB TxPowerIndex interval"
        )

    # TxPowerIndex is carried in 4 bits in APC_REQ.
    if len(values) > 16:
        raise ValueError(
            f"{entity_id}: {len(values)} TxPowerIndex levels "
            f"exceed 4-bit APC_REQ capacity (maximum 16 levels)"
        )

    # Defensive validation:
    # index 0 must always represent entity.ptx_max.
    if values[0] != ptx_max:
        raise ValueError(
            f"{entity_id}: invalid APC PTx mapping. "
            f"TxPowerIndex 0 must map to ptx_max={ptx_max} dBm, "
            f"but generated values are {values}"
        )

    return values


# ============================================================
# 2. PTx dBm -> TxPowerIndex
# ============================================================
def dbm_to_tx_power_index(entity, ptx_dbm: int) -> int:
    """
    Convert conducted PTx in dBm into APC TxPowerIndex.

    Principles:
        index 0 = highest permitted operational PTx
        larger index = lower PTx
        interval = APC_TX_INDEX_STEP_DBM

    The available mapping is determined dynamically from
    entity.ptx_min and entity.ptx_max.

    Example:
        If node.ptx_max = 17 dBm:

            17 dBm -> index 0
            15 dBm -> index 1
            13 dBm -> index 2

        If relay.ptx_max = 20 dBm:

            20 dBm -> index 0
            18 dBm -> index 1
            16 dBm -> index 2

    If ptx_dbm is not exactly one of the APC PTx levels,
    the nearest available operational PTx is selected.

    This function never returns an index corresponding to
    a PTx above entity.ptx_max.
    """

    values = _get_supported_ptx_values(entity)

    ptx_dbm = int(round(ptx_dbm))

    best_i = min(
        range(len(values)),
        key=lambda i: abs(values[i] - ptx_dbm),
    )

    return int(best_i)


# ============================================================
# 3. TxPowerIndex -> PTx dBm
# ============================================================
def tx_power_index_to_dbm(
    entity,
    tx_power_index: int,
) -> int:
    """
    Convert APC TxPowerIndex into physical conducted PTx.

    Principles:
        index 0 = entity.ptx_max
        larger index = lower conducted PTx

    The mapping follows the current operational PTx bounds
    of the transmitting entity.

    Example:
        entity.ptx_max = 17 dBm:

            index 0 -> 17 dBm
            index 1 -> 15 dBm
            index 2 -> 13 dBm

        entity.ptx_max = 20 dBm:

            index 0 -> 20 dBm
            index 1 -> 18 dBm
            index 2 -> 16 dBm

    Therefore, TxPowerIndex has no globally fixed dBm value.
    Its physical PTx depends on the operational PTx range
    of the corresponding transmitter.
    """

    values = _get_supported_ptx_values(entity)

    idx = int(tx_power_index)

    if idx < 0 or idx >= len(values):
        entity_id = getattr(entity, "id", "ENTITY")

        raise ValueError(
            f"{entity_id}: invalid TxPowerIndex={idx}. "
            f"Valid range: 0..{len(values) - 1}. "
            f"Operational PTx values: {values}"
        )

    return int(values[idx])


# ============================================================
# 4. MAXIMUM VALID TxPowerIndex
# ============================================================
def get_max_tx_power_index(entity) -> int:
    """
    Return the largest valid TxPowerIndex for the entity.

    Valid index range:
        0 .. get_max_tx_power_index(entity)

    Note:
        The maximum TxPowerIndex represents the LOWEST
        operational PTx in the current APC mapping.

    Examples:
        Node mapping:
            [17, 15, 13, 11, 9, 7, 5]
            valid index = 0..6
            get_max_tx_power_index() = 6

        Relay mapping:
            [20, 18, 16, 14, 12]
            valid index = 0..4
            get_max_tx_power_index() = 4
    """

    values = _get_supported_ptx_values(entity)

    return len(values) - 1
