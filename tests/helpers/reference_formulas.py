import math


FRAME_FIELD_BYTES = {
    "mhdr": 1,
    "src_id": 1,
    "next_hop": 1,
    "dst_id": 1,
    "hop_ctrl": 1,
    "fctrl": 1,
    "fcnt": 1,
    "fport": 1,
    "mic": 4,
}


def reference_frame_size(app_payload_size, fopts_len):
    """Independent sum of the documented wire-format fields."""
    return sum(FRAME_FIELD_BYTES.values()) + app_payload_size + fopts_len


def reference_energy_mj(voltage_v, current_ma, duration_s):
    return voltage_v * (current_ma / 1000.0) * duration_s * 1000.0


def reference_fspl_db(distance_m, frequency_mhz):
    distance_km = distance_m / 1000.0
    return 32.45 + 20.0 * math.log10(distance_km) + 20.0 * math.log10(
        frequency_mhz
    )


def reference_hata_correction_db(frequency_mhz, mobile_height_m):
    log_f = math.log10(frequency_mhz)
    return (1.1 * log_f - 0.7) * mobile_height_m - (1.56 * log_f - 0.8)


def reference_hata_db(
    distance_m,
    frequency_mhz,
    base_height_m,
    mobile_height_m,
    environment="urban",
):
    """Independent transcription of the standard Hata small/medium-city form."""
    log_f = math.log10(frequency_mhz)
    urban = (
        69.55
        + 26.16 * log_f
        - 13.82 * math.log10(base_height_m)
        - reference_hata_correction_db(frequency_mhz, mobile_height_m)
        + (44.9 - 6.55 * math.log10(base_height_m))
        * math.log10(distance_m / 1000.0)
    )
    if environment == "suburban":
        return urban - 2.0 * math.log10(frequency_mhz / 28.0) ** 2 - 5.4
    if environment == "rural":
        return urban - 4.78 * log_f**2 + 18.33 * log_f - 40.94
    return urban


def reference_noise_floor_dbm(bandwidth_hz, noise_figure_db):
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def reference_rssi_dbm(ptx_dbm, tx_gain_dbi, rx_gain_dbi, pathloss_db):
    return ptx_dbm + tx_gain_dbi + rx_gain_dbi - pathloss_db


def reference_lora_payload_symbols(
    sf,
    bandwidth_hz,
    payload_bytes,
    cr_index=1,
    crc_enabled=True,
    header_enabled=True,
    low_data_rate_opt=None,
):
    """LoRa payload-symbol formula; CR index 1..4 represents 4/5..4/8."""
    if low_data_rate_opt is None:
        low_data_rate_opt = int(bandwidth_hz == 125_000 and sf >= 11)
    de = int(low_data_rate_opt)
    ih = 0 if header_enabled else 1
    crc = 1 if crc_enabled else 0
    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih
    denominator = 4 * (sf - 2 * de)
    coded_symbols = math.ceil(numerator / denominator) * (cr_index + 4)
    return int(8 + max(coded_symbols, 0))


def reference_lora_toa_s(
    sf,
    bandwidth_hz,
    payload_bytes,
    cr_index=1,
    preamble_len=8,
    crc_enabled=True,
    header_enabled=True,
    low_data_rate_opt=None,
):
    symbol_time = (2**sf) / bandwidth_hz
    payload_symbols = reference_lora_payload_symbols(
        sf=sf,
        bandwidth_hz=bandwidth_hz,
        payload_bytes=payload_bytes,
        cr_index=cr_index,
        crc_enabled=crc_enabled,
        header_enabled=header_enabled,
        low_data_rate_opt=low_data_rate_opt,
    )
    return (preamble_len + 4.25 + payload_symbols) * symbol_time

