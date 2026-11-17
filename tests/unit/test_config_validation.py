from __future__ import annotations

import math

import pytest

from loraphymac.config.validation import ConfigurationError, validate_configuration
from loraphymac.engine import DEFAULT_CONFIG, build_config
from loraphymac.utils.generator import NodeGenerator


def _config(**overrides):
    config = build_config(overrides)
    validate_configuration(config, known_keys=DEFAULT_CONFIG)
    return config


def test_default_configuration_is_valid():
    _config()


def test_build_config_rejects_unknown_key():
    with pytest.raises(ConfigurationError, match="node_peroid"):
        build_config({"node_peroid": 60})


def test_gzip_scientific_tables_is_opt_in_and_requires_managed_persistence():
    assert "gzip_scientific_tables" not in DEFAULT_CONFIG
    _config(persistence_profile="scientific", gzip_scientific_tables=True)
    with pytest.raises(ConfigurationError, match="requires persistence_profile"):
        _config(gzip_scientific_tables=True)
    with pytest.raises(ConfigurationError, match="must be a boolean"):
        _config(persistence_profile="scientific", gzip_scientific_tables="yes")


def test_build_config_does_not_share_mutable_defaults_between_runs():
    first = build_config()
    first["node_sf_list"].append(8)

    second = build_config()
    assert second["node_sf_list"] == [7]
    assert DEFAULT_CONFIG["node_sf_list"] == [7]


def test_legacy_export_results_also_enables_explicit_summary_export():
    config = build_config({"export_results": True})

    assert config["export_results"] is True
    assert config["export_summary"] is True


def test_explicit_summary_switch_can_disable_legacy_alias():
    config = build_config({"export_results": True, "export_summary": False})

    assert config["export_results"] is True
    assert config["export_summary"] is False


@pytest.mark.parametrize(
    ("overrides", "key"),
    [
        ({"sim_time": 0}, "sim_time"),
        ({"sim_time": math.nan}, "sim_time"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"area_x": 0}, "area_x"),
        ({"area_y": 2_000}, "area_y"),
        ({"area_y": 2_001}, "area_y"),
        ({"node_num": -1}, "node_num"),
        ({"relay_num": 1.5}, "relay_num"),
        ({"node_period": 0}, "node_period"),
        ({"node_period": 10, "node_jitter": 10}, "node_jitter"),
        ({"node_payload_size": -1}, "node_payload_size"),
        ({"node_sf_list": []}, "node_sf_list"),
        ({"node_sf_list": [6]}, "node_sf_list"),
        ({"node_sf_list": [7, 7]}, "node_sf_list"),
        ({"node_bw": 100_000}, "node_bw"),
        ({"relay_sf": 13}, "relay_sf"),
        ({"routing_ch_id": 8}, "routing_ch_id"),
        ({"relay_tx_ch_ids": []}, "relay_tx_ch_ids"),
        ({"relay_tx_ch_ids": [1, 1]}, "relay_tx_ch_ids"),
        ({"node_ant_height": 0.4}, "node_ant_height"),
        ({"relay_duty_cycle": 0}, "relay_duty_cycle"),
        ({"relay_duty_cycle": 1.01}, "relay_duty_cycle"),
        (
            {"relay_tx_waiting_queue_capacity_packets": -1},
            "relay_tx_waiting_queue_capacity_packets",
        ),
        (
            {"gateway_tx_waiting_queue_capacity_packets": 2.5},
            "gateway_tx_waiting_queue_capacity_packets",
        ),
        ({"tx_queue_overflow_policy": "overwrite_oldest"}, "tx_queue_overflow_policy"),
        ({"apc_fopts_size": 16}, "apc_fopts_size"),
        ({"node_rx_window_duration_s": 0}, "node_rx_window_duration_s"),
        ({"rx_idle_poll_interval_ms": 0}, "rx_idle_poll_interval_ms"),
        ({"max_rx_idle_wait_ms": -1}, "max_rx_idle_wait_ms"),
        ({"output_dir": ""}, "output_dir"),
        ({"verbose": 1}, "verbose"),
        ({"debug_preview_rows": 1}, "debug_preview_rows"),
        ({"debug_preview_rows": 10.5}, "debug_preview_rows"),
        ({"node_ptx_mode": "dynamic"}, "node_ptx_mode"),
        ({"apc_downlink_mode": "logical"}, "apc_downlink_mode"),
        ({"node_ptx_init": 20}, "node_ptx_init"),
        ({"relay_ptx_init": 20, "gw_ant_gain": 12}, "relay_ptx_init"),
        ({"apc_downlink_ptx_dbm": 5}, "apc_downlink_ptx_dbm"),
        ({"tx_power_policy_id": "unknown"}, "tx_power_policy_id"),
        (
            {
                "tx_power_policy_id": "islam_algorithm2_fixed_sf_ptx_v1",
                "node_ptx_mode": "fixed",
                "relay_ptx_mode": "fixed",
            },
            "tx_power_model_id",
        ),
    ],
)
def test_invalid_configuration_is_rejected(overrides, key):
    with pytest.raises(ConfigurationError, match=key):
        _config(**overrides)


def test_adaptive_power_requires_frame_support():
    with pytest.raises(ConfigurationError, match="frame_enabled"):
        _config(frame_enabled=False)


def test_node_adaptive_power_requires_receive_window():
    with pytest.raises(ConfigurationError, match="node_rx_window_enabled"):
        _config(node_rx_window_enabled=False)


def test_custom_sir_matrix_must_cover_all_configured_spreading_factors():
    with pytest.raises(ConfigurationError, match="collision_sir_matrix"):
        _config(
            node_sf_list=[7, 8],
            collision_sir_matrix={7: {7: 6.0}},
            collision_sir_matrix_version="test-matrix",
        )


def test_node_generator_rejects_area_fully_inside_exclusion_zone():
    with pytest.raises(ValueError, match="exclusion"):
        NodeGenerator.uniform_random(
            n=1,
            x_range=(-10.0, 10.0),
            y_range=(-999.0, 999.0),
            height=2,
            radio=None,
            mcu=None,
            gain=0,
            channel_plan=None,
            air_interface=None,
            routing_engine=None,
            collision_engine=None,
            seed=1,
        )
