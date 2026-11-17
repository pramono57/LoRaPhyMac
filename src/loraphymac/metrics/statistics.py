"""Mutable runtime statistics state and backward-compatible metrics facade.

Recorder methods in this module own simulation-time mutation. Read-only table
derivation lives in the sibling metric-family modules and is exposed through
the stable ``Statistics`` methods used by the simulator and notebooks.
"""

from collections import Counter, defaultdict

import pandas as pd

from loraphymac.mac.frame_defs import CID, MType, mtype_name
from loraphymac.metrics.apc_metrics import (
    build_apc_control_history_df as _build_apc_control_history_df,
    build_final_ptx_distribution_df as _build_final_ptx_distribution_df,
    build_margin_history_df as _build_margin_history_df,
    build_ptx_history_df as _build_ptx_history_df,
)
from loraphymac.metrics.collision_metrics import (
    COLLISION_DECISION_COLUMNS as COLLISION_DECISION_COLUMNS,
    COLLISION_INTERFERENCE_EDGE_COLUMNS as COLLISION_INTERFERENCE_EDGE_COLUMNS,
    build_collision_event_log_df as _build_collision_event_log_df,
    build_collision_interference_edges_df as _build_collision_interference_edges_df,
    build_collision_summary_df as _build_collision_summary_df,
)
from loraphymac.metrics.core_summary import (
    build_core_summary_df as _build_core_summary_df,
)
from loraphymac.metrics.demodulator_metrics import (
    build_demodulator_event_log_df as _build_demodulator_event_log_df,
    build_demodulator_pool_summary_df as _build_demodulator_pool_summary_df,
)
from loraphymac.metrics.duty_cycle_metrics import (
    build_per_channel_duty_cycle_df as _build_per_channel_duty_cycle_df,
    build_relay_duty_cycle_df as _build_relay_duty_cycle_df,
)
from loraphymac.metrics.energy_metrics import (
    build_energy_breakdown_df as _build_energy_breakdown_df,
    continuous_rx_energy_mj as _continuous_rx_energy_mj,
)
from loraphymac.metrics.excel_export import (
    export_bundle_to_excel as _export_bundle_to_excel,
)
from loraphymac.metrics.frame_metrics import (
    build_frame_drop_reason_by_hop_df as _build_frame_drop_reason_by_hop_df,
    build_frame_hop_event_summary_df as _build_frame_hop_event_summary_df,
    build_frame_mtype_summary_df as _build_frame_mtype_summary_df,
    build_frame_summary_df as _build_frame_summary_df,
    normalize_frame_event_df as _normalize_frame_event_df,
)
from loraphymac.metrics.instrumentation_metrics import (
    CHANNEL_OCCUPANCY_COLUMNS as CHANNEL_OCCUPANCY_COLUMNS,
    NODE_TX_DUTY_CYCLE_COLUMNS as NODE_TX_DUTY_CYCLE_COLUMNS,
    TX_AIRTIME_COLUMNS as TX_AIRTIME_COLUMNS,
    build_apc_feedback_lifecycle_df as _build_apc_feedback_lifecycle_df,
    build_tx_airtime_by_entity_mtype_ptx_channel_df as _build_tx_airtime_df,
    build_wang_snr_report_lifecycle_df as _build_wang_snr_report_lifecycle_df,
    build_wang_power_command_lifecycle_df as _build_wang_power_command_lifecycle_df,
)
from loraphymac.metrics.output_bundles import (
    build_debug_bundle as _build_debug_bundle,
    build_result_bundle as _build_result_bundle,
    build_summary_bundle as _build_summary_bundle,
)
from loraphymac.metrics.packet_metrics import (
    PACKET_FATE_COLUMNS as PACKET_FATE_COLUMNS,
    build_delivered_packet_df as _build_delivered_packet_df,
    build_packet_failure_df as _build_packet_failure_df,
    build_packet_fate_df as _build_packet_fate_df,
    build_packet_fate_summary_df as _build_packet_fate_summary_df,
)
from loraphymac.metrics.queue_metrics import (
    TX_QUEUE_LIFECYCLE_COLUMNS as TX_QUEUE_LIFECYCLE_COLUMNS,
    build_relay_tx_mgc_queue_metrics_legacy_df as _build_relay_tx_mgc_queue_metrics_legacy_df,
    build_relay_tx_queue_metrics_df as _build_relay_tx_queue_metrics_df,
    build_tx_queue_lifecycle_df as _build_tx_queue_lifecycle_df,
    build_tx_queue_traffic_class_metrics_df as _build_tx_queue_traffic_class_metrics_df,
    build_tx_queue_waiting_time_df as _build_tx_queue_waiting_time_df,
    build_tx_queue_waiting_time_summary_df as _build_tx_queue_waiting_time_summary_df,
    percentile_ms as _queue_delay_percentile_ms,
    queue_length_metrics as _build_queue_length_metrics,
    queue_mean_ms as _build_queue_mean_ms,
    queue_percentile_ms as _build_queue_percentile_ms,
    tx_queue_state_metrics as _build_tx_queue_state_metrics,
)
from loraphymac.metrics.receiver_metrics import (
    build_hop_summary_df as _build_hop_summary_df,
    build_node_rx_matrix_df as _build_node_rx_matrix_df,
    build_relay_rx_stats_df as _build_relay_rx_stats_df,
    build_rx_channel_usage_df as _build_rx_channel_usage_df,
)
from loraphymac.metrics.views import (
    select_existing_columns as _select_existing_columns,
    view_apc_control_history as _view_apc_control_history,
    view_as_vertical as _view_as_vertical,
    view_core_summary as _view_core_summary,
    view_energy_breakdown as _view_energy_breakdown,
    view_final_ptx_distribution as _view_final_ptx_distribution,
    view_frame_summary as _view_frame_summary,
    view_frame_summary_vertical as _view_frame_summary_vertical,
    view_per_channel_duty_cycle as _view_per_channel_duty_cycle,
    view_relay_duty_cycle as _view_relay_duty_cycle,
)
from loraphymac.runtime.packet import Packet

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)


class Statistics:
    """Packet-centric runtime collector and stable table-access facade.

    ``record_*`` and finite-horizon finalization methods mutate raw state.
    ``get_*`` methods either expose raw state or delegate read-only derivation
    to a focused module in :mod:`loraphymac.metrics`.
    """

    # =========================================================
    # 1. INIT & BASIC HELPERS
    # =========================================================
    def __init__(self, output_profile: str = "detailed"):
        if output_profile not in {"summary", "detailed"}:
            raise ValueError("output_profile must be 'summary' or 'detailed'")
        self.output_profile = output_profile
        self.is_summary_profile = output_profile == "summary"
        # -----------------------------------------------------
        # Existing core storage
        # -----------------------------------------------------
        self.records = []
        # Exact event-count aggregates used by Summary instead of retaining a
        # Python dictionary for every TX/RX/DROP event.
        self.summary_frame_event_counts = Counter()
        self.summary_drop_reason_counts = Counter()
        # Keep Kahan compensation terms so streaming sums match pandas' grouped
        # floating-point accumulation used by Detailed output.
        self.summary_tx_airtime = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
        self.summary_node_tx_airtime = defaultdict(lambda: [0.0, 0.0])
        self.summary_channel_airtime = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.summary_channel_union = defaultdict(lambda: [None, None, 0.0])
        self.summary_ptx_update_count = 0
        self.summary_delivered_count = 0
        self.relay_stats = {}
        self.node_stats = {}
        self.rx_channel_use = defaultdict(lambda: defaultdict(int))
        self.rx_channel_duration = defaultdict(lambda: defaultdict(float))

        # -----------------------------------------------------
        # New storage
        # -----------------------------------------------------
        self.run_metadata = {}
        self.tx_queue_waiting_time = defaultdict(list)

        # Time-series queue length samples for RelayTX waiting queue.
        # Each sample stores queue length after an enqueue/dequeue event.
        # Used to compute time-weighted avg_queue_length and max_queue_length.
        self.tx_queue_length_samples = defaultdict(list)
        self.tx_queue_state_samples = defaultdict(list)
        self.tx_queue_lifecycle_records = []

        self.ptx_history = []          # time-series of PTX updates
        self.margin_history = []       # time-series of observed link quality
        self.delivered_packets = []    # successful packet arrivals at GW
        self.apc_control_history = []  # APC control history, physical or legacy
        self.apc_feedback_lifecycle = []
        self.apc_feedback_sequence = 0
        self.apc_feedback_event_sequence = 0
        self.wang_snr_report_lifecycle = []
        self.wang_snr_report_event_sequence = 0
        self.wang_snr_terminal_report_ids = set()
        self.wang_power_command_lifecycle = []
        self.wang_power_command_event_sequence = 0

        # SX1302 demodulator-pool packet-level instrumentation.
        self.demodulator_events = []
        self.summary_demodulator = {}

        # One receiver-scoped collision/capture decision per feasible packet.
        self.collision_decision_records = []
        # One directed row per desired packet and direct RF interferer.
        self.collision_interference_edge_records = []

        # One row per original end-device uplink. Only the latest non-terminal
        # state is retained during the run; finalization classifies every open
        # packet at the finite observation horizon.
        self.packet_fates = {}
        self.packet_fate_terminal_conflicts = []
        self.packet_observation_finalized = False
        self.application_queue_metrics = []
        # APC feedback robustness instrumentation is inert unless the new
        # experiment profile is explicitly selected in run metadata.
        self.feedback_attempt_records = []
        self.feedback_transactions = {}
        # AirInterface delivers Packet.copy() instances to receivers, so Python
        # object identity does not survive the TX/RX boundary. packet_uid does.
        self.feedback_attempt_by_packet_uid = {}
        self.feedback_attempt_counts = Counter()
        self.feedback_physical_sequence = 0
        self.feedback_tpi_desired = {}
        self.feedback_tpi_applied = {}
        self.feedback_tpi_version = {}
        # TPI versions belong to a controller/target/control-scope tuple.
        # Version zero is the resolved initial transmitter state; generated
        # commands start at one. DATA-TX snapshots are ephemeral and folded
        # immediately into bounded per-transmitter counters.
        self.feedback_tpi_registered_transmitters = {}
        self.feedback_tpi_scopes_by_target = defaultdict(set)
        self.feedback_tpi_decision_seen_by_target = set()
        self.feedback_tpi_latest_decided_by_scope = {}
        self.feedback_tpi_applied_by_scope = {}
        self.feedback_tpi_scope_by_feedback_id = {}
        self.feedback_tpi_version_by_feedback_id = {}
        self.feedback_tpi_data_tx_counters = defaultdict(Counter)
        self.feedback_tpi_physical_tx_sequence = 0
        self.feedback_tpi_event_sequence = 0
        self.feedback_airtime_by_role = Counter()
        self.feedback_airtime_by_mtype = Counter()
        self.feedback_total_tx_count = 0
        self.feedback_total_tx_airtime_s = 0.0
        self.feedback_control_tx_count = 0
        self.feedback_control_airtime_s = 0.0
        self.feedback_convergence_rows = []
        self.feedback_latest_transaction_by_target = {}

    def _feedback_robustness_enabled(self) -> bool:
        return self.run_metadata.get("experiment_profile") == "apc_feedback_robustness_sweep_v1"

    @staticmethod
    def _feedback_tpi_scope_id(context):
        """Return a stable control-link scope, or None when it is ambiguous."""
        controller = context.get("controller_id")
        target = context.get("target_id")
        scope = context.get("control_scope")
        if not controller or not target or scope not in {"node_apc", "relay_apc"}:
            return None
        return f"{controller}|{target}|{scope}"

    def register_feedback_tpi_transmitter(
        self, transmitter_id: str, transmitter_role: str, initial_tpi_index=None
    ) -> None:
        """Register initial operational state without adding a simulated event."""
        if not self._feedback_robustness_enabled():
            return
        role = str(transmitter_role)
        if role not in {"node", "relay"}:
            return
        key = str(transmitter_id)
        prior = self.feedback_tpi_registered_transmitters.get(key)
        state = {"transmitter_role": role, "initial_tpi_index": initial_tpi_index}
        if prior is not None and prior != state:
            raise ValueError(f"inconsistent APC TPI transmitter registration: {key}")
        self.feedback_tpi_registered_transmitters[key] = state

    def _next_feedback_tpi_event_sequence(self) -> int:
        self.feedback_tpi_event_sequence += 1
        return self.feedback_tpi_event_sequence

    def _record_feedback_data_tx_snapshot(
        self, packet, transmitter_id, event_time, packet_uid
    ):
        """Classify one physical DATA TX at its actual TX-start callback."""
        if not self._feedback_robustness_enabled():
            return None
        try:
            mtype = MType(int(getattr(packet, "mtype", -1)))
        except (TypeError, ValueError):
            return None
        if mtype not in {MType.DATA_UP, MType.DATA_UP_FWD}:
            return None

        transmitter_id = str(transmitter_id)
        mtype_name = mtype.name
        expected_role = "node" if mtype == MType.DATA_UP else "relay"
        registration = self.feedback_tpi_registered_transmitters.get(transmitter_id)
        transmitter_role = (
            registration["transmitter_role"]
            if registration is not None
            else self._infer_entity_type(transmitter_id)
        )
        tx_timestamp = self._safe_float(event_time)
        actual_ptx = self._safe_float(getattr(packet, "ptx_dbm", None))

        # A repeated callback for the same live packet object and TX start is
        # instrumentation duplication. Packet copies do not copy this private
        # callback marker, so a later forwarding/retry start receives a new ID.
        signature = (str(packet_uid), tx_timestamp)
        packet_signature = getattr(
            packet, "_statistics_last_physical_tx_signature", None
        )
        if packet_signature == signature:
            prior_snapshot = getattr(
                packet, "_statistics_last_physical_tx_snapshot", None
            )
            if prior_snapshot is not None and prior_snapshot["packet_uid"] == packet_uid:
                return prior_snapshot

        self.feedback_tpi_physical_tx_sequence += 1
        physical_tx_id = (
            f"{self.run_metadata.get('run_id') or 'run'}:data-tx:"
            f"{self.feedback_tpi_physical_tx_sequence:08d}"
        )
        try:
            setattr(packet, "physical_tx_id", physical_tx_id)
            setattr(packet, "_statistics_last_physical_tx_signature", signature)
        except (AttributeError, TypeError):
            pass

        decision_seen = transmitter_id in self.feedback_tpi_decision_seen_by_target
        scopes = self.feedback_tpi_scopes_by_target.get(transmitter_id, set())
        scope_id = next(iter(scopes)) if len(scopes) == 1 else None
        latest_version = (
            self.feedback_tpi_latest_decided_by_scope.get(scope_id)
            if scope_id is not None
            else None
        )
        applied_version = (
            self.feedback_tpi_applied_by_scope.get(scope_id)
            if scope_id is not None
            else None
        )

        if not decision_seen:
            status = "PRE-DECISION"
        elif (
            len(scopes) != 1
            or scope_id is None
            or latest_version is None
            or applied_version is None
            or tx_timestamp is None
            or actual_ptx is None
            or transmitter_role != expected_role
        ):
            status = "UNCLASSIFIED"
        elif applied_version > latest_version:
            status = "UNCLASSIFIED"
        elif applied_version < latest_version:
            status = "STALE"
        else:
            status = "CURRENT"

        sequence = self._next_feedback_tpi_event_sequence()
        snapshot = {
            "run_id": self.run_metadata.get("run_id"),
            "physical_tx_id": physical_tx_id,
            "packet_uid": packet_uid,
            "transmitter_id": transmitter_id,
            "transmitter_role": transmitter_role,
            "mtype": mtype_name,
            "control_scope_id": scope_id,
            "tx_timestamp_s": tx_timestamp,
            "actual_ptx_dbm": actual_ptx,
            "applied_tpi_version": applied_version,
            "latest_decided_tpi_version": latest_version,
            "stale_tpi_status": status,
            # Ephemeral sequencing proves same-timestamp callback order in
            # deterministic fixtures without persisting an event history.
            "event_sequence": sequence,
        }

        counter_key = (transmitter_id, transmitter_role, mtype_name, scope_id)
        counts = self.feedback_tpi_data_tx_counters[counter_key]
        counts["total_data_physical_tx_count"] += 1
        if status == "PRE-DECISION":
            counts["pre_decision_tx_count"] += 1
        else:
            counts["post_decision_data_tx_count"] += 1
            if status == "STALE":
                counts["eligible_data_tx_count"] += 1
                counts["stale_tpi_tx_count"] += 1
            elif status == "CURRENT":
                counts["eligible_data_tx_count"] += 1
                counts["current_tpi_tx_count"] += 1
            else:
                counts["unclassified_tx_count"] += 1
                if (
                    latest_version is not None
                    and applied_version is not None
                    and applied_version > latest_version
                ):
                    counts["version_invariant_violation_count"] += 1

        try:
            setattr(packet, "_statistics_last_physical_tx_snapshot", snapshot)
            setattr(packet, "_statistics_last_physical_tx_signature", signature)
        except (AttributeError, TypeError):
            pass
        return snapshot

    def _feedback_attempt_for_packet(self, packet):
        packet_uid = self._packet_uid(packet)
        if packet_uid is None:
            return None
        index = self.feedback_attempt_by_packet_uid.get(packet_uid)
        if index is None:
            return None
        return self.feedback_attempt_records[index]

    def _record_feedback_physical_tx(self, packet, node_id, event_time, observed_airtime_s):
        if not self._feedback_robustness_enabled():
            return
        from loraphymac.experiments.apc_feedback_robustness_sweep.feedback_loss import is_apc_feedback_frame

        role = self._infer_entity_type(str(node_id))
        self.feedback_total_tx_count += 1
        self.feedback_total_tx_airtime_s += float(observed_airtime_s)
        mtype_value = int(getattr(packet, "mtype", -1))
        if not is_apc_feedback_frame(packet, getattr(packet, "next_hop", "")):
            return
        info = packet.apc_info
        context = info["context"]
        feedback_id = str(info["feedback_id"])
        self.feedback_attempt_counts[feedback_id] += 1
        attempt_number = int(self.feedback_attempt_counts[feedback_id])
        self.feedback_physical_sequence += 1
        attempt_id = f"{self.run_metadata.get('run_id', 'run')}:apc-tx:{self.feedback_physical_sequence:06d}"
        target_id = str(context["target_id"])
        row = {
            "run_id": self.run_metadata.get("run_id"), "feedback_id": feedback_id,
            "physical_attempt_id": attempt_id, "attempt_number": attempt_number,
            "controller_id": str(context["controller_id"]), "target_transmitter_id": target_id,
            "transmitter_role": role, "mtype": mtype_value, "tx_timestamp_s": event_time,
            "channel_id": getattr(packet, "ch_id", None), "tx_airtime_s": float(observed_airtime_s),
            "ptx_dbm": getattr(packet, "ptx_dbm", None), "native_reception_status": "pending",
            "native_failure_reason": None, "logical_failure_reason": None,
            "synthetic_loss_probability": float(self.run_metadata.get("feedback_additional_loss_probability", 0.0)),
            "synthetic_erasure_decision": False, "logical_delivery_status": "pending",
            "tpi_application_status": "pending", "tpi_version": context.get("tpi_version"),
        }
        packet_uid = self.ensure_packet_uid(packet)
        if packet_uid in self.feedback_attempt_by_packet_uid:
            raise RuntimeError(
                f"duplicate physical APC feedback packet_uid {packet_uid!r}"
            )
        self.feedback_attempt_by_packet_uid[packet_uid] = len(self.feedback_attempt_records)
        self.feedback_attempt_records.append(row)
        transaction = self.feedback_transactions.get(feedback_id)
        if transaction is not None:
            transaction["physical_attempts"] += 1
        self.feedback_control_tx_count += 1
        self.feedback_control_airtime_s += float(observed_airtime_s)
        self.feedback_airtime_by_role[role] += float(observed_airtime_s)
        self.feedback_airtime_by_mtype[str(MType(mtype_value).name)] += float(observed_airtime_s)

    def evaluate_additional_feedback_loss(self, packet, receiver_id, time_s) -> bool:
        """Mark native success and apply only the opt-in independent erasure."""
        if not self._feedback_robustness_enabled():
            return False
        from loraphymac.experiments.apc_feedback_robustness_sweep.feedback_loss import (
            additional_feedback_erasure, is_apc_feedback_frame,
        )
        if not is_apc_feedback_frame(packet, receiver_id):
            return False
        row = self._feedback_attempt_for_packet(packet)
        if row is None:
            raise RuntimeError("eligible APC feedback reception has no physical-attempt record")
        row["native_reception_status"] = "success"
        transaction = self.feedback_transactions.get(str(packet.apc_info["feedback_id"]))
        if transaction is not None:
            transaction["native_successful_receptions"] += 1
        info = packet.apc_info
        context = info["context"]
        probability = float(self.run_metadata.get("feedback_additional_loss_probability", 0.0))
        erased, _uniform = additional_feedback_erasure(
            probability,
            seed=int(self.run_metadata["seed"]),
            controller_id=str(context["controller_id"]),
            target_id=str(context["target_id"]),
            feedback_id=str(info["feedback_id"]),
            attempt_number=int(row["attempt_number"]),
        )
        row["synthetic_erasure_decision"] = bool(erased)
        row["logical_delivery_status"] = "synthetic_erased" if erased else "delivered"
        if erased:
            transaction = self.feedback_transactions[str(info["feedback_id"])]
            transaction["synthetic_erasures"] += 1
            transaction["status"] = "pending_retry_or_application"
        return bool(erased)

    def _record_feedback_margin_observation(self, entity_id, margin, decoded, time_s):
        if not self._feedback_robustness_enabled() or not decoded or margin is None:
            return
        target = str(entity_id)
        if not 4.0 < float(margin) < 8.0:
            return
        version = self.feedback_tpi_applied.get(target, 0)
        feedback_id = self.feedback_latest_transaction_by_target.get(target)
        transaction = self.feedback_transactions.get(feedback_id)
        if transaction is None or not transaction.get("tpi_state_changed") or transaction.get("tpi_version") != version:
            return
        if transaction.get("application_timestamp_s") is None or time_s <= transaction["application_timestamp_s"]:
            return
        if transaction.get("first_entry_timestamp_s") is not None:
            return
        transaction["first_entry_timestamp_s"] = float(time_s)
        self.feedback_convergence_rows.append({
            "run_id": self.run_metadata.get("run_id"), "feedback_id": feedback_id,
            "target_transmitter_id": target, "tpi_version": version,
            "decision_timestamp_s": transaction["decision_timestamp_s"],
            "application_timestamp_s": transaction["application_timestamp_s"],
            "observation_timestamp_s": float(time_s), "margin_db": float(margin),
            "margin_first_entry_latency_s": float(time_s) - transaction["decision_timestamp_s"],
            "observation_status": "first_entry_proxy",
        })

    def finalize_feedback_robustness(self, horizon_s):
        if not self._feedback_robustness_enabled():
            return
        for attempt in self.feedback_attempt_records:
            if attempt["native_reception_status"] == "pending":
                attempt["native_reception_status"] = "not_observed_by_horizon"
                attempt["logical_delivery_status"] = "horizon_censored"
        for transaction in self.feedback_transactions.values():
            if transaction.get("application_timestamp_s") is None:
                if transaction.get("retry_exhausted"):
                    transaction["status"] = "retry_exhausted"
                else:
                    transaction["status"] = "horizon_censored_pending"
            if transaction.get("tpi_state_changed") and transaction.get("application_timestamp_s") is not None and transaction.get("first_entry_timestamp_s") is None:
                self.feedback_convergence_rows.append({
                    "run_id": self.run_metadata.get("run_id"), "feedback_id": transaction["feedback_id"],
                    "target_transmitter_id": transaction["target_transmitter_id"],
                    "tpi_version": transaction["tpi_version"],
                    "decision_timestamp_s": transaction["decision_timestamp_s"],
                    "application_timestamp_s": transaction["application_timestamp_s"],
                    "observation_timestamp_s": float(horizon_s), "margin_db": None,
                    "margin_first_entry_latency_s": float(horizon_s) - transaction["decision_timestamp_s"],
                    "observation_status": "right_censored_no_first_entry",
                })

    def get_feedback_robustness_tables(self):
        from loraphymac.experiments.apc_feedback_robustness_sweep.contract import (
            AIRTIME_COLUMNS, ATTEMPT_COLUMNS, CONVERGENCE_COLUMNS,
            STALE_TPI_COLUMNS, STALE_TPI_TABLE_KEY, TRANSACTION_COLUMNS,
        )
        run_id = self.run_metadata.get("run_id")
        attempts = pd.DataFrame(self.feedback_attempt_records, columns=ATTEMPT_COLUMNS)
        transactions = pd.DataFrame(list(self.feedback_transactions.values()), columns=TRANSACTION_COLUMNS)
        zero_key_rows = set(self.feedback_tpi_data_tx_counters)
        for transmitter_id, registered in self.feedback_tpi_registered_transmitters.items():
            role = registered["transmitter_role"]
            mtype_name = "DATA_UP" if role == "node" else "DATA_UP_FWD"
            if not any(
                key[0] == transmitter_id and key[2] == mtype_name
                for key in zero_key_rows
            ):
                zero_key_rows.add((transmitter_id, role, mtype_name, None))
        stale_rows = []
        for transmitter_id, role, mtype_name, scope_id in sorted(
            zero_key_rows,
            key=lambda key: (key[0], key[1], key[2], key[3] or ""),
        ):
            counts = self.feedback_tpi_data_tx_counters.get(
                (transmitter_id, role, mtype_name, scope_id), Counter()
            )
            values = {
                name: int(counts[name])
                for name in (
                    "total_data_physical_tx_count", "pre_decision_tx_count",
                    "post_decision_data_tx_count", "eligible_data_tx_count",
                    "stale_tpi_tx_count", "current_tpi_tx_count",
                    "unclassified_tx_count", "version_invariant_violation_count",
                )
            }
            if values["total_data_physical_tx_count"] != (
                values["pre_decision_tx_count"] + values["post_decision_data_tx_count"]
            ):
                raise ValueError("stale-TPI total/pre/post accounting invariant failed")
            if values["post_decision_data_tx_count"] != (
                values["eligible_data_tx_count"] + values["unclassified_tx_count"]
            ):
                raise ValueError("stale-TPI post-decision accounting invariant failed")
            if values["eligible_data_tx_count"] != (
                values["stale_tpi_tx_count"] + values["current_tpi_tx_count"]
            ):
                raise ValueError("stale-TPI eligible accounting invariant failed")
            eligible = values["eligible_data_tx_count"]
            stale_rows.append({
                "run_id": run_id,
                "transmitter_id": transmitter_id,
                "transmitter_role": role,
                "mtype": mtype_name,
                "control_scope_id": scope_id,
                **values,
                "stale_tpi_transmission_fraction": (
                    values["stale_tpi_tx_count"] / eligible if eligible else None
                ),
            })
        stale = pd.DataFrame(stale_rows, columns=STALE_TPI_COLUMNS)
        convergence = pd.DataFrame(self.feedback_convergence_rows, columns=CONVERGENCE_COLUMNS)
        by_role = {key: float(value) for key, value in sorted(self.feedback_airtime_by_role.items())}
        by_mtype = {key: float(value) for key, value in sorted(self.feedback_airtime_by_mtype.items())}
        pct = 100.0 * self.feedback_control_airtime_s / self.feedback_total_tx_airtime_s if self.feedback_total_tx_airtime_s else 0.0
        airtime = pd.DataFrame([{
            "run_id": run_id, "apc_control_tx_count": int(self.feedback_control_tx_count),
            "apc_control_airtime_s": float(self.feedback_control_airtime_s),
            "total_observed_tx_count": int(self.feedback_total_tx_count),
            "total_observed_tx_airtime_s": float(self.feedback_total_tx_airtime_s),
            "apc_control_airtime_percent": float(pct),
            "by_transmitter_role_json": __import__("json").dumps(by_role, sort_keys=True),
            "by_mtype_json": __import__("json").dumps(by_mtype, sort_keys=True),
        }], columns=AIRTIME_COLUMNS)
        return {
            "feedback_attempt_metrics_df": attempts,
            "feedback_transaction_metrics_df": transactions,
            STALE_TPI_TABLE_KEY: stale,
            "apc_convergence_metrics_df": convergence,
            "apc_control_airtime_summary_df": airtime,
        }

    def _record_frame_event_summary(self, *, event, packet, reason=None):
        """Accumulate the exact fields used by frame/drop summary tables."""
        mtype = getattr(packet, "mtype", None)
        try:
            hop_count = max(0, len(getattr(packet, "path", [])) - 1)
        except TypeError:
            hop_count = 0
        key = (
            event,
            mtype,
            getattr(packet, "hop_index", None),
            hop_count,
            getattr(packet, "fctrl", None),
            len(getattr(packet, "fopts", b"") or b""),
            self._downlink_control_cid(packet),
            reason,
        )
        self.summary_frame_event_counts[key] += 1
        if event == "DROP":
            self.summary_drop_reason_counts[reason] += 1

    @staticmethod
    def _kahan_add(total, compensation, value):
        adjusted = value - compensation
        updated = total + adjusted
        compensation = (updated - total) - adjusted
        return updated, compensation

    def _record_summary_tx(self, packet, node_id, event_time, toa_s, observed_airtime_s):
        """Keep only grouped TX measurements and online channel busy unions."""
        mtype = getattr(packet, "mtype", None)
        if mtype == int(MType.DATA_UP):
            traffic_class = "uplink"
        elif mtype == int(MType.DATA_UP_FWD):
            traffic_class = "forwarding"
        elif mtype in (int(MType.DATA_DOWN), int(MType.DATA_DOWN_FWD)):
            traffic_class = "control_feedback"
        else:
            traffic_class = "other"
        entity_id = str(node_id)
        entity_type = self._infer_entity_type(entity_id)
        ptx = self._safe_float(getattr(packet, "ptx_dbm", None))
        channel = getattr(packet, "ch_id", None)
        sf = getattr(packet, "sf", None)
        bw_hz = getattr(packet, "bw_hz", None)
        key = (entity_id, entity_type, traffic_class, mtype, ptx, channel, sf, bw_hz)
        aggregate = self.summary_tx_airtime[key]
        aggregate[0] += 1
        aggregate[1], aggregate[2] = self._kahan_add(
            aggregate[1], aggregate[2], observed_airtime_s
        )
        aggregate[3], aggregate[4] = self._kahan_add(
            aggregate[3], aggregate[4], toa_s
        )
        self.summary_node_tx_airtime[entity_id][0], self.summary_node_tx_airtime[entity_id][1] = (
            self._kahan_add(
                self.summary_node_tx_airtime[entity_id][0],
                self.summary_node_tx_airtime[entity_id][1],
                observed_airtime_s,
            )
        )

        if channel is None or event_time is None:
            return
        sim_time_s = self._safe_float(self.run_metadata.get("sim_time"))
        if sim_time_s is None:
            sim_time_s = float("inf")
        start = max(0.0, float(event_time))
        end = min(sim_time_s, start + max(0.0, observed_airtime_s))
        channel_key = channel
        channel_totals = self.summary_channel_airtime[channel_key]
        channel_totals[0], channel_totals[2] = self._kahan_add(
            channel_totals[0], channel_totals[2], toa_s
        )
        channel_totals[1], channel_totals[3] = self._kahan_add(
            channel_totals[1], channel_totals[3], observed_airtime_s
        )

        # SimPy processes are serviced in nondecreasing env.now order. TX
        # records are emitted at physical TX start, so this is the same sorted
        # interval order used by build_channel_occupancy_df.
        current_start, current_end, busy = self.summary_channel_union[channel_key]
        if end <= start:
            return
        if current_start is None:
            self.summary_channel_union[channel_key] = [start, end, busy]
        elif start <= current_end:
            self.summary_channel_union[channel_key] = [
                current_start,
                max(current_end, end),
                busy,
            ]
        else:
            busy += current_end - current_start
            self.summary_channel_union[channel_key] = [start, end, busy]

    @staticmethod
    def _mj_to_j(x):
        return float(x) / 1000.0

    @staticmethod
    def _downlink_control_cid(packet):
        """Decode the control identity from physical downlink FOpts."""
        try:
            if int(getattr(packet, "mtype", -1)) not in (
                int(MType.DATA_DOWN),
                int(MType.DATA_DOWN_FWD),
            ):
                return None
            raw = bytes(getattr(packet, "fopts", b"") or b"")
            return int(raw[0]) if raw else None
        except (TypeError, ValueError):
            return None

    def _infer_entity_type(self, entity_id: str) -> str:
        s = str(entity_id)
        if s == "GW" or s.startswith("GW"):
            return "gateway"
        if s.startswith("R"):
            return "relay"
        return "node"

    def _select_existing_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Return selected columns that exist in df.
        Prevents KeyError when some columns are not available.
        """
        return _select_existing_columns(df, columns)

    # =========================================================
    # 2. RUN METADATA
    # =========================================================
    def set_run_metadata(self, **kwargs):
        """
        Save metadata for one run.
        """
        self.run_metadata = dict(kwargs)

    def get_run_metadata_df(self) -> pd.DataFrame:
        if not self.run_metadata:
            return pd.DataFrame()
        return pd.DataFrame([self.run_metadata])

    # =========================================================
    # 2B. ORIGINAL-UPLINK PACKET FATE
    # =========================================================
    @staticmethod
    def _packet_uid(packet):
        value = getattr(packet, "packet_uid", None)
        return str(value) if value not in (None, "") else None

    @staticmethod
    def _is_original_uplink(packet):
        try:
            hop_index = int(getattr(packet, "hop_index", -1))
        except (TypeError, ValueError):
            return False
        mtype = getattr(packet, "mtype", None)
        return (
            hop_index == 0
            and str(getattr(packet, "src_id", "")).startswith("N")
            and mtype in (None, MType.DATA_UP, int(MType.DATA_UP))
        )

    def ensure_packet_uid(self, packet):
        """Assign a deterministic within-run UID when a caller omitted one."""
        uid = self._packet_uid(packet)
        if uid is not None:
            return uid
        src_id = str(getattr(packet, "src_id", "UNKNOWN"))
        fcnt = getattr(packet, "fcnt", 0)
        mtype = getattr(packet, "mtype", None)
        prefix = (
            "UP"
            if mtype in (None, MType.DATA_UP, MType.DATA_UP_FWD,
                         int(MType.DATA_UP), int(MType.DATA_UP_FWD))
            else "CTRL"
        )
        uid = f"{prefix}:{src_id}:{fcnt}"
        packet.packet_uid = uid
        return uid

    def _register_offered_packet(self, packet, offered_time=None):
        uid = self.ensure_packet_uid(packet)
        offered_time = self._safe_float(offered_time)
        if offered_time is None:
            offered_time = self._safe_float(
                getattr(packet, "offered_time", None)
            )
        if offered_time is None:
            offered_time = self._safe_float(getattr(packet, "tx_start", None))
        existing = self.packet_fates.get(uid)
        if existing is not None:
            return existing
        row = {
            "run_id": self.run_metadata.get("run_id"),
            "packet_uid": uid,
            "src_id": getattr(packet, "src_id", None),
            "fcnt": getattr(packet, "fcnt", None),
            "birth_time": self._safe_float(getattr(packet, "birth_time", None)),
            "offered_time": offered_time,
            "final_status": "OPEN",
            "terminal_reason": None,
            "final_time": None,
            "observation_end_time": None,
            "age_at_horizon_s": None,
            "latest_state": "OFFERED",
            "latest_state_time": offered_time,
            "latest_entity_id": getattr(packet, "current_tx", None),
            "latest_hop_index": getattr(packet, "hop_index", None),
            "latest_next_hop": getattr(packet, "next_hop", None),
            "delivered_time": None,
            "e2e_delay_s": None,
            "e2e_delay_ms": None,
        }
        if self.is_summary_profile:
            row = {
                key: row[key]
                for key in (
                    "run_id", "packet_uid", "src_id", "fcnt", "birth_time",
                    "offered_time", "final_status", "terminal_reason",
                    "final_time", "observation_end_time", "age_at_horizon_s",
                    "latest_state", "latest_state_time", "delivered_time",
                    "e2e_delay_ms",
                )
            }
        self.packet_fates[uid] = row
        return row

    def record_application_offer(self, application):
        """Register one application admission before any physical TX exists."""
        row = self._register_offered_packet(
            application,
            offered_time=getattr(application, "offered_time", None),
        )
        self.record_packet_fate_state(
            application,
            state="QUEUE_WAITING",
            entity_id=getattr(application, "src_id", None),
            time=getattr(application, "offered_time", None),
        )
        return row

    def record_application_queue_metrics(self, record):
        self.application_queue_metrics.append(dict(record))

    def get_application_queue_metrics_df(self):
        columns = [
            "run_id",
            "node_id",
            "scheduling_mode",
            "rng_policy",
            "queue_policy",
            "horizon_policy",
            "observation_end_time",
            "generated_count",
            "dequeued_count",
            "max_backlog_packets",
            "mean_queue_waiting_time_s",
            "max_queue_waiting_time_s",
            "waiting_at_horizon_count",
            "in_service_at_horizon_count",
            "outstanding_at_horizon_count",
        ]
        if not self.application_queue_metrics:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(self.application_queue_metrics)
            .reindex(columns=columns)
            .sort_values("node_id", kind="stable")
            .reset_index(drop=True)
        )

    def record_packet_fate_state(
        self,
        packet,
        state,
        entity_id=None,
        time=None,
        reason=None,
    ):
        """Update the latest state of a tracked original uplink packet."""
        uid = self._packet_uid(packet)
        if uid is None or uid not in self.packet_fates:
            return
        row = self.packet_fates[uid]
        state = str(state).upper().strip()
        event_time = self._safe_float(time)
        if event_time is None:
            event_time = self._safe_float(getattr(packet, "rx_time", None))
        if event_time is None:
            event_time = self._safe_float(getattr(packet, "tx_start", None))

        terminal_status = {
            "DELIVERED": "DELIVERED",
            "TERMINAL_DROP": "TERMINAL_DROP",
        }.get(state)
        if terminal_status is not None:
            if row["final_status"] not in ("OPEN", terminal_status):
                self.packet_fate_terminal_conflicts.append({
                    "packet_uid": uid,
                    "previous_status": row["final_status"],
                    "new_status": terminal_status,
                    "time": event_time,
                })
                return
            row["final_status"] = terminal_status
            row["terminal_reason"] = reason
            row["final_time"] = event_time
            if terminal_status == "DELIVERED":
                row["delivered_time"] = event_time
                birth_time = row.get("birth_time")
                if birth_time is not None and event_time is not None:
                    delay_s = max(0.0, event_time - birth_time)
                    row["e2e_delay_s"] = delay_s
                    row["e2e_delay_ms"] = delay_s * 1000.0
        elif row["final_status"] != "OPEN":
            return

        row["latest_state"] = state
        row["latest_state_time"] = event_time
        row["latest_entity_id"] = (
            entity_id if entity_id is not None
            else getattr(packet, "current_tx", None)
        )
        row["latest_hop_index"] = getattr(packet, "hop_index", None)
        row["latest_next_hop"] = getattr(packet, "next_hop", None)

    def finalize_packet_observation(self, observation_end_time):
        """Classify every open offered packet at the hard run horizon."""
        if self.packet_observation_finalized:
            return
        horizon = float(observation_end_time)
        state_to_status = {
            "QUEUE_WAITING": "CENSORED_WAITING",
            "QUEUE_IN_SERVICE": "CENSORED_IN_SERVICE",
            "RX_ACTIVE": "CENSORED_RX",
        }
        for row in self.packet_fates.values():
            row["observation_end_time"] = horizon
            if row["final_status"] == "OPEN":
                row["final_status"] = state_to_status.get(
                    row.get("latest_state"),
                    "CENSORED_IN_FLIGHT",
                )
                row["final_time"] = horizon
            if str(row["final_status"]).startswith("CENSORED_"):
                offered_time = row.get("offered_time")
                row["age_at_horizon_s"] = (
                    max(0.0, horizon - offered_time)
                    if offered_time is not None else None
                )
        self.packet_observation_finalized = True

    def get_packet_fate_df(self):
        return _build_packet_fate_df(self.packet_fates)

    def get_packet_fate_summary_df(self):
        return _build_packet_fate_summary_df(
            self.get_packet_fate_df(),
            self.run_metadata,
            len(self.packet_fate_terminal_conflicts),
        )

    # =========================================================
    # 3. BASIC EVENT LOG
    # =========================================================
    @staticmethod
    def _safe_float(value):
        """Convert timing metadata to float without raising an exception."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _event_timing_fields(self, packet: Packet, event_time=None) -> dict:
        """
        Build a consistent timing snapshot for TX, RX, and DROP events.

        Definitions
        -----------
        tx_start:
            Physical transmission start of the current hop.
        tx_end:
            Physical transmission end of the current hop.
        t_arrive:
            Propagation-aware packet arrival time at the receiver.
        rx_start:
            Start of the receiver RF window for this packet.
        rx_end:
            End of the packet RF window (normally rx_start + ToA).
        event_time:
            Time at which the logged event is considered to occur.
        drop_time:
            Explicit runtime drop decision/interruption time. It remains None
            when the runtime module did not set packet.drop_time.
        queue_arrival_time:
            Time the packet entered a RelayTX shared waiting queue.
        queue_waiting_time:
            Time from queue arrival until the current-hop TX started.
        """
        tx_start = self._safe_float(getattr(packet, "tx_start", None))
        tx_end = self._safe_float(getattr(packet, "tx_end", None))
        t_arrive = self._safe_float(getattr(packet, "t_arrive", None))
        rx_start = self._safe_float(getattr(packet, "rx_start", None))
        rx_end = self._safe_float(getattr(packet, "rx_end", None))
        drop_time = self._safe_float(getattr(packet, "drop_time", None))
        queue_arrival_time = self._safe_float(
            getattr(packet, "tx_queue_arrival_time", None)
        )
        queue_dequeue_time = self._safe_float(
            getattr(packet, "tx_queue_dequeue_time", None)
        )
        duty_ready_time = self._safe_float(
            getattr(packet, "tx_duty_ready_time", None)
        )
        rx_idle_time = self._safe_float(
            getattr(packet, "tx_rx_idle_time", None)
        )
        service_end_time = self._safe_float(
            getattr(packet, "tx_service_end_time", None)
        )

        event_time = self._safe_float(event_time)

        fifo_waiting_time = None
        if queue_arrival_time is not None and queue_dequeue_time is not None:
            fifo_waiting_time = max(
                0.0,
                queue_dequeue_time - queue_arrival_time,
            )

        time_to_tx_start = None
        if queue_arrival_time is not None and tx_start is not None:
            time_to_tx_start = max(0.0, tx_start - queue_arrival_time)

        return {
            "tx_start": tx_start,
            "tx_end": tx_end,
            "t_arrive": t_arrive,
            "rx_start": rx_start,
            "rx_end": rx_end,
            "event_time": event_time,
            "drop_time": drop_time,
            "queue_visit_id": getattr(packet, "tx_queue_visit_id", None),
            "queue_owner_id": getattr(packet, "tx_queue_owner_id", None),
            "queue_arrival_time": queue_arrival_time,
            "queue_dequeue_time": queue_dequeue_time,
            "duty_ready_time": duty_ready_time,
            "rx_idle_time": rx_idle_time,
            "service_end_time": service_end_time,
            "queue_worker_id": getattr(packet, "tx_worker_id", None),
            "fifo_waiting_time": fifo_waiting_time,
            "time_to_tx_start": time_to_tx_start,
            # Backward-compatible alias. Historically this field represented
            # queue arrival to physical TX start, not FIFO-only waiting.
            "queue_waiting_time": time_to_tx_start,
        }

    def record_tx(self, packet: Packet, node_id: str):
        event_time = self._safe_float(getattr(packet, "tx_start", None))
        packet_uid = self.ensure_packet_uid(packet)
        stale_tpi_snapshot = self._record_feedback_data_tx_snapshot(
            packet, node_id, event_time, packet_uid
        )
        toa_s = max(float(getattr(packet, "toa", 0.0) or 0.0), 0.0)
        observed_tx_airtime_s = toa_s
        sim_time_s = self._safe_float(self.run_metadata.get("sim_time"))
        if event_time is not None and sim_time_s is not None:
            observed_tx_airtime_s = min(
                toa_s,
                max(sim_time_s - event_time, 0.0),
            )
        if self._is_original_uplink(packet):
            self._register_offered_packet(packet, offered_time=event_time)
            self.record_packet_fate_state(
                packet,
                state="TX_IN_FLIGHT",
                entity_id=node_id,
                time=event_time,
            )
        elif packet_uid in self.packet_fates:
            self.record_packet_fate_state(
                packet,
                state="FORWARD_TX",
                entity_id=node_id,
                time=event_time,
            )

        event_record = {
            # -------------------------------------------------
            # LoRaPhyMAC frame annotation
            # -------------------------------------------------
            "mtype": getattr(packet, "mtype", None),
            "packet_uid": packet_uid,
            "src": packet.src_id,
            "dst_id": getattr(packet, "dst_id", None),
            "current_tx": packet.current_tx,
            "next_hop": packet.next_hop,
            "hop_index": getattr(packet, "hop_index", None),
            "hop_limit": getattr(packet, "hop_limit", None),
            "hop_count": max(0, len(getattr(packet, "path", [])) - 1),
            "path": "->".join(getattr(packet, "path", [])),
            "fctrl": getattr(packet, "fctrl", None),
            "fcnt": getattr(packet, "fcnt", None),
            "fport": getattr(packet, "fport", None),
            "fopts_len": len(getattr(packet, "fopts", b"") or b""),
            "control_cid": self._downlink_control_cid(packet),

            # -------------------------------------------------
            # Event core
            # -------------------------------------------------
            "event": "TX",
            "node": node_id,
            "ch": packet.ch_id,
            "sf": packet.sf,
            "bw_hz": packet.bw_hz,
            "ptx": packet.ptx_dbm,
            "toa_s": toa_s,
            "observed_tx_airtime_s": observed_tx_airtime_s,

            # Backward-compatible event timestamp.
            "time": event_time,

            # Explicit timing metadata.
            **self._event_timing_fields(packet, event_time=event_time),
        }
        if self.is_summary_profile:
            self._record_frame_event_summary(event="TX", packet=packet)
            self._record_summary_tx(
                packet, node_id, event_time, toa_s, observed_tx_airtime_s
            )
        else:
            self.records.append(event_record)
        self._record_feedback_physical_tx(
            packet, node_id, event_time, observed_tx_airtime_s
        )

        if self._is_apc_feedback_packet(packet):
            self.record_apc_feedback_event(
                event="sent",
                time=event_time,
                packet=packet,
            )
        self._record_wang_carried_reports(
            packet,
            event="forwarded",
            time=event_time,
            receiver_id=getattr(packet, "next_hop", None),
        )
        if self._is_wang_power_command_packet(packet):
            self.record_wang_power_command_event(
                event="command_sent",
                time=event_time,
                packet=packet,
                receiver_id=getattr(packet, "next_hop", None),
            )
        return stale_tpi_snapshot

    def record_rx(self, packet: Packet, node_id: str):
        event_time = self._safe_float(getattr(packet, "rx_time", None))
        packet_uid = self.ensure_packet_uid(packet)
        attempt = self._feedback_attempt_for_packet(packet)
        if (
            attempt is not None
            and self._feedback_robustness_enabled()
            and str(attempt["target_transmitter_id"]) == str(node_id)
        ):
            attempt["native_reception_status"] = "success"
            attempt["logical_delivery_status"] = "delivered"
        self.record_packet_fate_state(
            packet,
            state="RX_DECODED",
            entity_id=node_id,
            time=event_time,
        )

        event_record = {
            "mtype": getattr(packet, "mtype", None),
            "packet_uid": packet_uid,
            "src": packet.src_id,
            "dst_id": getattr(packet, "dst_id", None),
            "current_tx": packet.current_tx,
            "next_hop": getattr(packet, "next_hop", None),
            "hop_index": getattr(packet, "hop_index", None),
            "hop_limit": getattr(packet, "hop_limit", None),
            "hop_count": max(0, len(getattr(packet, "path", [])) - 1),
            "path": "->".join(getattr(packet, "path", [])),
            "fctrl": getattr(packet, "fctrl", None),
            "fcnt": getattr(packet, "fcnt", None),
            "fport": getattr(packet, "fport", None),
            "fopts_len": len(getattr(packet, "fopts", b"") or b""),
            "control_cid": self._downlink_control_cid(packet),

            "event": "RX",
            "node": node_id,
            "ch": packet.ch_id,
            "sf": packet.sf,
            "bw_hz": packet.bw_hz,
            "rssi": getattr(packet, "rssi", None),
            "snr": getattr(packet, "snr", None),
            "margin": getattr(packet, "margin", None),
            "decoded": getattr(packet, "decoded", None),
            "collided": getattr(packet, "collided", None),

            # Backward-compatible event timestamp.
            "time": event_time,

            # Explicit timing metadata.
            **self._event_timing_fields(packet, event_time=event_time),
        }
        if self.is_summary_profile:
            self._record_frame_event_summary(event="RX", packet=packet)
        else:
            self.records.append(event_record)

        if self._is_apc_feedback_packet(packet):
            self.record_apc_feedback_event(
                event="received",
                time=event_time,
                packet=packet,
            )

    def record_drop(self, packet: Packet, reason: str, node_id: str):
        """
        Record a packet drop.

        event_time prioritizes an explicit packet.drop_time. If the runtime
        module did not set drop_time, a compatibility fallback is used for the
        event timestamp, while the exported drop_time column remains None.
        This makes missing drop-time instrumentation visible in the event log.
        """
        explicit_drop_time = self._safe_float(
            getattr(packet, "drop_time", None)
        )

        event_time = explicit_drop_time
        if event_time is None:
            event_time = self._safe_float(getattr(packet, "rx_time", None))
        if event_time is None:
            event_time = self._safe_float(getattr(packet, "t_arrive", None))
        if event_time is None:
            event_time = self._safe_float(getattr(packet, "tx_end", None))

        packet_uid = self.ensure_packet_uid(packet)
        attempt = self._feedback_attempt_for_packet(packet)
        if (
            attempt is not None
            and self._feedback_robustness_enabled()
            and str(attempt["target_transmitter_id"]) == str(node_id)
        ):
            if attempt["native_reception_status"] == "pending":
                attempt["native_reception_status"] = "failed"
                attempt["native_failure_reason"] = str(reason)
                attempt["logical_delivery_status"] = "native_failed"
            elif attempt["native_reception_status"] == "success":
                attempt["logical_delivery_status"] = "native_success_but_not_applied"
                attempt["logical_failure_reason"] = str(reason)
        self.record_packet_fate_state(
            packet,
            state="TERMINAL_DROP",
            entity_id=node_id,
            time=event_time,
            reason=reason,
        )

        event_record = {
            "mtype": getattr(packet, "mtype", None),
            "packet_uid": packet_uid,
            "src": packet.src_id,
            "dst_id": getattr(packet, "dst_id", None),
            "current_tx": getattr(packet, "current_tx", None),
            "next_hop": getattr(packet, "next_hop", None),
            "hop_index": getattr(packet, "hop_index", None),
            "hop_limit": getattr(packet, "hop_limit", None),
            "hop_count": max(0, len(getattr(packet, "path", [])) - 1),
            "path": "->".join(getattr(packet, "path", [])),
            "fctrl": getattr(packet, "fctrl", None),
            "fcnt": getattr(packet, "fcnt", None),
            "fport": getattr(packet, "fport", None),
            "fopts_len": len(getattr(packet, "fopts", b"") or b""),
            "control_cid": self._downlink_control_cid(packet),

            "event": "DROP",
            "node": node_id,
            "decoded": getattr(packet, "decoded", False),
            "collided": getattr(packet, "collided", False),
            "reason": reason,
            "ch": getattr(packet, "ch_id", None),
            "sf": getattr(packet, "sf", None),
            "bw_hz": getattr(packet, "bw_hz", None),
            "rssi": getattr(packet, "rssi", None),
            "snr": getattr(packet, "snr", None),
            "margin": getattr(packet, "margin", None),

            # Backward-compatible event timestamp.
            "time": event_time,

            # Explicit timing metadata.
            **self._event_timing_fields(packet, event_time=event_time),
        }
        if self.is_summary_profile:
            self._record_frame_event_summary(
                event="DROP", packet=packet, reason=reason
            )
        else:
            self.records.append(event_record)

        if self._is_apc_feedback_packet(packet):
            lifecycle_event = (
                "collided"
                if bool(getattr(packet, "collided", False)) or str(reason) == "collision"
                else "dropped"
            )
            self.record_apc_feedback_event(
                event=lifecycle_event,
                time=event_time,
                packet=packet,
                reason=reason,
            )
        self._record_wang_carried_reports(
            packet,
            event=("overflow" if str(reason) == "forwarding_queue_overflow" else "lost"),
            time=event_time,
            receiver_id=node_id,
            reason=reason,
            terminal=True,
        )
        if self._is_wang_power_command_packet(packet):
            self.record_wang_power_command_event(
                event="command_dropped",
                time=event_time,
                packet=packet,
                receiver_id=node_id,
                reason=reason,
            )

    def to_dataframe(self) -> pd.DataFrame:
        if self.is_summary_profile:
            return pd.DataFrame()
        return pd.DataFrame(self.records)

    @staticmethod
    def _is_apc_feedback_packet(packet: Packet) -> bool:
        """Return whether a packet is a physical APC feedback frame."""
        try:
            is_downlink = int(getattr(packet, "mtype", -1)) in (
                int(MType.DATA_DOWN),
                int(MType.DATA_DOWN_FWD),
            )
            return is_downlink and getattr(packet, "apc_info", None) is not None
        except (TypeError, ValueError):
            return False

    def allocate_apc_feedback_id(self) -> str:
        """Allocate a deterministic run-local identity for one APC decision."""
        self.apc_feedback_sequence += 1
        run_id = self.run_metadata.get("run_id") or "run"
        return f"{run_id}:apc:{self.apc_feedback_sequence:06d}"

    def record_apc_feedback_event(
        self,
        *,
        event,
        time,
        packet=None,
        apc_info=None,
        feedback_id=None,
        mtype=None,
        applied_tx_power_index=None,
        old_ptx_dbm=None,
        applied_ptx_dbm=None,
        stale_tpi=None,
        apc_rx_margin=None,
        apc_rx_rssi=None,
        apc_rx_snr=None,
        reason=None,
    ):
        """Record only observed APC lifecycle states; no ACK is inferred."""
        packet_info = (
            getattr(packet, "apc_info", None)
            if packet is not None
            else None
        )
        info = dict(apc_info or packet_info or {})
        context = dict(info.get("context", {}) or {})

        if feedback_id is None:
            feedback_id = info.get("feedback_id")
        if feedback_id is None:
            raise ValueError("APC feedback lifecycle event requires feedback_id")

        timestamp_s = self._safe_float(time)
        robustness = self._feedback_robustness_enabled()
        target_id = str(context.get("target_id", ""))
        if robustness and event in {"generated", "applied"}:
            self._next_feedback_tpi_event_sequence()
        if robustness and event == "generated":
            desired = context.get("tx_power_index")
            prior = self.feedback_tpi_desired.get(target_id)
            state_changed = prior is None or int(prior) != int(desired)
            feedback_key = str(feedback_id)
            new_decision = feedback_key not in self.feedback_tpi_version_by_feedback_id
            scope_id = self._feedback_tpi_scope_id(context)
            if target_id:
                self.feedback_tpi_decision_seen_by_target.add(target_id)
            if new_decision:
                version = None
                if scope_id is not None:
                    scopes = self.feedback_tpi_scopes_by_target[target_id]
                    scopes.add(scope_id)
                    version = self.feedback_tpi_latest_decided_by_scope.get(scope_id, 0) + 1
                    self.feedback_tpi_latest_decided_by_scope[scope_id] = version
                    # Version zero represents the initialized operational TPI.
                    # A generated but unapplied command is therefore stale
                    # against the known initial state, never implicitly current.
                    if target_id in self.feedback_tpi_registered_transmitters:
                        self.feedback_tpi_applied_by_scope.setdefault(scope_id, 0)
                self.feedback_tpi_scope_by_feedback_id[feedback_key] = scope_id
                self.feedback_tpi_version_by_feedback_id[feedback_key] = version
            else:
                scope_id = self.feedback_tpi_scope_by_feedback_id[feedback_key]
                version = self.feedback_tpi_version_by_feedback_id[feedback_key]

            if desired is not None:
                self.feedback_tpi_desired[target_id] = int(desired)
            if version is not None:
                self.feedback_tpi_version[target_id] = version
                context["tpi_version"] = version
                if apc_info is not None:
                    apc_info.setdefault("context", {})["tpi_version"] = version
            if new_decision:
                self.feedback_transactions[feedback_key] = {
                    "run_id": self.run_metadata.get("run_id"),
                    "decision_id": feedback_key, "feedback_id": feedback_key,
                    "target_transmitter_id": target_id,
                    "controller_id": str(context.get("controller_id", "")),
                    "decision_timestamp_s": timestamp_s,
                    "desired_tpi": desired, "tpi_version": version,
                    "tpi_state_changed": bool(state_changed),
                    "physical_attempts": 0, "native_successful_receptions": 0,
                    "synthetic_erasures": 0, "application_timestamp_s": None,
                    "application_latency_s": None, "retry_exhausted": False,
                    "status": "pending",
                }
                self.feedback_latest_transaction_by_target[target_id] = feedback_key
                for previous_id, previous in self.feedback_transactions.items():
                    if state_changed and previous_id != feedback_key and previous["target_transmitter_id"] == target_id and previous.get("first_entry_timestamp_s") is None:
                        previous["status"] = "superseded"
        elif robustness and event in {"retry_generated", "retry_exhausted"}:
            version = self.feedback_tpi_version_by_feedback_id.get(str(feedback_id))
            if version is not None:
                context["tpi_version"] = version
                if apc_info is not None:
                    apc_info.setdefault("context", {})["tpi_version"] = version
            transaction = self.feedback_transactions.get(str(feedback_id))
            if transaction is not None and event == "retry_exhausted":
                transaction["retry_exhausted"] = True
                transaction["status"] = "retry_exhausted"
        elif robustness and event == "received":
            transaction = self.feedback_transactions.get(str(feedback_id))
            if transaction is not None:
                transaction["status"] = "logically_delivered"
        elif robustness and event == "applied":
            target = target_id or str(context.get("target_id", ""))
            feedback_key = str(feedback_id)
            scope_id = self.feedback_tpi_scope_by_feedback_id.get(
                feedback_key, self._feedback_tpi_scope_id(context)
            )
            version = context.get(
                "tpi_version", self.feedback_tpi_version_by_feedback_id.get(feedback_key)
            )
            if scope_id is not None and version is not None:
                applied_version = int(version)
                self.feedback_tpi_applied_by_scope[scope_id] = applied_version
                self.feedback_tpi_applied[target] = applied_version
            transaction = self.feedback_transactions.get(str(feedback_id))
            if transaction is not None:
                transaction["application_timestamp_s"] = self._safe_float(time)
                transaction["application_latency_s"] = max(
                    0.0,
                    float(time) - float(transaction["decision_timestamp_s"]),
                )
                transaction["status"] = "applied"
            attempt = self._feedback_attempt_for_packet(packet) if packet is not None else None
            if attempt is not None:
                attempt["tpi_application_status"] = "applied"
        elif robustness and event in {"collided", "dropped"}:
            transaction = self.feedback_transactions.get(str(feedback_id))
            if transaction is not None and transaction.get("application_timestamp_s") is None:
                transaction["status"] = "native_failed_or_rejected"

        decision_time_s = self._safe_float(info.get("decision_time"))
        feedback_age_s = None
        if timestamp_s is not None and decision_time_s is not None:
            feedback_age_s = max(timestamp_s - decision_time_s, 0.0)

        packet_mtype = getattr(packet, "mtype", None) if packet is not None else None
        if mtype is None:
            mtype = packet_mtype
        mtype_value = None if mtype is None else int(mtype)

        packet_uid = None
        if packet is not None:
            packet_uid = getattr(packet, "packet_uid", None)
        retry_timeout_s = self._safe_float(info.get("retry_timeout_s"))

        self.apc_feedback_event_sequence += 1
        self.apc_feedback_lifecycle.append(
            {
                "run_id": self.run_metadata.get("run_id"),
                "event_sequence": self.apc_feedback_event_sequence,
                "feedback_id": feedback_id,
                "packet_uid": packet_uid,
                "event": str(event),
                "timestamp_s": timestamp_s,
                "decision_time_s": decision_time_s,
                "feedback_age_s": feedback_age_s,
                "retry_timeout_s": retry_timeout_s,
                "stale_tpi": stale_tpi,
                "controller_id": context.get("controller_id"),
                "controller_type": context.get("controller_type"),
                "target_id": context.get("target_id"),
                "target_type": context.get("target_type"),
                "src_id": info.get("src_id"),
                "hop_index": info.get("hop_index"),
                "control_scope": context.get("control_scope"),
                "mtype": mtype_value,
                "mtype_name": (
                    None
                    if mtype_value is None
                    else MType(mtype_value).name
                ),
                "channel": (
                    getattr(packet, "ch_id", info.get("uplink_ch_id"))
                    if packet is not None
                    else info.get("uplink_ch_id")
                ),
                "sf": (
                    getattr(packet, "sf", info.get("uplink_sf"))
                    if packet is not None
                    else info.get("uplink_sf")
                ),
                "bw_hz": (
                    getattr(packet, "bw_hz", info.get("uplink_bw_hz"))
                    if packet is not None
                    else info.get("uplink_bw_hz")
                ),
                "control_airtime_s": (
                    self._safe_float(getattr(packet, "toa", None))
                    if packet is not None
                    else None
                ),
                "old_tx_power_index": context.get("old_tx_power_index"),
                "decision_tx_power_index": context.get("tx_power_index"),
                "applied_tx_power_index": applied_tx_power_index,
                "old_ptx_dbm": old_ptx_dbm,
                "applied_ptx_dbm": applied_ptx_dbm,
                "decision_margin_db": info.get("decision_margin"),
                "decision_rssi_dbm": info.get("decision_rssi"),
                "decision_snr_db": info.get("decision_snr"),
                "apc_rx_margin_db": (
                    apc_rx_margin
                    if apc_rx_margin is not None
                    else getattr(packet, "margin", None)
                ),
                "apc_rx_rssi_dbm": (
                    apc_rx_rssi
                    if apc_rx_rssi is not None
                    else getattr(packet, "rssi", None)
                ),
                "apc_rx_snr_db": (
                    apc_rx_snr
                    if apc_rx_snr is not None
                    else getattr(packet, "snr", None)
                ),
                "decoded": getattr(packet, "decoded", None),
                "collided": getattr(packet, "collided", None),
                "reason": (
                    reason
                    if reason is not None
                    else getattr(packet, "reason", None)
                ),
            }
        )

    def get_apc_feedback_lifecycle_df(self) -> pd.DataFrame:
        return _build_apc_feedback_lifecycle_df(
            self.apc_feedback_lifecycle,
            self.run_metadata,
        )

    def record_wang_snr_report_event(
        self,
        *,
        event,
        time,
        packet,
        report=None,
        receiver_id=None,
        reason=None,
    ):
        """Record an observable Wang SNR-report lifecycle transition."""
        event = str(event)
        report_id = getattr(report, "report_id", None)
        if event in {"lost", "overflow"} and report_id is not None:
            if report_id in self.wang_snr_terminal_report_ids:
                return
            self.wang_snr_terminal_report_ids.add(report_id)
        self.wang_snr_report_event_sequence += 1
        timestamp = self._safe_float(time)
        observation_time = self._safe_float(
            getattr(report, "observation_time_s", None)
        )
        age = None
        if timestamp is not None and observation_time is not None:
            age = max(0.0, timestamp - observation_time)
        self.wang_snr_report_lifecycle.append(
            {
                "event_sequence": self.wang_snr_report_event_sequence,
                "report_id": report_id,
                "packet_uid": (
                    getattr(report, "packet_uid", None)
                    if report is not None
                    else getattr(packet, "packet_uid", None)
                ),
                "event": event,
                "timestamp_s": timestamp,
                "observation_time_s": observation_time,
                "report_age_s": age,
                "tx_id": getattr(report, "tx_id", None),
                "rx_id": getattr(report, "rx_id", None),
                "hop_index": getattr(report, "hop_index", None),
                "snr_db": getattr(report, "snr_db", None),
                "observed_ptx_dbm": getattr(report, "observed_ptx_dbm", None),
                "ptx_epoch": getattr(report, "ptx_epoch", None),
                "last_applied_command_seq": getattr(
                    report, "last_applied_command_seq", None
                ),
                "ptx_provenance": (
                    "transmitter_declared_frame" if report is not None else None
                ),
                "carrier_hop_index": getattr(packet, "hop_index", None),
                "receiver_id": receiver_id,
                "fopts_len": len(getattr(packet, "fopts", b"") or b""),
                "reason": reason,
            }
        )

    def _record_wang_carried_reports(
        self,
        packet,
        *,
        event,
        time,
        receiver_id=None,
        reason=None,
        terminal=False,
    ):
        """Decode only physically carried Wang reports for TX/drop evidence."""
        raw = bytes(getattr(packet, "fopts", b"") or b"")
        if not raw or raw[0] != int(CID.WANG_SNR_REPORT):
            return
        try:
            from loraphymac.runtime.wang_snr_reporting import decode_reports

            reports = decode_reports(
                packet,
                routing_map=getattr(self, "wang_routing_map", None),
            )
        except ValueError as exc:
            self.record_wang_snr_report_event(
                event="invalid",
                time=time,
                packet=packet,
                receiver_id=receiver_id,
                reason=str(exc),
            )
            return
        for report in reports:
            if terminal and report.report_id in self.wang_snr_terminal_report_ids:
                continue
            self.record_wang_snr_report_event(
                event=event,
                time=time,
                packet=packet,
                report=report,
                receiver_id=receiver_id,
                reason=reason,
            )

    def get_wang_snr_report_lifecycle_df(self) -> pd.DataFrame:
        return _build_wang_snr_report_lifecycle_df(
            self.wang_snr_report_lifecycle,
            self.run_metadata,
        )

    @staticmethod
    def _is_wang_power_command_packet(packet) -> bool:
        raw = bytes(getattr(packet, "fopts", b"") or b"")
        return bool(raw) and raw[0] == int(CID.WANG_POWER_COMMAND)

    def record_wang_power_command_event(
        self,
        *,
        event,
        time,
        packet=None,
        command=None,
        receiver_id=None,
        reason=None,
    ):
        """Record a Wang command event without feeding simulator behavior."""
        if command is None and packet is not None:
            try:
                from loraphymac.runtime.wang_power_commands import (
                    decode_wang_power_command,
                )

                command = decode_wang_power_command(packet)
            except ValueError:
                command = None
        self.wang_power_command_event_sequence += 1
        self.wang_power_command_lifecycle.append(
            {
                "event_sequence": self.wang_power_command_event_sequence,
                "command_id": getattr(command, "command_id", None),
                "event": str(event),
                "timestamp_s": self._safe_float(time),
                "target_id": getattr(command, "target_id", None),
                "command_seq": getattr(command, "command_seq", None),
                "window_index": getattr(command, "window_index", None),
                "expected_ptx_dbm": getattr(command, "expected_ptx_dbm", None),
                "expected_epoch": getattr(command, "expected_epoch", None),
                "target_ptx_dbm": getattr(command, "target_ptx_dbm", None),
                "issued_at_s": getattr(command, "issued_at_s", None),
                "expiry_s": getattr(command, "expiry_s", None),
                "packet_uid": getattr(packet, "packet_uid", None),
                "current_tx": getattr(packet, "current_tx", None),
                "next_hop": getattr(packet, "next_hop", None),
                "receiver_id": receiver_id,
                "hop_index": getattr(packet, "hop_index", None),
                "mtype": getattr(packet, "mtype", None),
                "channel": getattr(packet, "ch_id", None),
                "sf": getattr(packet, "sf", None),
                "bw_hz": getattr(packet, "bw_hz", None),
                "fopts_len": len(getattr(packet, "fopts", b"") or b""),
                "control_airtime_s": getattr(packet, "toa", None),
                "reason": reason,
            }
        )

    def get_wang_power_command_lifecycle_df(self) -> pd.DataFrame:
        return _build_wang_power_command_lifecycle_df(
            self.wang_power_command_lifecycle,
            self.run_metadata,
        )

    def get_tx_airtime_by_entity_mtype_ptx_channel_df(
        self,
        nodes,
    ) -> pd.DataFrame:
        if self.is_summary_profile:
            rows = []
            entity_recorded = {
                str(getattr(node, "id", node)): float(
                    getattr(node, "total_tx_time_s", 0.0)
                )
                for node in nodes
            }
            for key, (count, total_airtime, _, _, _) in self.summary_tx_airtime.items():
                entity_id, entity_type, traffic_class, mtype, ptx, channel, sf, bw_hz = key
                rows.append({
                    "run_id": self.run_metadata.get("run_id"),
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "traffic_class": traffic_class,
                    "mtype": mtype,
                    "mtype_name": "LEGACY" if mtype is None else mtype_name(int(mtype)),
                    "ptx_dbm": ptx,
                    "channel": channel,
                    "sf": sf,
                    "bw_hz": bw_hz,
                    "tx_packet_count": count,
                    "total_airtime_s": total_airtime,
                })
            sort_columns = [
                "entity_id", "entity_type", "traffic_class", "mtype",
                "mtype_name", "ptx_dbm", "channel", "sf", "bw_hz",
            ]
            frame = pd.DataFrame(rows, columns=TX_AIRTIME_COLUMNS).sort_values(
                sort_columns, kind="stable", na_position="last"
            ).reset_index(drop=True)
            by_entity = defaultdict(lambda: [0.0, 0.0])
            for row in frame.itertuples(index=False):
                entity_id = str(row.entity_id)
                by_entity[entity_id][0], by_entity[entity_id][1] = self._kahan_add(
                    by_entity[entity_id][0],
                    by_entity[entity_id][1],
                    float(row.total_airtime_s),
                )
            for row in rows:
                row["entity_grouped_airtime_s"] = by_entity[row["entity_id"]][0]
                recorded = entity_recorded.get(row["entity_id"])
                row["entity_recorded_total_tx_time_s"] = recorded
                row["airtime_consistency_delta_s"] = (
                    row["entity_grouped_airtime_s"] - recorded
                    if recorded is not None
                    else None
                )
                row["airtime_consistent"] = (
                    abs(row["airtime_consistency_delta_s"]) <= 1e-9
                    if row["airtime_consistency_delta_s"] is not None
                    else False
                )
            frame = pd.DataFrame(rows, columns=TX_AIRTIME_COLUMNS)
            return frame.sort_values(
                sort_columns, kind="stable", na_position="last"
            ).reset_index(drop=True)
        return _build_tx_airtime_df(
            self.records,
            nodes,
            self.run_metadata,
        )

    def get_summary_node_tx_duty_cycle_df(self, nodes):
        sim_time_s = float(self.run_metadata.get("sim_time", 0.0) or 0.0)
        rows = []
        for entity in nodes:
            entity_id = str(getattr(entity, "id", ""))
            if entity_id == "GW" or entity_id.startswith(("GW", "R")):
                continue
            airtime_s = float(self.summary_node_tx_airtime.get(entity_id, [0.0])[0])
            rows.append({
                "run_id": self.run_metadata.get("run_id"),
                "entity_id": entity_id,
                "entity_type": "node",
                "sim_time_s": sim_time_s,
                "observed_tx_airtime_s": airtime_s,
                "tx_duty_fraction": airtime_s / sim_time_s if sim_time_s > 0 else 0.0,
                "tx_duty_cycle_percent": airtime_s / sim_time_s * 100.0 if sim_time_s > 0 else 0.0,
                "measurement_scope": "physical_tx_airtime_over_simulation_horizon",
            })
        return pd.DataFrame(rows, columns=NODE_TX_DUTY_CYCLE_COLUMNS)

    def get_summary_channel_occupancy_df(self):
        sim_time_s = float(self.run_metadata.get("sim_time", 0.0) or 0.0)
        rows = []
        for channel in sorted(self.summary_channel_airtime):
            sum_offered, sum_observed = self.summary_channel_airtime[channel][:2]
            start, end, busy = self.summary_channel_union[channel]
            if start is not None:
                busy += end - start
            rows.append({
                "run_id": self.run_metadata.get("run_id"),
                "channel": channel,
                "sim_time_s": sim_time_s,
                "sum_offered_tx_airtime_s": sum_offered,
                "sum_observed_tx_airtime_s": sum_observed,
                "channel_busy_union_s": busy,
                "channel_busy_fraction": busy / sim_time_s if sim_time_s > 0 else 0.0,
                "channel_busy_percent": busy / sim_time_s * 100.0 if sim_time_s > 0 else 0.0,
                "occupancy_method": "union_of_simulated_tx_intervals",
            })
        return pd.DataFrame(rows, columns=CHANNEL_OCCUPANCY_COLUMNS)

    # =========================================================
    # DEMODULATOR POOL INSTRUMENTATION
    # =========================================================
    def record_demodulator_event(self, **event):
        """Store one ACQUIRE/REJECT/RELEASE occupancy transition."""
        if self.is_summary_profile:
            receiver = event.get("receiver_id")
            aggregate = self.summary_demodulator.setdefault(
                receiver,
                {
                    "acquire_attempt_count": 0, "acquire_success_count": 0,
                    "release_count": 0, "reject_count": 0,
                    "demod_busy_count": 0, "unsupported_sf_count": 0,
                    "duplicate_acquire_count": 0, "release_missing_count": 0,
                    "pool_a_acquire_count": 0, "pool_b_acquire_count": 0,
                    "capacity_a": int(event.get("capacity_a", 0) or 0),
                    "capacity_b": int(event.get("capacity_b", 0) or 0),
                    "capacity_total": int(event.get("capacity_total", 0) or 0),
                    "peak_used_a": 0, "peak_used_b": 0,
                    "peak_used_total": 0, "used_a_at_last_event": 0,
                    "used_b_at_last_event": 0, "last_time": -float("inf"),
                },
            )
            kind = str(event.get("event", ""))
            reason = str(event.get("reason", ""))
            if kind in {"ACQUIRE", "REJECT"}:
                aggregate["acquire_attempt_count"] += 1
            if kind == "ACQUIRE":
                aggregate["acquire_success_count"] += 1
                pool = str(event.get("pool", ""))
                if pool in {"A", "B"}:
                    aggregate[f"pool_{pool.lower()}_acquire_count"] += 1
            if kind == "RELEASE":
                aggregate["release_count"] += 1
            if kind == "REJECT":
                aggregate["reject_count"] += 1
                if reason == "demod_busy":
                    aggregate["demod_busy_count"] += 1
                elif reason == "demod_unsupported_sf":
                    aggregate["unsupported_sf_count"] += 1
                elif reason == "demod_duplicate_acquire":
                    aggregate["duplicate_acquire_count"] += 1
            if kind == "RELEASE_MISSING":
                aggregate["release_missing_count"] += 1
            for field, event_field in (
                ("used_a", "used_a_after"),
                ("used_b", "used_b_after"),
                ("used_total", "used_total_after"),
            ):
                value = event.get(event_field)
                if value is not None:
                    aggregate[f"peak_{field}"] = max(
                        aggregate[f"peak_{field}"], int(value)
                    )
                    if event.get("time", -float("inf")) >= aggregate["last_time"]:
                        if field != "used_total":
                            aggregate[f"{field}_at_last_event"] = int(value)
            aggregate["last_time"] = max(
                aggregate["last_time"], float(event.get("time", -float("inf")))
            )
            return
        self.demodulator_events.append(dict(event))

    def get_demodulator_event_log_df(self) -> pd.DataFrame:
        """Packet-level modem-pool occupancy transitions."""
        if self.is_summary_profile:
            return pd.DataFrame()
        return _build_demodulator_event_log_df(self.demodulator_events)

    def get_demodulator_pool_summary_df(self) -> pd.DataFrame:
        """Aggregate occupancy and rejection evidence per receiver."""
        if self.is_summary_profile:
            rows = []
            for receiver, counts in self.summary_demodulator.items():
                rows.append({
                    "receiver_id": receiver,
                    "timing_model": "rx_start_to_rx_end_v2",
                    **{key: value for key, value in counts.items() if key != "last_time"},
                })
            columns = [
                "receiver_id", "timing_model", "capacity_a", "capacity_b",
                "capacity_total", "acquire_attempt_count", "acquire_success_count",
                "release_count", "reject_count", "demod_busy_count",
                "unsupported_sf_count", "duplicate_acquire_count",
                "release_missing_count", "pool_a_acquire_count", "pool_b_acquire_count",
                "peak_used_a", "peak_used_b", "peak_used_total",
                "used_a_at_last_event", "used_b_at_last_event",
            ]
            if not rows:
                return pd.DataFrame(columns=columns)
            return (
                pd.DataFrame(rows)
                .reindex(columns=columns)
                .sort_values("receiver_id", kind="stable")
                .reset_index(drop=True)
            )
        return _build_demodulator_pool_summary_df(
            self.get_demodulator_event_log_df()
        )

    # =========================================================
    # COLLISION / CAPTURE DECISION INSTRUMENTATION
    # =========================================================
    def record_collision_decision(self, **event):
        """Store one desired-packet decision from a receiver RF window."""
        if self.is_summary_profile:
            keep = {
                "collision_model_version", "spectral_model_version", "collision_mode",
                "sf_interference_mode", "sir_matrix_version", "resolution_id",
                "receiver_id", "hop_index", "mtype", "ch_id", "sf", "outcome",
                "survived", "direct_interferer_count", "contender_count",
                "capture_delta_db", "min_pair_margin_db", "max_pair_overlap_s",
            }
            self.collision_decision_records.append(
                {key: value for key, value in event.items() if key in keep}
            )
        else:
            self.collision_decision_records.append(dict(event))

    def record_collision_interference_edge(self, **event):
        """Store one directional desired-to-interferer RF edge."""
        if self.is_summary_profile:
            keep = {
                "desired_sf", "interferer_sf", "spectral_overlap_fraction",
                "pair_pass", "dominant_interferer",
            }
            self.collision_interference_edge_records.append(
                {key: value for key, value in event.items() if key in keep}
            )
        else:
            self.collision_interference_edge_records.append(dict(event))

    def get_collision_interference_edges_df(self) -> pd.DataFrame:
        """Return schema-complete pair-level RF interference evidence."""
        return _build_collision_interference_edges_df(
            self.collision_interference_edge_records
        )

    def get_collision_event_log_df(self) -> pd.DataFrame:
        """Return stable, schema-complete packet-level collision evidence."""
        return _build_collision_event_log_df(self.collision_decision_records)

    def get_collision_summary_df(self) -> pd.DataFrame:
        """Decision-centric collision/capture metrics for paper analysis.

        Rows are stratified by model, receiver, hop, traffic type, channel,
        and SF. ``overlap_pair_equivalent_count`` is half the directed edge
        count inside that stratum; the core-summary pair count is the exact
        run-wide value and should be used when pairs cross strata.
        """
        return _build_collision_summary_df(self.get_collision_event_log_df())

    # =========================================================
    # 4. RELAY RX STATISTICS
    # =========================================================
    def _ensure_relay(self, relay_id):
        if relay_id not in self.relay_stats:
            self.relay_stats[relay_id] = {
                "rx_arrival_count": 0,
                "rx_not_feasible": 0,
                "rx_feasible": 0,
                "rx_survivor": 0,
                "rx_collision": 0,
                "rx_demod_acquired": 0,
                "demodulator_drop_count": 0,
                # -------------------------------------------------
                # Half-duplex drop
                # -------------------------------------------------
                # Packets arriving while a half-duplex relay/gateway is transmitting.
                "half_duplex_drop_count": 0,
                "rx_interrupted_by_tx_count": 0,
                
                "rx_decoded": 0,
                "rx_gateway": 0,
                "rx_forwarded": 0,
                "tx_queue_arrival_count": 0,
                "tx_queue_enqueue_attempt_count": 0,
                "tx_queue_accepted_count": 0,
                "tx_queue_overflow_drop_count": 0,
                "tx_queue_internal_drop_count": 0,
                "tx_done": 0,
            }

    def record_relay_rx(self, relay_id, stage):
        self._ensure_relay(relay_id)

        # Defensive guard for a new or misspelled stage from a runtime module.
        if stage not in self.relay_stats[relay_id]:
            self.relay_stats[relay_id][stage] = 0

        self.relay_stats[relay_id][stage] += 1

    def get_relay_rx_stats_df(self):
        return _build_relay_rx_stats_df(self.relay_stats)

    # =========================================================
    # 5. RX CHANNEL USAGE
    # =========================================================
    def record_rx_channel_use(self, relay_id, ch_id, toa):
        self.rx_channel_use[relay_id][ch_id] += 1
        self.rx_channel_duration[relay_id][ch_id] += toa

    def get_rx_channel_usage_df(self):
        return _build_rx_channel_usage_df(
            self.rx_channel_use,
            self.rx_channel_duration,
        )


    # =========================================================
    # 6. NODE TX/RX STATISTICS
    # =========================================================
    def _ensure_node(self, node_id):
        if node_id not in self.node_stats:
            self.node_stats[node_id] = {
                "hop_index": None,
                "tx": 0,
                "network_rx_events_by_source": 0,
                "rx_feasible": 0,
                "rx_not_feasible": 0,
                "rx_survivor": 0,
                "rx_collision": 0,
                "rx_demod_acquired": 0,
                "demodulator_drop_count": 0,

                # -------------------------------------------------
                # Half-duplex drop
                # -------------------------------------------------
                # Packets from this node dropped because a half-duplex
                # relay/gateway was transmitting.
                "half_duplex_drop_count": 0,
                "rx_interrupted_by_tx_count": 0,

                "rx_decoded": 0,
                "rx_forwarded": 0,
                "rx_gateway": 0,
                "e2e_delay_sum_s": 0.0,
                "e2e_delay_count": 0,
            }

    def record_node_tx(self, node_id):
        self._ensure_node(node_id)
        self.node_stats[node_id]["tx"] += 1

    def record_node_rx(self, node_id, stage):
        self._ensure_node(node_id)

        # Defensive guard for a new or misspelled stage from a runtime module.
        if stage not in self.node_stats[node_id]:
            self.node_stats[node_id][stage] = 0

        self.node_stats[node_id][stage] += 1

    def record_node_hop_index(self, node_id, hop_index):
        self._ensure_node(node_id)
        self.node_stats[node_id]["hop_index"] = hop_index

    def record_node_e2e_delay(self, node_id, delay_s):
        self._ensure_node(node_id)
        self.node_stats[node_id]["e2e_delay_sum_s"] += delay_s
        self.node_stats[node_id]["e2e_delay_count"] += 1

    def get_node_rx_matrix_df(self):
        return _build_node_rx_matrix_df(self.node_stats)

    def get_hop_summary_df(self):
        return _build_hop_summary_df(self.get_node_rx_matrix_df())

    # =========================================================
    # 7. ENERGY STATISTICS
    # =========================================================
    def _continuous_rx_energy_mJ(self, n):
        """
        Continuous RX model for relay/gateway.
    
        Half-duplex:
            RX is active while not transmitting.
            E_rx = V * I_rx * (T_sim - T_tx)
    
        Full-duplex:
            RX remains active throughout the simulation, including during TX.
            E_rx = V * I_rx * T_sim
    
        Returns:
            rx_energy_mJ, rx_time_s, tx_time_s
        """
        return _continuous_rx_energy_mj(
            n,
            self.run_metadata.get("sim_time", 0.0),
            self.run_metadata.get("energy_accounting_mode", "legacy"),
        )


    def get_energy_breakdown_df(self, nodes: list) -> pd.DataFrame:
        """
        Full energy breakdown matrix in Joule (J).

        Internal simulator accumulates energy in mJ,
        but this output is standardized to J for paper/reporting.
        """
        return _build_energy_breakdown_df(
            nodes,
            self.run_metadata,
            self._infer_entity_type,
            self._continuous_rx_energy_mJ,
            self._mj_to_j,
        )

    # =========================================================
    # 8. DUTY CYCLE STATISTICS
    # =========================================================

    def get_relay_duty_cycle_df(self, nodes: list | None = None) -> pd.DataFrame:
        """
        Aggregate duty cycle per relay/gateway.
    
        Formula:
            relay_duty_cycle_percent = total_tx_time_s / sim_time_s * 100
    
        Notes:
        - This is aggregate TX duty cycle per relay/gateway.
        - Per-channel duty cycle is handled by get_per_channel_duty_cycle_df().
        """
    
        return _build_relay_duty_cycle_df(
            nodes,
            self.run_metadata,
            self._infer_entity_type,
        )


    
    def get_per_channel_duty_cycle_df(self, nodes: list, sim_time=None):
        """
        Matrix:
            per_channel_duty_cycle_percent = total_airtime_per_channel / sim_time * 100

        Notes:
        - Uses relay.tx_airtime_per_channel recorded in RelayTX.
        - This is TX channel duty cycle, not RX channel usage.
        """
        return _build_per_channel_duty_cycle_df(
            nodes,
            sim_time,
            self.run_metadata,
            self._infer_entity_type,
        )

    # =========================================================
    # 9. TX Waiting Time (wq)
    # =========================================================
    def record_tx_queue_length(self, relay_id, queue_length, time=None, event=None):
        """
        Record RelayTX waiting-queue length after enqueue/dequeue events.

        queue_length is the number of packets waiting in the shared TX queue,
        not including the packet currently being transmitted by a TX worker.
        """
        try:
            q_len = max(0, int(queue_length))
        except Exception:
            q_len = 0

        try:
            t = float(self.run_metadata.get("sim_time", 0.0))
            # If no explicit time is supplied, do not guess from sim_time.
        except Exception:
            pass

        if time is None:
            # Runtime callers should pass env.now. Fallback to 0.0 so the
            # output remains safe if called manually.
            t = 0.0
        else:
            try:
                t = float(time)
            except Exception:
                t = 0.0

        self.tx_queue_length_samples[relay_id].append({
            "time": t,
            "queue_length": q_len,
            "event": event,
        })

    def record_tx_queue_state(
        self,
        relay_id,
        time,
        waiting_count,
        in_service_count,
        event=None,
    ):
        """Record the complete RelayTX queue-system occupancy state."""
        t = float(time)
        waiting = max(0, int(waiting_count))
        in_service = max(0, int(in_service_count))
        self.record_tx_queue_length(
            relay_id=relay_id,
            queue_length=waiting,
            time=t,
            event=event,
        )
        self.tx_queue_state_samples[relay_id].append({
            "time": t,
            "waiting_count": waiting,
            "in_service_count": in_service,
            "system_count": waiting + in_service,
            "event": event,
        })

    def record_tx_queue_lifecycle(self, record):
        """Store one completed, dropped, or right-censored queue visit."""
        missing = {
            "queue_visit_id", "entity_id", "status",
        }.difference(record)
        if missing:
            raise ValueError(
                "Queue lifecycle record missing fields: "
                f"{sorted(missing)}"
            )
        if self.is_summary_profile:
            compact_columns = {
                "entity_id", "entity_type", "traffic_class", "status",
                "queue_dequeue_time", "tx_start", "fifo_wait_s", "duty_wait_s",
                "half_duplex_wait_s", "airtime_s", "observed_airtime_s",
                "worker_service_s", "time_to_tx_start_s", "system_sojourn_s",
            }
            row = {key: value for key, value in record.items() if key in compact_columns}
            row.setdefault("entity_id", record.get("entity_id"))
            row.setdefault("status", record.get("status"))
        else:
            row = dict(record)
        self.tx_queue_lifecycle_records.append(row)

    def get_tx_queue_lifecycle_df(self):
        return _build_tx_queue_lifecycle_df(self.tx_queue_lifecycle_records)

    def _get_queue_length_metrics(self, relay_id, sim_time_s=None):
        """
        Compute time-weighted average queue length and maximum queue length.

        The queue length is sampled after every enqueue/dequeue event. Between
        two samples, the queue length is assumed constant. This is the correct
        time-weighted metric for queue occupancy.
        """
        if sim_time_s is None:
            sim_time_s = self.run_metadata.get("sim_time", None)
        return _build_queue_length_metrics(
            list(self.tx_queue_length_samples.get(relay_id, []) or []),
            sim_time_s,
        )

    @staticmethod
    def _percentile_ms(delays_s, q):
        return _queue_delay_percentile_ms(delays_s, q)

    def record_tx_queue_waiting_time(self, relay_id, waiting_time_wq_s):
        self.tx_queue_waiting_time[relay_id].append(waiting_time_wq_s)

    def get_tx_queue_waiting_time_df(self):
        return _build_tx_queue_waiting_time_df(self.tx_queue_waiting_time)

    def get_tx_queue_waiting_time_summary_df(self):
        return _build_tx_queue_waiting_time_summary_df(
            self.tx_queue_waiting_time,
            self.relay_stats,
            self.tx_queue_length_samples,
            self.run_metadata,
            self._get_queue_length_metrics,
            self._percentile_ms,
        )


    def _get_tx_queue_state_metrics(self, relay_id, sim_time_s):
        """Integrate waiting, in-service, and total occupancy over time."""
        return _build_tx_queue_state_metrics(
            list(self.tx_queue_state_samples.get(relay_id, []) or []),
            sim_time_s,
        )

    @staticmethod
    def _queue_mean_ms(frame, column):
        return _build_queue_mean_ms(frame, column)

    @staticmethod
    def _queue_percentile_ms(frame, column, quantile):
        return _build_queue_percentile_ms(frame, column, quantile)

    def get_relay_tx_queue_metrics_df(self, nodes, sim_time_s=None):
        """Build measured FIFO/worker queue metrics for RelayTX v2."""
        if sim_time_s is None:
            sim_time_s = self.run_metadata.get("sim_time", 0.0)
        return _build_relay_tx_queue_metrics_df(
            nodes,
            sim_time_s,
            self.get_tx_queue_lifecycle_df(),
            self.relay_stats,
            self.tx_queue_state_samples,
            self._infer_entity_type,
            self._get_tx_queue_state_metrics,
            self._queue_mean_ms,
            self._queue_percentile_ms,
        )

    def get_tx_queue_traffic_class_metrics_df(self):
        """Summarize completed/censored queue visits by traffic class."""
        return _build_tx_queue_traffic_class_metrics_df(
            self.get_tx_queue_lifecycle_df(),
            self._infer_entity_type,
            self._queue_mean_ms,
            self._queue_percentile_ms,
        )

    def get_relay_tx_mgc_queue_metrics_df(self, nodes, sim_time_s=None):
        """Deprecated compatibility alias for queue metrics v2."""
        return self.get_relay_tx_queue_metrics_df(
            nodes,
            sim_time_s=sim_time_s,
        )

    def _get_relay_tx_mgc_queue_metrics_legacy(self, nodes, sim_time_s=None):
        """
        Build M/G/c queue metrics for RelayTX.
    
        Metrics:
        - arrival_count: packets entering the TX queue
        - completed_count: packets completing TX
        - lambda_pkt_s: arrival rate λ
        - mean_service_time_s: mean service time S
        - mu_pkt_s_per_server: service rate μ per TX server
        - rho: traffic intensity λ / (c * μ)
        - rho_percent: TX server utilization percent
        """
    
        if sim_time_s is None:
            sim_time_s = self.run_metadata.get("sim_time", None)
        return _build_relay_tx_mgc_queue_metrics_legacy_df(
            nodes,
            sim_time_s,
            self.relay_stats,
            self.tx_queue_waiting_time,
            self.tx_queue_length_samples,
            self._infer_entity_type,
            self._get_queue_length_metrics,
            self._percentile_ms,
        )


    # =========================================================
    # 10. ADAPTIVE PTX HISTORY
    # =========================================================
    def record_ptx_update(
        self,
        entity_id,
        entity_type,
        old_ptx,
        new_ptx,

        # TxPowerIndex-domain fields.
        old_tx_power_index=None,
        new_tx_power_index=None,
        tx_power_index=None,

        # Backward-compatible legacy fields.
        # margin/rssi/snr are decision/uplink metrics, not APC downlink
        # RX metrics.
        margin=None,
        rssi=None,
        snr=None,

        feedback_from=None,
        src_id=None,
        hop_index=None,
        time=None,

        # New explicit fields
        decision_margin=None,
        decision_rssi=None,
        decision_snr=None,

        apc_rx_margin=None,
        apc_rx_rssi=None,
        apc_rx_snr=None,

        apply_source=None,
        control_mtype=None,
    ):
        """
        Record PTx update.

        old_ptx / new_ptx:
            Physical PTx values in dBm.

        old_tx_power_index / new_tx_power_index / tx_power_index:
            APC/ADR-like state in the TxPowerIndex domain.

        decision_*:
            DATA_UP/DATA_UP_FWD uplink metrics used by the controller
            to calculate a TxPowerIndex change.

        apc_rx_*:
            DATA_DOWN/DATA_DOWN_FWD downlink metrics measured when the
            target receives an APC command.

        Legacy margin/rssi/snr fields remain aliases of decision_* to
        preserve backward-compatible output.
        """

        if decision_margin is None:
            decision_margin = margin
        if decision_rssi is None:
            decision_rssi = rssi
        if decision_snr is None:
            decision_snr = snr

        # Use new_tx_power_index as the final value when tx_power_index
        # is not provided explicitly.
        if tx_power_index is None:
            tx_power_index = new_tx_power_index

        self.summary_ptx_update_count += 1
        self.ptx_history.append({
            "run_id": self.run_metadata.get("run_id"),
            "time": time,

            "entity_id": entity_id,
            "entity_type": entity_type,

            # Physical dBm-domain.
            "old_ptx": old_ptx,
            "new_ptx": new_ptx,

            # APC TxPowerIndex-domain.
            "old_tx_power_index": old_tx_power_index,
            "new_tx_power_index": new_tx_power_index,
            "tx_power_index": tx_power_index,

            # Backward-compatible aliases.
            # These now represent uplink/decision metrics.
            "margin": decision_margin,
            "rssi": decision_rssi,
            "snr": decision_snr,

            # Explicit decision/uplink metrics.
            "decision_margin": decision_margin,
            "decision_rssi": decision_rssi,
            "decision_snr": decision_snr,

            # Explicit physical APC downlink RX metrics.
            "apc_rx_margin": apc_rx_margin,
            "apc_rx_rssi": apc_rx_rssi,
            "apc_rx_snr": apc_rx_snr,

            "feedback_from": feedback_from,
            "src_id": src_id,
            "hop_index": hop_index,

            "apply_source": apply_source,
            "control_mtype": control_mtype,
        })

    def get_ptx_history_df(self):
        return _build_ptx_history_df(self.ptx_history)


    # =========================================================
    #  APC CONTROL HISTORY
    # =========================================================
    def record_apc_control(
        self,
        time,
        event,

        sender_id,
        sender_type,
        receiver_id,
        receiver_type,

        controller_id,
        controller_type,
        target_id,
        target_type,

        control_direction,
        control_scope,

        old_ptx,
        new_ptx,

        # TxPowerIndex-domain fields.
        old_tx_power_index=None,
        new_tx_power_index=None,
        tx_power_index=None,

        dr=None,
        ch_mask=None,
        nb_trans=None,
        status=None,

        margin=None,
        rssi=None,
        snr=None,

        decision_margin=None,
        decision_rssi=None,
        decision_snr=None,

        apc_rx_margin=None,
        apc_rx_rssi=None,
        apc_rx_snr=None,

        src_id=None,
        hop_index=None,
        uplink_prev_hop=None,
        uplink_next_hop=None,

        fopts_hex=None,
        note=None,
    ):
        """
        Record APC control event.

        Notes:
        - Does not change ToA, duty cycle, collision, or energy.
        - Records only the APC_REQ/APC_ANS representation.
        - old_ptx/new_ptx remain physical dBm values.
        - old_tx_power_index/new_tx_power_index belong to the APC domain.
        """

        if tx_power_index is None:
            tx_power_index = new_tx_power_index

        self.apc_control_history.append({
            "run_id": self.run_metadata.get("run_id"),
            "time": time,
            "event": event,

            "sender_id": sender_id,
            "sender_type": sender_type,
            "receiver_id": receiver_id,
            "receiver_type": receiver_type,

            "controller_id": controller_id,
            "controller_type": controller_type,
            "target_id": target_id,
            "target_type": target_type,

            "control_direction": control_direction,
            "control_scope": control_scope,

            # Physical dBm-domain.
            "old_ptx": old_ptx,
            "new_ptx": new_ptx,

            # APC TxPowerIndex-domain.
            "old_tx_power_index": old_tx_power_index,
            "new_tx_power_index": new_tx_power_index,
            "tx_power_index": tx_power_index,

            "dr": dr,
            "ch_mask": ch_mask,
            "nb_trans": nb_trans,
            "status": status,

            "margin": margin,
            "rssi": rssi,
            "snr": snr,

            "decision_margin": decision_margin,
            "decision_rssi": decision_rssi,
            "decision_snr": decision_snr,

            "apc_rx_margin": apc_rx_margin,
            "apc_rx_rssi": apc_rx_rssi,
            "apc_rx_snr": apc_rx_snr,

            "src_id": src_id,
            "hop_index": hop_index,
            "uplink_prev_hop": uplink_prev_hop,
            "uplink_next_hop": uplink_next_hop,

            "fopts_hex": fopts_hex,
            "note": note,
        })
    
    def record_apc_shadow(self, **kwargs):
        """
        Backward-compatible alias for APC shadow/logical answer events.
        Used by legacy NodeRX/RelayRX callers of ``record_apc_shadow()``.
        All events still enter ``apc_control_history``.
        """
        return self.record_apc_control(**kwargs)

    
    def get_apc_control_history_df(self):
        return _build_apc_control_history_df(self.apc_control_history)

    
    # =========================================================
    # 11. LINK MARGIN HISTORY
    # =========================================================
    def record_margin_sample(
        self,
        entity_id,
        entity_type,
        margin,
        rssi=None,
        snr=None,
        decoded=None,
        prev_hop=None,
        next_hop=None,
        time=None,
        reason=None,
        mtype=None,
        sample_type=None,
    ):
        # ---------------------------------------------------------
        # Guard: the margin sample must be valid
        # ---------------------------------------------------------
        if margin is None:
            return
    
        try:
            margin = float(margin)
        except Exception:
            return
    
        if pd.isna(margin):
            return
    
        # Normalize optional rssi/snr values when None or NaN
        try:
            rssi = float(rssi) if rssi is not None and not pd.isna(rssi) else None
        except Exception:
            rssi = None
    
        try:
            snr = float(snr) if snr is not None and not pd.isna(snr) else None
        except Exception:
            snr = None
    
        self.margin_history.append({
            "run_id": self.run_metadata.get("run_id"),
            "time": time,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "margin": margin,
            "rssi": rssi,
            "snr": snr,
            "decoded": decoded,
            "reason": reason,
            "prev_hop": prev_hop,
            "next_hop": next_hop,
            "mtype": mtype,
            "sample_type": sample_type,
        })
        self._record_feedback_margin_observation(
            entity_id, margin, bool(decoded), self._safe_float(time)
        )




    def get_margin_history_df(self):
        return _build_margin_history_df(self.margin_history)

    # =========================================================
    # 12. DELIVERED PACKET MATRIX
    # =========================================================
    def record_delivered_packet(
        self,
        src_id,
        birth_time,
        rx_time,
        hop_count=None,
        current_tx=None,
        node_id="GW",
        packet_uid=None,
        fcnt=None,
    ):
        e2e_delay_s = None
        if birth_time is not None and rx_time is not None and rx_time >= birth_time:
            e2e_delay_s = rx_time - birth_time

        self.summary_delivered_count += 1
        if self.is_summary_profile:
            return
        self.delivered_packets.append({
            "run_id": self.run_metadata.get("run_id"),
            "packet_uid": packet_uid,
            "src_id": src_id,
            "fcnt": fcnt,
            "node": node_id,
            "current_tx": current_tx,
            "birth_time": birth_time,
            "rx_time": rx_time,
            "e2e_delay_s": e2e_delay_s,
            "e2e_delay_ms": e2e_delay_s * 1000.0 if e2e_delay_s is not None else None,
            "hop_count": hop_count,
            "delivery_status": "delivered",
        })

    def get_delivered_packet_df(self):
        if self.is_summary_profile:
            return pd.DataFrame()
        return _build_delivered_packet_df(self.delivered_packets)

    # =========================================================
    # 13. FINAL PTX DISTRIBUTION
    # =========================================================
    def get_final_ptx_distribution_df(self, nodes: list):
        return _build_final_ptx_distribution_df(
            nodes,
            self.run_metadata,
            self._infer_entity_type,
        )

    # =========================================================
    # 14. FAILURE MATRIX
    # =========================================================
    def get_packet_failure_df(self):
        if self.is_summary_profile:
            return pd.DataFrame([
                {"reason": reason, "event_count": count}
                for reason, count in self.summary_drop_reason_counts.items()
            ])
        return _build_packet_failure_df(self.to_dataframe())

    # =========================================================
    # 14B. FRAME-LEVEL SUMMARY
    # =========================================================
    def _get_frame_event_df(self) -> pd.DataFrame:
        """
        Event log with light normalization for the frame-level summary.
        Uses ``to_dataframe()`` to follow the existing Statistics pattern.
        """
        if self.is_summary_profile:
            rows = [
                {
                    "event": event,
                    "mtype": mtype,
                    "hop_index": hop_index,
                    "hop_count": hop_count,
                    "fctrl": fctrl,
                    "fopts_len": fopts_len,
                    "control_cid": control_cid,
                    "reason": reason,
                    "event_count": count,
                }
                for (
                    event, mtype, hop_index, hop_count, fctrl,
                    fopts_len, control_cid, reason,
                ), count in self.summary_frame_event_counts.items()
            ]
            return _normalize_frame_event_df(pd.DataFrame(rows))
        return _normalize_frame_event_df(self.to_dataframe())

    def get_frame_summary_df(self) -> pd.DataFrame:
        """Return the general LoRaPhyMAC frame and physical APC summary."""
        frame_events = self._get_frame_event_df()
        if frame_events.empty:
            return _build_frame_summary_df(
                frame_events,
                self.run_metadata,
                self.summary_ptx_update_count if self.is_summary_profile else len(self.ptx_history),
                0,
            )

        delivered_df = self.get_delivered_packet_df()
        delivered_count = (
            self.summary_delivered_count
            if self.is_summary_profile
            else (0 if delivered_df.empty else len(delivered_df))
        )
        return _build_frame_summary_df(
            frame_events,
            self.run_metadata,
            self.summary_ptx_update_count if self.is_summary_profile else len(self.ptx_history),
            delivered_count,
        )

    def get_frame_mtype_summary_df(self) -> pd.DataFrame:
        """Return the event distribution by LoRaPhyMAC MType."""
        return _build_frame_mtype_summary_df(self._get_frame_event_df())

    def get_frame_hop_event_summary_df(self) -> pd.DataFrame:
        """Return frame-event counts grouped by end-to-end hop count."""
        return _build_frame_hop_event_summary_df(self._get_frame_event_df())

    def get_frame_drop_reason_by_hop_df(self) -> pd.DataFrame:
        """Return frame drop-reason counts grouped by hop count."""
        return _build_frame_drop_reason_by_hop_df(self._get_frame_event_df())


    # =========================================================
    # 15. CORE SUMMARY
    # =========================================================
    def get_core_summary_df(self, nodes: list):
        """Return a single-row DataFrame summarizing one simulation run."""
        return _build_core_summary_df(self, nodes)

    # =========================================================
    # 16. OUTPUT BUNDLES
    # =========================================================
    def get_summary_bundle(self, nodes: list) -> dict:
        """Return the compact scientific summary workbook payload."""
        return _build_summary_bundle(self, nodes)

    def get_result_bundle(
        self,
        nodes: list,
        selected_next_hop_links_df,
        relay_chain_links_df=None,
        visual_links_df=None,
    ) -> dict:
        """Analysis-grade tables used to reproduce figures and diagnoses."""
        return _build_result_bundle(
            self,
            nodes,
            selected_next_hop_links_df,
            relay_chain_links_df,
            visual_links_df,
        )

    def get_debug_bundle(self, nodes=None, selected_next_hop_links_df=None) -> dict:
        """Raw event/lifecycle evidence kept separate from analysis results."""
        return _build_debug_bundle(self)
    
    # =========================================================
    # 17. EXPORT FUNCTIONS
    # =========================================================
    def export_bundle_to_excel(self, bundle: dict, filepath: str):
        """Export a bundle to one Excel workbook."""
        _export_bundle_to_excel(bundle, filepath)

    def export_summary_workbook(self, nodes: list, filepath: str):
        """
        Export primary and detailed outputs.
        """
        bundle = self.get_summary_bundle(nodes)
        self.export_bundle_to_excel(bundle, filepath)
        return bundle

    def export_result_workbook(
        self,
        nodes: list,
        filepath: str,
        selected_next_hop_links_df,
        relay_chain_links_df=None,
        visual_links_df=None,
    ):
        """Export analysis-grade result tables without raw event logs."""
        bundle = self.get_result_bundle(
            nodes,
            selected_next_hop_links_df=selected_next_hop_links_df,
            relay_chain_links_df=relay_chain_links_df,
            visual_links_df=visual_links_df,
        )
        self.export_bundle_to_excel(bundle, filepath)
        return bundle

    def export_debug_workbook(self, nodes: list, filepath: str, selected_next_hop_links_df):
        """
        Export output debug.
        """
        bundle = self.get_debug_bundle(
            nodes,
            selected_next_hop_links_df=selected_next_hop_links_df,
        )
        self.export_bundle_to_excel(bundle, filepath)
        return bundle

    # =========================================================
    # 18. PRINT / VIEW HELPERS
    # =========================================================
    def view_core_summary(self, core_summary_df: pd.DataFrame) -> pd.DataFrame:
        """Return the compact CORE SUMMARY console view."""
        return _view_core_summary(core_summary_df)

    def view_frame_summary(self, frame_summary_df: pd.DataFrame) -> pd.DataFrame:
        """Return the compact FRAME SUMMARY console view."""
        return _view_frame_summary(frame_summary_df)


    def view_relay_duty_cycle(self, relay_duty_cycle_df: pd.DataFrame) -> pd.DataFrame:
        """Return the compact relay duty-cycle console view."""
        return _view_relay_duty_cycle(relay_duty_cycle_df)


    def view_per_channel_duty_cycle(self, per_channel_duty_cycle_df: pd.DataFrame) -> pd.DataFrame:
        """Return the compact per-channel duty-cycle console view."""
        return _view_per_channel_duty_cycle(per_channel_duty_cycle_df)

    def view_energy_breakdown(self, energy_breakdown_df: pd.DataFrame) -> pd.DataFrame:
        """Return the compact energy-breakdown console view."""
        return _view_energy_breakdown(energy_breakdown_df)

    def view_final_ptx_distribution(self, final_ptx_distribution_df: pd.DataFrame) -> pd.DataFrame:
        """Return the compact final-PTX console view."""
        return _view_final_ptx_distribution(final_ptx_distribution_df)


    def view_as_vertical(self, df: pd.DataFrame) -> pd.DataFrame:
        return _view_as_vertical(df)


    def view_apc_control_history(self, apc_control_history_df: pd.DataFrame) -> pd.DataFrame:
        """Return the compact APC-control-history console view."""
        return _view_apc_control_history(apc_control_history_df)

    def view_frame_summary_vertical(self, frame_summary_df: pd.DataFrame) -> pd.DataFrame:
        """Return the two-column-pair vertical FRAME SUMMARY view."""
        return _view_frame_summary_vertical(frame_summary_df)
