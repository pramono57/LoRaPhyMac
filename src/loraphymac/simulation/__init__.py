"""Internal orchestration components for one simulation run.

The public API remains :mod:`loraphymac.simulator`.  This package separates
construction and result assembly from the event runner without exposing those
implementation details to paper workflows.
"""

from loraphymac.simulation.builders import (
    build_infrastructure,
    build_routing,
    build_topology,
)
from loraphymac.simulation.context import (
    RoutingArtifacts,
    SimulationInfrastructure,
    SimulationTopology,
)
from loraphymac.simulation.exporting import export_simulation_outputs
from loraphymac.simulation.output_collection import (
    CollectedSimulationOutputs,
    collect_simulation_outputs,
)
from loraphymac.simulation.reporting import (
    print_one_run_summary,
    report_simulation_outputs,
)
from loraphymac.simulation.results import build_master_summary_row
from loraphymac.simulation.state import (
    initialize_gateway_ptx_state,
    initialize_node_ptx_state,
    initialize_relay_ptx_state,
)


__all__ = [
    "CollectedSimulationOutputs",
    "RoutingArtifacts",
    "SimulationInfrastructure",
    "SimulationTopology",
    "build_infrastructure",
    "build_master_summary_row",
    "build_routing",
    "build_topology",
    "collect_simulation_outputs",
    "export_simulation_outputs",
    "initialize_gateway_ptx_state",
    "initialize_node_ptx_state",
    "initialize_relay_ptx_state",
    "print_one_run_summary",
    "report_simulation_outputs",
]
