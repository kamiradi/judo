"""Run the judo CLI, but hold each dora node back until the dataflow is running.

Works around a startup race in dora_utils.launch.run, which spawns the dynamic nodes in
parallel with `dora up && dora start`; nodes that connect before the dataflow is registered
fail with "no node with ID ..." / "Could not connect to the daemon".

Usage: python judo_wait.py task=uncertain_cylinder_push optimizer=mppi
"""

import subprocess

from judo import cli

_Base = subprocess.Popen  # judo.cli has already swapped in its _TrackingPopen


class _WaitingPopen(_Base):  # type: ignore[misc, valid-type]
    def __init__(self, args, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        if isinstance(args, str) and "_launch_node.py" in args:
            args = (
                "for i in $(seq 1 300); do dora list 2>/dev/null | grep -q Running && break; sleep 0.1; done; " + args
            )
        super().__init__(args, *a, **kw)


subprocess.Popen = _WaitingPopen  # type: ignore[misc]

if __name__ == "__main__":
    cli.app()
