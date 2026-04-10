# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import random

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from torchspec.utils import accelerator as accel
from torchspec.utils.logging import logger
from torchspec.utils.misc import _to_local_gpu_id, get_current_node_ip, get_free_port


def _get_accel_resource_name() -> str:
    """Return the Ray resource name for the active accelerator.

    NVIDIA CUDA → ``"GPU"`` (Ray built-in).
    Ascend NPU  → ``"NPU"`` (custom resource; register with
    ``ray start --resources='{"NPU": N}'`` or equivalent).
    """
    return "NPU" if accel.is_npu() else "GPU"


def _get_accel_ids() -> list:
    """Return the accelerator IDs assigned to the current Ray worker.

    For CUDA uses ``ray.get_gpu_ids()`` (Ray manages CUDA_VISIBLE_DEVICES).
    For NPU uses ``ray.get_resource_ids()`` with the ``"NPU"`` custom resource
    key; falls back to an empty list if the resource is not tracked at
    device granularity by this Ray installation.
    """
    resource_name = _get_accel_resource_name()
    if resource_name == "GPU":
        return ray.get_gpu_ids()
    resource_ids = ray.get_resource_ids()
    return [int(entry[0]) for entry in resource_ids.get(resource_name, [])]


def _accel_options(fraction: float) -> dict:
    """Return resource kwargs for ``ray.remote()`` / ``.options()`` calls.

    On CUDA returns ``{"num_gpus": fraction}`` (Ray built-in GPU resource).
    On NPU returns ``{"resources": {"NPU": fraction}}`` (custom resource).

    Usage::

        engine = SomeActor.options(
            num_cpus=0.2,
            **_accel_options(0.2),
            scheduling_strategy=...,
        ).remote(...)
    """
    resource_name = _get_accel_resource_name()
    if resource_name == "GPU":
        return {"num_gpus": fraction}
    return {"resources": {resource_name: fraction}}


def node_affinity_for_ip(ip: str, name: str = None) -> NodeAffinitySchedulingStrategy:
    """Return a NodeAffinitySchedulingStrategy pinned to the live Ray node with the given IP.

    Args:
        ip: Node IP address to pin to.
        name: Optional actor name for log messages.

    Returns:
        NodeAffinitySchedulingStrategy with soft=False.

    Raises:
        RuntimeError: If no live Ray node with that IP is found.
    """
    for node in ray.nodes():
        if node.get("Alive", False) and node.get("NodeManagerAddress") == ip:
            node_id = node["NodeID"]
            label = f"{name} " if name else ""
            logger.info(f"Pinning {label}actor to node {ip} (id={node_id[:8]}...)")
            return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    live_ips = [n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive", False)]
    raise RuntimeError(f"No live Ray node with IP {ip!r} found. Live nodes: {live_ips}")


class RayActor:
    """Base class for all torchspec Ray actors."""

    @staticmethod
    def get_node_ip() -> str:
        """Get current node IP address."""
        return get_current_node_ip()

    @staticmethod
    def find_free_port(start_port=10000, consecutive=1) -> int:
        """Find available port(s) on current node."""
        return get_free_port(start_port=start_port, consecutive=consecutive)

    @staticmethod
    def resolve_local_gpu_id(physical_gpu_id: int) -> int:
        """Convert physical GPU ID to node-local GPU ID."""
        return _to_local_gpu_id(physical_gpu_id)

    def setup_gpu(self, base_gpu_id: int | None = None) -> int:
        """Resolve accelerator device, set it as current, set LOCAL_RANK env var.

        Works for both NVIDIA CUDA (ray.get_gpu_ids) and Ascend NPU
        (ray.get_resource_ids with the "NPU" custom resource key).

        Args:
            base_gpu_id: Physical device ID. If None, auto-detect from Ray's
                resource assignment for the current worker.

        Returns:
            Local device ID (0-based within CUDA_VISIBLE_DEVICES /
            ASCEND_RT_VISIBLE_DEVICES).
        """
        if base_gpu_id is None:
            ids = _get_accel_ids()
            if ids:
                base_gpu_id = int(float(ids[0]))
            else:
                resource_name = _get_accel_resource_name()
                logger.warning(
                    f"No '{resource_name}' resource IDs found for this Ray worker "
                    f"(is the cluster started with --resources='{\"{resource_name}\": N}')? "
                    "Falling back to device 0, which may be incorrect for multi-device setups."
                )
                base_gpu_id = 0
        local_gpu_id = self.resolve_local_gpu_id(base_gpu_id)
        accel.set_device(local_gpu_id)
        os.environ["LOCAL_RANK"] = str(local_gpu_id)
        return local_gpu_id

    def setup_master(self, master_addr=None, master_port=None, port_range=(10000, 11000)):
        """Resolve master address/port for distributed communication.

        If master_addr is provided, use it directly. Otherwise auto-resolve.
        Stores result in self.master_addr, self.master_port.
        """
        if master_addr:
            self.master_addr = master_addr
            self.master_port = master_port
        else:
            self.master_addr = self.get_node_ip()
            self.master_port = self.find_free_port(start_port=random.randint(*port_range))

    def get_master_addr_and_port(self):
        """Return (master_addr, master_port) tuple."""
        return self.master_addr, self.master_port
