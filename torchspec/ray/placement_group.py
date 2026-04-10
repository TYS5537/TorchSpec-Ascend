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
import socket
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torchspec.ray.ray_actor import _get_accel_ids, _get_accel_resource_name
from torchspec.ray.train_group import RayTrainGroup
from torchspec.utils.logging import logger


@ray.remote
class InfoActor:
    """Lightweight probe actor used to discover which physical device ID
    Ray assigned to a specific placement-group bundle.

    Resource requirements are NOT declared in the decorator — they are
    injected dynamically via ``.options()`` at creation time so the correct
    resource type (``GPU`` for CUDA, ``NPU`` for Ascend) can be selected
    at runtime.
    """

    def get_ip_and_device_id(self):
        ids = _get_accel_ids()
        device_id = ids[0] if ids else 0
        return ray.util.get_node_ip_address(), device_id


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_devices, strategy="PACK", name=None):
    """Create a placement group with the specified number of accelerator devices.

    Supports both NVIDIA CUDA (``GPU`` Ray resource) and Ascend NPU
    (``NPU`` custom Ray resource registered via
    ``ray start --resources='{"NPU": N}'``).
    """
    resource_name = _get_accel_resource_name()
    bundles = [{resource_name: 1, "CPU": 1} for _ in range(num_devices)]
    pg = placement_group(bundles, strategy=strategy, name=name)
    num_bundles = len(bundles)

    ray.get(pg.ready())

    # Probe each bundle to discover the physical device ID assigned to it.
    info_actors = []
    for i in range(num_bundles):
        # Inject resource requirement matching the bundle type so Ray places
        # the InfoActor inside the correct bundle slot.
        options_kwargs: dict = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        }
        if resource_name == "GPU":
            options_kwargs["num_gpus"] = 1
        else:
            options_kwargs["resources"] = {resource_name: 1}
        info_actors.append(InfoActor.options(**options_kwargs).remote())

    device_ids = ray.get([actor.get_ip_and_device_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, device_ids[i][0], device_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical device ID
    pg_reordered_device_ids = [device_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {device_ids[actual_bundle_index][0]}, "
            f"{resource_name.lower()}: {device_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_device_ids


def _ensure_ray_initialized():
    """Connect to an existing Ray cluster, or start a local instance as fallback."""
    if ray.is_initialized():
        return

    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        logger.info(f"Connected to Ray cluster at {ray_address}")
    except ConnectionError:
        logger.warning("No existing Ray cluster found, starting a local instance")
        ray.init(ignore_reinit_error=True)


def _get_expected_gpu_count(args) -> int:
    training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    inference_gpus = getattr(args, "inference_num_gpus", 0)
    if (
        getattr(args, "colocate", False)
        or getattr(args, "debug_train_only", False)
        or getattr(args, "debug_inference_only", False)
    ):
        return max(training_gpus, inference_gpus)
    return training_gpus + inference_gpus


def _wait_for_gpu_resources(expected_devices: int, timeout: int = 300, poll_interval: int = 5):
    """Block until the Ray cluster has at least ``expected_devices`` accelerators.

    Checks the ``GPU`` resource for CUDA or the ``NPU`` custom resource for
    Ascend, depending on the active accelerator backend.
    """
    resource_name = _get_accel_resource_name()
    available = int(ray.cluster_resources().get(resource_name, 0))
    if available >= expected_devices:
        logger.info(f"Ray cluster has {available} {resource_name}s (need {expected_devices})")
        return

    logger.info(
        f"Waiting for {expected_devices} {resource_name}s "
        f"(currently {available}), timeout={timeout}s..."
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        available = int(ray.cluster_resources().get(resource_name, 0))
        logger.info(f"Ray cluster {resource_name}s: {available}/{expected_devices}")
        if available >= expected_devices:
            logger.info(f"All {expected_devices} {resource_name}s available")
            return

    raise RuntimeError(
        f"Timed out waiting for {resource_name}s: {available}/{expected_devices} "
        f"after {timeout}s. Check that all Ray worker nodes have joined the cluster."
    )


def create_placement_groups(args):
    """Initialize Ray, wait for accelerator resources, and create placement groups.

    This is the single entry point for all device placement setup.
    Supports both NVIDIA CUDA (``GPU`` resource) and Ascend NPU
    (``NPU`` custom resource).
    """
    _ensure_ray_initialized()
    _wait_for_gpu_resources(_get_expected_gpu_count(args))

    if args.debug_train_only:
        num_training_devices = args.training_num_nodes * args.training_num_gpus_per_node
        logger.info(f"Creating training placement group with {num_training_devices} devices...")
        training_pg, training_bundle_indices, training_device_ids = _create_placement_group(
            num_training_devices, strategy="PACK", name="training_pg"
        )
        return {
            "training": (training_pg, training_bundle_indices, training_device_ids),
            "inference": (training_pg, [], []),
        }

    if args.debug_inference_only:
        num_inference_devices = args.inference_num_gpus
        logger.info(f"Creating inference placement group with {num_inference_devices} devices...")
        inference_pg, inference_bundle_indices, inference_device_ids = _create_placement_group(
            num_inference_devices, strategy="PACK", name="inference_pg"
        )
        return {
            "training": (inference_pg, [], []),
            "inference": (inference_pg, inference_bundle_indices, inference_device_ids),
        }

    if args.colocate:
        num_devices = args.training_num_nodes * args.training_num_gpus_per_node
        logger.info(f"Creating colocated placement group with {num_devices} devices...")
        pg, bundle_indices, device_ids = _create_placement_group(
            num_devices, strategy="PACK", name="colocate_pg"
        )
        return {
            "training": (pg, bundle_indices, device_ids),
            "inference": (pg, bundle_indices, device_ids),
        }

    num_training_devices = args.training_num_nodes * args.training_num_gpus_per_node
    num_inference_devices = args.inference_num_gpus

    logger.info(
        f"Creating placement groups: {num_inference_devices} devices for inference and "
        f"{num_training_devices} devices for training..."
    )

    placement_strategy = getattr(args, "placement_strategy", "training_first")

    if placement_strategy == "training_first":
        logger.info(
            "Creating training placement group first (placement_strategy=training_first)..."
        )
        training_pg, training_bundle_indices, training_device_ids = _create_placement_group(
            num_training_devices, strategy="PACK", name="training_pg"
        )
        logger.info("Creating inference placement group...")
        inference_pg, inference_bundle_indices, inference_device_ids = _create_placement_group(
            num_inference_devices, strategy="PACK", name="inference_pg"
        )
    else:
        logger.info(
            "Creating inference placement group first (placement_strategy=inference_first)..."
        )
        inference_pg, inference_bundle_indices, inference_device_ids = _create_placement_group(
            num_inference_devices, strategy="PACK", name="inference_pg"
        )
        logger.info("Creating training placement group...")
        training_pg, training_bundle_indices, training_device_ids = _create_placement_group(
            num_training_devices, strategy="PACK", name="training_pg"
        )

    return {
        "training": (training_pg, training_bundle_indices, training_device_ids),
        "inference": (inference_pg, inference_bundle_indices, inference_device_ids),
    }


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, training_class=None):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        training_class=training_class,
    )


def create_train_group(args, training_pg, training_class=None, mooncake_config=None):
    train_group = allocate_train_group(
        args=args,
        num_nodes=args.training_num_nodes,
        num_gpus_per_node=args.training_num_gpus_per_node,
        pg=training_pg,
        training_class=training_class,
    )

    some_ids = ray.get(
        train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )
    )

    assert len(set(some_ids)) == 1

    return train_group
