"""Map thesis resource requirements to Submitit parameters.

Example: ``resolve_resources('gpu_any_5h', resources, cluster)``.
"""

from __future__ import annotations


def resolve_resources(name: str, resources: dict, cluster: dict | None) -> dict:
    """Keep recorded requirements and apply site-specific Submitit parameters."""
    profile = resources.get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown resource profile {name!r}")
    parameters = {key: value for key, value in profile.items() if key != "gpu_class"}
    if cluster is not None:
        mapping = cluster.get("profiles", {}).get(name)
        if not isinstance(mapping, dict):
            raise ValueError(f"Resource profile {name!r} must exist in resources and cluster configuration")
        parameters.update(cluster.get("parameters", {}))
        parameters.update(mapping)
        if not parameters.get("slurm_partition"):
            raise ValueError(f"Cluster mapping for {name!r} requires slurm_partition")
    return {"profile": name, "requirements": profile, "parameters": parameters}
