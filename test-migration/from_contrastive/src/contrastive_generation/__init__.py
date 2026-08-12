"""Public API for the contrastive generation pipeline."""

from .catalog import ContrastiveCatalog, load_catalog
from .planning import load_provider_profile
from .pre_generation import (
    apply_pre_generation_overrides,
    build_pre_generation_plan,
    load_pre_generation_config,
    prepare_generation_handoff,
    verify_pre_generation_run,
    with_pre_generation_provider,
    write_pre_generation_plan,
)
from .runner import (
    GenerationRunner,
    load_generation_run_config,
    validate_twin_runs,
)

__all__ = [
    "ContrastiveCatalog",
    "GenerationRunner",
    "apply_pre_generation_overrides",
    "build_pre_generation_plan",
    "load_catalog",
    "load_generation_run_config",
    "load_pre_generation_config",
    "load_provider_profile",
    "prepare_generation_handoff",
    "verify_pre_generation_run",
    "validate_twin_runs",
    "with_pre_generation_provider",
    "write_pre_generation_plan",
]
