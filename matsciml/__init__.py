from __future__ import annotations

# this ensures that the package registry is built so optional
# features can be checked later on
from matsciml.common.packages import package_registry  # noqa: F401

if package_registry["ipex"]:
    import intel_extension_for_pytorch as ipex  # noqa: F401

__version__ = "1.2.0"
