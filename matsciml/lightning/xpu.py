from __future__ import annotations

from datetime import timedelta
from typing import Callable, Union, List, Dict, Any

from matsciml.common.packages import package_registry

_has_xpu = False

if package_registry["ipex"]:
    # this is not used, but we need to do so in order
    # to make sure it's registered with PyTorch
    import intel_extension_for_pytorch as ipex  # noqa: F401
    import torch

    if hasattr(torch, "xpu"):
        _has_xpu = True

if _has_xpu:
    from matsciml.lightning.ddp import MPIEnvironment
    from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry
    from pytorch_lightning.strategies import SingleDeviceStrategy
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.plugins import CheckpointIO, ClusterEnvironment
    from pytorch_lightning.plugins.collectives.torch_collective import (
        default_pg_timeout,
    )
    from pytorch_lightning.plugins.precision import (
        PrecisionPlugin,
        MixedPrecisionPlugin,
    )

    from torch import distributed as dist

    class XPUAccelerator(Accelerator):

        """
        Implements a class for handling Intel XPU offloading, particularly the Data Center
        GPU Max Series (previously codename Ponte Vecchio).
        """

        @staticmethod
        def parse_devices(devices: Union[int, List[int]]) -> List[int]:
            """
            Parse the `trainer` input for devices and homogenize them.
            Parameters
            ----------
            devices : Union[int, List[int]]
                Single or list of device numbers to use
            Returns
            -------
            List[int]
                List of device numbers to use
            """
            if isinstance(devices, int):
                devices = [
                    devices,
                ]
            return devices

        def setup_device(self, device: torch.device) -> None:
            # first try and see if we can grab the index from the device
            index = getattr(device, "index", None)
            if index is None and not dist.is_initialized():
                index = 0
            torch.xpu.set_device(index)

        def teardown(self) -> None:
            # as it suggests, this is run on cleanup
            torch.xpu.empty_cache()

        def get_device_stats(self, device) -> Dict[str, Any]:
            return torch.xpu.memory_stats(device)

        @staticmethod
        def get_parallel_devices(devices: List[int]) -> List[torch.device]:
            """
            Return a list of torch devices corresponding to what is available.
            Essentially maps indices to `torch.device` objects.
            Parameters
            ----------
            devices : List[int]
                List of integers corresponding to device numbers
            Returns
            -------
            List[torch.device]
                List of `torch.device` objects for each device
            """
            return [torch.device("xpu", i) for i in devices]

        @staticmethod
        def auto_device_count() -> int:
            # by default, PVC has two tiles per GPU
            return torch.xpu.device_count()

        @staticmethod
        def is_available() -> bool:
            """
            Determines if an XPU is actually available.
            Returns
            -------
            bool
                True if devices are detected, otherwise False
            """
            try:
                return torch.xpu.device_count() != 0
            except (AttributeError, NameError):
                return False

        @classmethod
        def register_accelerators(cls, accelerator_registry) -> None:
            accelerator_registry.register(
                "xpu",
                cls,
                description="Intel Data Center GPU Max Series, formerly codenamed Ponte Vecchio",
            )

    # add XPU to the registry, allowing you to just specify it as `xpu`
    # as a string in `trainer` arguments.
    AcceleratorRegistry.register("xpu", XPUAccelerator)

    class SingleXPUStrategy(SingleDeviceStrategy):

        """
        This class implements the strategy for using a single XPU device,
        which depending on whether explicit or implicit scaling is used,
        can represent either the whole GPU or a single tile within a GPU.
        """

        strategy_name = "xpu_single"

        def __init__(
            self,
            device: str | None = "xpu",
            checkpoint_io=None,
            precision_plugin=None,
        ):
            super().__init__(
                device=device,
                accelerator=XPUAccelerator(),
                checkpoint_io=checkpoint_io,
                precision_plugin=precision_plugin,
            )

        @property
        def is_distributed(self) -> bool:
            return False

        def setup(self, trainer) -> None:
            self.model_to_device()
            super().setup(trainer)

        def setup_optimizers(self, trainer) -> None:
            super().setup_optimizers(trainer)

        def model_to_device(self) -> None:
            self.model.to(self.root_device)

        @classmethod
        def register_strategies(cls, strategy_registry) -> None:
            strategy_registry.register(
                cls.strategy_name, cls, description=f"{cls.__class__.__name__}"
            )

    class DDPXPUStrategy(DDPStrategy):
        def __init__(
            self,
            parallel_devices: List[torch.device] | None = None,
            cluster_environment: ClusterEnvironment | None = None,
            checkpoint_io: CheckpointIO | None = None,
            precision_plugin: PrecisionPlugin | None = None,
            ddp_comm_state: object | None = None,
            ddp_comm_hook: Callable[..., Any] | None = None,
            ddp_comm_wrapper: Callable[..., Any] | None = None,
            model_averaging_period: int | None = None,
            process_group_backend: str | None = "ccl",
            timeout: timedelta | None = default_pg_timeout,
            **kwargs: Any,
        ) -> None:
            accelerator = XPUAccelerator()
            if cluster_environment is None:
                cluster_environment = MPIEnvironment()
            super().__init__(
                accelerator,
                parallel_devices,
                cluster_environment,
                checkpoint_io,
                precision_plugin,
                ddp_comm_state,
                ddp_comm_hook,
                ddp_comm_wrapper,
                model_averaging_period,
                process_group_backend,
                timeout,
                **kwargs,
            )

        @classmethod
        def register_strategies(cls, strategy_registry) -> None:
            strategy_registry.register(
                cls.strategy_name,
                cls,
                description=f"{cls.__class__.__name__} - uses distributed data parallelism"
                " to divide data across multiple XPU tiles.",
            )

    class XPUBF16Plugin(MixedPrecisionPlugin):
        def __init__(self):
            super().__init__(torch.bfloat16, "xpu")

        def auto_cast_context_manager(self):
            """
            Overrides the default behavior, which relies on `torch.amp` where only
            CPU and CUDA backends are supported. This uses the `xpu.amp` interface
            explicitly, as done in the IPEX documentation.
            """
            return torch.xpu.amp.autocast(
                self.device, enabled=True, dtype=torch.bfloat16
            )
