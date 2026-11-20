from typing import Dict, Type

from decoupled_engine.benchmark.datasets.agentic_trace import AgenticTraceDataset
from decoupled_engine.benchmark.datasets.common import BaseDataset, DatasetRow
from decoupled_engine.benchmark.datasets.custom import CustomDataset
from decoupled_engine.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixDataset,
)
from decoupled_engine.benchmark.datasets.image import ImageDataset
from decoupled_engine.benchmark.datasets.longbench_v2 import LongBenchV2Dataset
from decoupled_engine.benchmark.datasets.mmmu import MMMUDataset
from decoupled_engine.benchmark.datasets.mooncake import MooncakeDataset
from decoupled_engine.benchmark.datasets.openai_dataset import OpenAIDataset
from decoupled_engine.benchmark.datasets.random import RandomDataset
from decoupled_engine.benchmark.datasets.sharegpt import ShareGPTDataset
from decoupled_engine.benchmark.datasets.speed_bench import SpeedBenchDataset

DATASET_MAPPING: Dict[str, Type[BaseDataset]] = {
    "agentic-trace": AgenticTraceDataset,
    "sharegpt": ShareGPTDataset,
    "custom": CustomDataset,
    "openai": OpenAIDataset,
    # TODO: "random" vs "random-ids" should be a flag (e.g. --random-source=sharegpt|integers),
    # not two separate dataset names sharing the same class.
    "random": RandomDataset,
    "random-ids": RandomDataset,
    "generated-shared-prefix": GeneratedSharedPrefixDataset,
    "mmmu": MMMUDataset,
    "image": ImageDataset,
    "mooncake": MooncakeDataset,
    "longbench_v2": LongBenchV2Dataset,
    "speed-bench": SpeedBenchDataset,
}


def get_dataset(args, tokenizer, model_id=None):
    dataset_name = args.dataset_name
    if dataset_name.startswith("random") and dataset_name not in DATASET_MAPPING:
        dataset_name = "random-ids"

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    dataset_cls = DATASET_MAPPING[dataset_name]
    dataset = dataset_cls.from_args(args)
    return dataset.load(tokenizer=tokenizer, model_id=model_id)


__all__ = [
    "DATASET_MAPPING",
    "DatasetRow",
    "get_dataset",
]
