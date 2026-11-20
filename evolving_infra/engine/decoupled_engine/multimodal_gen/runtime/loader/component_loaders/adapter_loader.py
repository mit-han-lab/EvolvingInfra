from decoupled_engine.multimodal_gen.configs.models.adapter.anima import (
    AnimaTextConditionerConfig,
)
from decoupled_engine.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from decoupled_engine.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
)
from decoupled_engine.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class AdapterLoader(PlainStateDictComponentLoader):
    component_names = ["connectors", "duration_head", "text_conditioner"]

    config_classes = {
        "text_conditioner": AnimaTextConditionerConfig,
        "connectors": LTX2ConnectorConfig,
        "duration_head": LTX2DurationHeadConfig,
    }
