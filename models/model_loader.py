from models import feature_extractors
from models import transformers


class ModelLoader:
    """
    Class to load model structure (feature extractor or transformer).
    Only models specified in feature_extractors_ or transformers_ can be loaded.
    """

    feature_extractors_ = {
        "cnn_extractor": feature_extractors.CnnExtractor,
        "resnet50_extractor": feature_extractors.Resnet50Extractor,
    }

    transformers_ = {
        "fake_transformer": transformers.FakeTransformer,
        "hubert_transformer": transformers.HubertTransformer,
        "sign_language_transformer": transformers.SignLanguageTransformer,
        "sign_language_cnn_transformer": transformers.SignLanguageCNNTransformer,
        "sign_language_double_transformer": transformers.SignLanguageDoubleTransformer,
        "sign_language_keyframe_selector": transformers.SignLanguageKeyframeSelector,
        "lstm": transformers.LSTM,
    }

    def load_feature_extractor(self, feature_extractor_name, *args, **kwargs):
        try:
            return self.feature_extractors_[feature_extractor_name](*args, **kwargs)
        except KeyError:
            raise Exception(
                f"Feature extractor {feature_extractor_name} does not exists"
            )

    def load_transformer(self, transformer_name, *args, **kwargs):
        try:
            return self.transformers_[transformer_name](*args, **kwargs)
        except KeyError:
            raise Exception(f"Transformer {transformer_name} does not exists")
