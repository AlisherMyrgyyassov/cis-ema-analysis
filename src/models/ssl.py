"""Self-supervised audio models for the Exp 1.5 ceiling (wav2vec2 / HuBERT / XLSR).

Wraps HuggingFace ``AutoModelForSequenceClassification`` with a fixed CNN feature
encoder (frozen) and SpecAugment disabled (isolated syllables are shorter than
the default mask length). Returns the model, processor, and model id.
"""
import torch


TARGET_SR = 16000

# Candidate ceilings. 'auto' resolves to the first that loads.
AUTO_MODELS = [
    "TencentGameMate/chinese-hubert-large",
    "facebook/wav2vec2-large-xlsr-53",
    "kehanlu/mandarin-wav2vec2",
    "facebook/wav2vec2-base",
]


def load_ssl_model(model_arg, num_labels):
    from transformers import (AutoConfig, AutoFeatureExtractor,
                              HubertForSequenceClassification,
                              Wav2Vec2ForSequenceClassification)

    ids = AUTO_MODELS if model_arg == "auto" else [model_arg]
    last_err = None
    for mid in ids:
        print(f"loading {mid} ...")
        try:
            try:
                processor = AutoFeatureExtractor.from_pretrained(mid)
            except Exception:
                processor = None

            cfg = AutoConfig.from_pretrained(mid)
            cls = (HubertForSequenceClassification if cfg.model_type == "hubert"
                   else Wav2Vec2ForSequenceClassification)
            model = cls.from_pretrained(
                mid, num_labels=num_labels, ignore_mismatched_sizes=True,
            )
            # Isolated syllables are shorter than default SpecAugment mask_time_length.
            for attr in ("mask_time_prob", "mask_feature_prob"):
                if hasattr(model.config, attr):
                    setattr(model.config, attr, 0.0)

            # Freeze the CNN feature encoder; fine-tune transformer + head.
            frozen = 0
            for name, p in model.named_parameters():
                if "feature_extractor" in name or "feature_projection" in name:
                    p.requires_grad = False
                    frozen += 1
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"loaded {mid} ({n_params:.1f}M params, {frozen} frozen)")
            return mid, processor, model
        except Exception as e:
            last_err = e
            print(f"  failed ({type(e).__name__}: {e})")
    raise RuntimeError(f"could not load any of {ids}") from last_err


def default_processor():
    from transformers import Wav2Vec2FeatureExtractor
    return Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=TARGET_SR, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
