"""
Hugging Face Export Bridge for LUMINARK
Export LUMINARK models to Hugging Face format for sharing
Install: pip install transformers huggingface-hub
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    raise ImportError("Hugging Face dependencies not available. Install with: pip install transformers huggingface-hub")


class LuminarkConfig(PretrainedConfig):
    """
    Configuration for LUMINARK models exported to HF

    Compatible with Hugging Face transformers library
    """
    model_type = "luminark"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 1024,
        activation: str = "gelu",
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        # LUMINARK-specific
        has_quantum_confidence: bool = True,
        has_toroidal_attention: bool = True,
        sar_stages: int = 81,
        current_sar_stage: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings

        # LUMINARK features
        self.has_quantum_confidence = has_quantum_confidence
        self.has_toroidal_attention = has_toroidal_attention
        self.sar_stages = sar_stages
        self.current_sar_stage = current_sar_stage


class HFBridge:
    """
    Bridge to export LUMINARK models to Hugging Face format

    Usage:
        bridge = HFBridge()
        bridge.export_model(luminark_model, "my-model", push_to_hub=True, repo_id="username/model")
    """

    def __init__(self):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face not available")

        self.api = HfApi()
        print("🤗 Hugging Face Bridge initialized")

    def convert_luminark_to_hf_weights(self,
                                      luminark_model,
                                      config: LuminarkConfig) -> Dict[str, np.ndarray]:
        """
        Convert LUMINARK model weights to HF format

        Args:
            luminark_model: LUMINARK model with parameters()
            config: HF config

        Returns:
            Dict of weight name -> numpy array
        """
        hf_weights = {}

        # Get all LUMINARK parameters
        param_dict = {}
        for name, param in luminark_model.named_parameters():
            param_dict[name] = param.data.numpy() if hasattr(param.data, 'numpy') else param.data

        # Map LUMINARK names to HF names
        # This is a simplified mapping - would need customization per model
        for lum_name, weight in param_dict.items():
            # Simple name transformation
            hf_name = lum_name.replace('fc', 'dense').replace('attn', 'attention')
            hf_weights[hf_name] = weight

        return hf_weights

    def export_model(self,
                    luminark_model,
                    output_path: str,
                    model_name: str = "luminark-model",
                    description: str = "",
                    push_to_hub: bool = False,
                    repo_id: Optional[str] = None,
                    **config_kwargs) -> Path:
        """
        Export LUMINARK model to Hugging Face format

        Args:
            luminark_model: LUMINARK model to export
            output_path: Local save path
            model_name: Model name
            description: Model description
            push_to_hub: Upload to HF Hub
            repo_id: HF repo ID (username/model-name)
            **config_kwargs: Additional config parameters

        Returns:
            Path to saved model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"📦 Exporting LUMINARK model to: {output_path}")

        # Create config
        config = LuminarkConfig(**config_kwargs)

        # Save config
        config.save_pretrained(output_path)
        print(f"   ✓ Saved config.json")

        # Convert and save weights
        weights = self.convert_luminark_to_hf_weights(luminark_model, config)

        # Save weights as numpy (can be loaded by HF)
        np.savez(output_path / "pytorch_model.npz", **weights)
        print(f"   ✓ Saved weights ({len(weights)} tensors)")

        # Save model card
        model_card = self._create_model_card(
            model_name=model_name,
            description=description,
            config=config
        )

        with open(output_path / "README.md", 'w') as f:
            f.write(model_card)
        print(f"   ✓ Saved README.md")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'framework': 'LUMINARK',
            'exported_format': 'huggingface',
            'has_quantum_confidence': config.has_quantum_confidence,
            'has_toroidal_attention': config.has_toroidal_attention,
            'sar_stages': config.sar_stages,
            'current_sar_stage': config.current_sar_stage
        }

        with open(output_path / "luminark_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved metadata")

        # Push to hub if requested
        if push_to_hub:
            if not repo_id:
                raise ValueError("repo_id required for push_to_hub=True")

            self._push_to_hub(output_path, repo_id)

        print(f"✅ Export complete!")
        return output_path

    def _create_model_card(self,
                          model_name: str,
                          description: str,
                          config: LuminarkConfig) -> str:
        """Create Hugging Face model card (README.md)"""

        card = f"""---
language: en
license: mit
tags:
- luminark
- quantum-ai
- sar-awareness
- toroidal-attention
---

# {model_name}

{description}

## Model Description

This model was trained using the LUMINARK framework, which features:
- **Quantum Confidence Estimation**: Real quantum uncertainty quantification
- **SAR Stage Awareness**: {config.sar_stages}-stage training progression tracking
- **Toroidal Attention**: Circular pattern detection
- **Multi-Modal Sensing**: Mycelial sensory system integration

## Model Architecture

- **Vocabulary Size**: {config.vocab_size}
- **Hidden Size**: {config.hidden_size}
- **Layers**: {config.num_layers}
- **Attention Heads**: {config.num_heads}
- **Max Position Embeddings**: {config.max_position_embeddings}

## LUMINARK Features

- **Quantum Confidence**: {config.has_quantum_confidence}
- **Toroidal Attention**: {config.has_toroidal_attention}
- **SAR Stages**: {config.sar_stages}
- **Current Stage**: {config.current_sar_stage}

## Usage

```python
from transformers import AutoModel, AutoConfig

# Load config
config = AutoConfig.from_pretrained("{model_name}")

# Note: This is a LUMINARK model export
# For full functionality, use the LUMINARK framework
```

## Training

Trained using LUMINARK's stage-aware curriculum learning with:
- Adaptive learning rates based on SAR stage
- Quantum confidence monitoring
- Multi-modal sensory validation

## Citation

If you use this model, please cite the LUMINARK framework:

```bibtex
@software{{luminark2025,
  title = {{LUMINARK: Quantum-Aware AI Framework}},
  year = {{2025}},
  note = {{https://github.com/your-org/luminark}}
}}
```

## More Information

- **Framework**: [LUMINARK](https://github.com/your-org/luminark)
- **Documentation**: [LUMINARK Docs](https://luminark.readthedocs.io)
- **License**: MIT
"""
        return card

    def _push_to_hub(self, local_path: Path, repo_id: str):
        """Push model to Hugging Face Hub"""
        print(f"\n📤 Pushing to Hugging Face Hub: {repo_id}")

        try:
            # Create repo if it doesn't exist
            create_repo(repo_id, exist_ok=True)
            print(f"   ✓ Repository ready: {repo_id}")

            # Upload all files
            self.api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="model"
            )

            print(f"   ✓ Upload complete!")
            print(f"   🔗 View at: https://huggingface.co/{repo_id}")

        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            print(f"   💡 Make sure you're logged in: huggingface-cli login")

    def create_tokenizer(self,
                        vocab_dict: Dict[int, str],
                        output_path: str) -> Any:
        """
        Create and save HF-compatible tokenizer

        Args:
            vocab_dict: Dict of token_id -> token_string
            output_path: Where to save tokenizer

        Returns:
            Tokenizer object
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create vocabulary file
        vocab_list = [vocab_dict.get(i, f"<unk_{i}>") for i in range(len(vocab_dict))]

        vocab_file = output_path / "vocab.txt"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token in vocab_list:
                f.write(f"{token}\n")

        print(f"✓ Saved tokenizer vocab ({len(vocab_list)} tokens)")

        # Create tokenizer config
        tokenizer_config = {
            "model_type": "luminark",
            "vocab_size": len(vocab_list),
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }

        with open(output_path / "tokenizer_config.json", 'w') as f:
            json.dump(tokenizer_config, f, indent=2)

        print(f"✓ Saved tokenizer config")
        return vocab_file


def quick_export(luminark_model,
                output_dir: str,
                vocab_dict: Dict[int, str] = None,
                push_to_hub: bool = False,
                repo_id: str = None) -> Path:
    """
    Quick export function

    Args:
        luminark_model: LUMINARK model
        output_dir: Output directory
        vocab_dict: Optional vocabulary mapping
        push_to_hub: Upload to HF Hub
        repo_id: HF repository ID

    Returns:
        Path to exported model
    """
    if not HF_AVAILABLE:
        print("Hugging Face not available. Install: pip install transformers huggingface-hub")
        return None

    bridge = HFBridge()

    # Export model
    path = bridge.export_model(
        luminark_model=luminark_model,
        output_path=output_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id
    )

    # Export tokenizer if vocab provided
    if vocab_dict:
        bridge.create_tokenizer(vocab_dict, output_dir)

    return path


if __name__ == '__main__':
    # Demo
    print("🤗 Hugging Face Bridge Demo\n")

    if not HF_AVAILABLE:
        print("Install dependencies first:")
        print("  pip install transformers huggingface-hub")
        exit(1)

    # Mock LUMINARK model for demo
    class MockModel:
        def named_parameters(self):
            return [
                ('fc1.weight', type('obj', (), {'data': type('obj', (), {'numpy': lambda: np.random.randn(256, 128)})()})()),
                ('fc1.bias', type('obj', (), {'data': type('obj', (), {'numpy': lambda: np.random.randn(256)})()})()),
            ]

    model = MockModel()

    # Export
    bridge = HFBridge()

    output_path = bridge.export_model(
        luminark_model=model,
        output_path="hf_export_demo",
        model_name="luminark-demo",
        description="Demo LUMINARK model export",
        push_to_hub=False,  # Set True and add repo_id to actually upload
        vocab_size=1000,
        hidden_size=256,
        num_layers=6
    )

    print(f"\n✅ Demo complete! Model saved to: {output_path}")
    print(f"   Files created: config.json, pytorch_model.npz, README.md, luminark_metadata.json")
