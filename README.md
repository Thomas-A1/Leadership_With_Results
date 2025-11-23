# Prosit 3: Transformer-Based Speech Recognition System

This repository contains the implementation of Prosit 3, which focuses on building transformer-based speech recognition systems for Twi medical dictation using the AfriSpeech200 dataset.

## Project Structure

```
.
├── transformer_asr/              # Pretrained transformer ASR system
│   ├── data/                    # Data processing modules
│   │   ├── dataset.py           # AfriSpeech200 dataset class
│   │   ├── preprocessing.py     # Audio and text preprocessing
│   │   └── augmentation.py      # Data augmentation utilities
│   ├── models/                  # Model definitions
│   │   └── pretrained_asr.py    # Pretrained model wrapper
│   ├── training/                # Training utilities
│   │   ├── trainer.py           # Training loop
│   │   ├── evaluator.py         # Evaluation utilities
│   │   └── metrics.py           # CER/WER metrics
│   ├── utils/                   # Utility modules
│   │   ├── config.py            # Configuration management
│   │   └── visualization.py     # Plotting utilities
│   └── main_pretrained.py       # Main training script
│
├── transformer_scratch/         # From-scratch transformer implementation
│   ├── models/                  # Transformer components
│   │   ├── attention.py         # Multi-head attention
│   │   ├── positional_encoding.py  # Positional encoding
│   │   └── transformer.py       # Complete transformer
│   └── main_scratch.py          # Training script for scratch transformer
│
├── afrispeech200/               # Dataset directory
│   ├── transcripts/twi/         # Transcript CSV files
│   └── audio/twi/               # Audio files
│
└── checkpoints/                 # Saved model checkpoints
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies

Install required packages (verify against approved packages list):

```bash
pip install torch torchaudio
pip install transformers  # If approved
pip install librosa soundfile
pip install jiwer
pip install pandas numpy matplotlib seaborn
pip install tqdm
```

**Note**: Before installing, verify that packages are approved. Check the approved packages list: https://docs.google.com/spreadsheets/d/1Z_QXO6KulHjnewR_g2LwhuyI0C-XMUZKvCmJskHKang/edit?usp=sharing

## Dataset Setup

The AfriSpeech200 dataset should be organized as follows:

```
afrispeech200/
├── transcripts/
│   └── twi/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
└── audio/
    └── twi/
        ├── train/
        │   └── data/
        │       └── data/
        │           └── *.wav
        ├── dev/
        │   └── data/
        │       └── data/
        │           └── *.wav
        └── test/
            └── data/
                └── data/
                    └── *.wav
```

## Usage

### Pretrained Transformer ASR

Train a pretrained transformer model (e.g., Whisper) on AfriSpeech200:

```bash
cd transformer_asr
python main_pretrained.py
```

Configuration can be modified in `utils/config.py`:

- Model selection (Whisper vs Wav2Vec2)
- Training hyperparameters
- Data preprocessing settings

### From-Scratch Transformer

Train a transformer from scratch on a simple demonstration task:

```bash
cd transformer_scratch
python main_scratch.py
```

This demonstrates the core transformer architecture on a sequence reversal task, providing educational insights into how transformers work.

## Model Selection

### Pretrained Models

- **Whisper-small** (recommended): Excellent multilingual support, transformer-based architecture
- **Wav2Vec2-base**: Strong for low-resource languages, CNN + Transformer hybrid

Configure in `transformer_asr/utils/config.py`:

```python
model = ModelConfig(
    model_name='openai/whisper-small',  # or 'facebook/wav2vec2-base-960h'
    freeze_feature_extractor=True,
    freeze_encoder=False
)
```

### From-Scratch Transformer

The from-scratch transformer implements the complete architecture from "Attention Is All You Need":

- Multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Sinusoidal positional encoding

## Evaluation Metrics

- **CER** (Character Error Rate): Primary metric for ASR evaluation
- **WER** (Word Error Rate): Secondary metric for ASR evaluation

Metrics are calculated using the `jiwer` library.

## Results

Training results are saved in:

- `checkpoints/pretrained_transformer/`: Pretrained model checkpoints and training history
- `checkpoints/transformer_scratch.pt`: From-scratch transformer checkpoint

Visualizations:

- Training curves (loss, CER, WER)
- Attention weight visualizations
- Example predictions

## Technical Report

See `Prosit3_Technical_Report.pdf` for detailed explanation of:

- Transformer architecture and "Attention Is All You Need" concepts
- Pretrained models and fine-tuning strategies
- Implementation details for both systems
- Results and analysis

## Presentation

See `Prosit3_Presentation.pdf` or `.pptx` for the 20-minute presentation covering:

- Introduction to transformers
- Architecture overview
- Pretrained model approach
- From-scratch transformer demonstration
- Results and comparison

## License

This project is part of ICS553 Deep Learning coursework.

## Authors

[Your Name]

## Acknowledgments

- AfriSpeech200 dataset: https://huggingface.co/datasets/intronhealth/afrispeech-200
- Transformer architecture: "Attention Is All You Need" (Vaswani et al., 2017)
- Pretrained models: OpenAI Whisper, Facebook Wav2Vec2
