# Scribble AI

Scribble AI is an interactive doodle-recognition app built with Streamlit and PyTorch. Users draw on a browser canvas, the drawing is resized to a 28x28 grayscale image, and a convolutional neural network predicts the doodle class.

## Features

- Streamlit drawing canvas for live doodle input
- Google QuickDraw data downloader
- PyTorch CNN training pipeline
- Saved model checkpoints with class labels
- End-to-end prediction from user drawing to model output

## Project Structure

```text
.
|-- app.py                 # Streamlit frontend and inference flow
|-- download_data.py       # Google QuickDraw image downloader
|-- inference.py           # Image preprocessing and prediction helpers
|-- model.py               # CNN architecture and model save/load helpers
|-- train.py               # PyTorch training script
|-- pyproject.toml         # Project metadata and direct dependencies
|-- uv.lock                # Reproducible dependency lockfile
`-- README.md
```

Generated files are intentionally ignored:

```text
data/                    # downloaded QuickDraw images
models/                  # trained model checkpoints
.quickdrawcache/         # QuickDraw binary cache
```

## Setup

```bash
uv python install
uv sync
```

## Download Training Data

Download a small starter dataset:

```bash
uv run python download_data.py --classes ant cat dog --max-drawings 1000 --stroke-widths 4 5
```

This creates class folders under:

```text
data/quickdraw/
```

Example:

```text
data/quickdraw/ant
data/quickdraw/cat
data/quickdraw/dog
```

## Train the Model

```bash
uv run python train.py --data-dir data/quickdraw --epochs 20
```

The trained checkpoint is saved to:

```text
models/quickdraw_model.pt
```

The checkpoint includes both model weights and class names, so the Streamlit app can display readable predictions.

## Run the App

```bash
uv run streamlit run app.py
```

Then draw a doodle and click **Predict**.

If no trained model is found, the app will show the training command needed to create one.

## Current Model

A small local demo model can be trained with three classes:

```bash
uv run python download_data.py --classes ant cat dog --max-drawings 30 --stroke-widths 4 5
uv run python train.py --data-dir data/quickdraw --epochs 5
```

This is enough to test the full pipeline, but it is not meant to be highly accurate. For a portfolio demo, use more drawings per class and train for more epochs.

## Tech Stack

- Python
- Streamlit
- PyTorch
- Pillow
- Google QuickDraw dataset

## Notes

The model expects 28x28 grayscale images because the QuickDraw training images are generated at that size. The app applies the same preprocessing to user drawings before inference.
