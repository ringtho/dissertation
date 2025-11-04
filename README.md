# Forex Forecasting with LSTM & Temporal Fusion Transformer

This repository accompanies a research project comparing two deep-learning approaches—**Long Short-Term Memory (LSTM)** networks and the **Temporal Fusion Transformer (TFT)**—for one–day-ahead forecasting of the EUR/USD exchange rate. It contains executable Jupyter notebooks, reproducible data-processing pipelines, and guidance for running the experiments on Google Colab with GPU acceleration.

## Repository Layout

```
.
├── LSTM_forecasting.ipynb      # End-to-end LSTM workflow (data prep → training → evaluation)
├── TFT_forecasting.ipynb       # End-to-end TFT workflow with interpretability analyses
├── data/
│   └── eurusd_daily.csv        # Daily EUR/USD OHLCV data (can be regenerated via yfinance)
├── experiments/                # Output artefacts (created at runtime, .gitkeep only)
├── checkpoints/                # Saved model weights/checkpoints (created at runtime)
├── lightning_logs/             # PyTorch Lightning logs (created at runtime)
├── requirements-lstm.txt       # Minimal package set for running the LSTM notebook locally
├── requirements-tft.txt        # Minimal package set for running the TFT notebook locally
└── README.md
```

> **Note:** The `experiments/`, `checkpoints/`, and `lightning_logs/` folders are kept empty besides a `.gitkeep` placeholder so the repository stays lightweight. They will be populated automatically when you run the notebooks.

## Running the Notebooks

### Preferred Option – Google Colab (GPU Ready)

Both notebooks have a first code cell named **“Environment setup (Colab friendly)”**. When run inside Colab it will:

1. Detect the Colab runtime and install the required packages (TensorFlow for the LSTM notebook, PyTorch/Lightning for the TFT notebook) with GPU-enabled wheels.
2. Leave a summary of the pinned versions so results are reproducible.

To use Colab:

1. Upload the repository (or open the notebooks directly via GitHub → “Open in Colab”).
2. Go to **Runtime → Change runtime type** and select **GPU**.
3. Run the setup cell, then execute the rest of the notebook top to bottom.

The notebooks will create fresh outputs under `experiments/` and, when `RUN_TRAINING=True`, will train models from scratch.

### Local Execution (Advanced)

If you prefer to run locally:

1. Create dedicated virtual environments for each framework to avoid dependency conflicts (TensorFlow currently requires `numpy<2`, while the latest PyTorch stack is happy with newer NumPy versions).
2. Install the packages from the corresponding requirements file:

   ```bash
   # LSTM environment
   python -m venv .venv-lstm
   source .venv-lstm/bin/activate
   pip install -r requirements-lstm.txt

   # TFT environment
   python -m venv .venv-tft
   source .venv-tft/bin/activate
   pip install -r requirements-tft.txt
   ```

3. Launch Jupyter Lab/Notebook from within each environment and execute the desired notebook.

Because TensorFlow and PyTorch have conflicting binary dependencies on macOS (especially around NumPy), using separate environments—or Colab—is strongly recommended.

## Data

- The repository includes `data/eurusd_daily.csv`, a cleaned OHLCV dataset for EUR/USD.  
- Both notebooks can optionally download the latest data via `yfinance` if you toggle the relevant code paths; if you are offline, keep the CSV as-is.

## Outputs & Reports

- `experiments/lstm/` and `experiments/tft/` receive metrics (`report.json`), prediction CSVs, plots, and feature-importance tables after each run.
- Model checkpoints (TensorFlow `.keras` files or PyTorch state dictionaries) are written to `checkpoints/` when training is enabled.
- PyTorch Lightning logs are placed under `lightning_logs/` and can be inspected with TensorBoard if desired.

All of these directories are ignored by Git except for their `.gitkeep` placeholders, ensuring you don’t accidentally commit large artefacts.

## Reproducing the Research

1. Open `LSTM_forecasting.ipynb`, run the setup cell, and execute the pipeline (set `RUN_TRAINING=True` to retrain; disable it if you only want to inspect cached outputs).
2. Open `TFT_forecasting.ipynb`, run the setup cell (enable GPU if available), and execute the pipeline. Optional interpretation plots can be toggled via `RUN_TFT_INTERPRETATION`.
3. Compare the generated `report.json` files in `experiments/lstm/` and `experiments/tft/` for metrics such as RMSE, MAPE, directional accuracy, and Diebold–Mariano statistics.

## Contributing / Extending

- Add new features or alternative models by branching from the notebooks or porting the pipelines into Python scripts.
- If you modify the data schema, document the changes in this README and ensure the notebooks handle the new columns gracefully.
- Pull requests are welcome; keep generated artefacts out of commits and update the documentation when behaviour changes.

## License

Specify a license (e.g., MIT, Apache 2.0) before publishing on GitHub. If none is set, GitHub treats the project as “all rights reserved” by default.

---

For questions or improvements, open an issue or start a discussion once the repository is live on GitHub.
