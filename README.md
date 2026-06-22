---
title: Online Student Engagement Detector
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app/app.py
pinned: false
python_version: "3.10"
---

# Online Student Engagement Detection

An end-to-end deep learning framework designed to predict student engagement levels in online learning environments using facial landmark sequences extracted from the **DAiSEE** dataset.

The system processes video sequences, extracts face structures using **MediaPipe FaceMesh**, and makes predictions using a **Soft-Voting Ensemble** combining a **Spatial-Temporal Graph Convolutional Network (ST-GCN)**, a **Bidirectional Gated Recurrent Unit (Bi-GRU)**, and a **ConvNeXt1D** sequence classifier.

---

## 🔬 1. Project Architecture & Methodology

The pipeline integrates spatial structural modeling, sequential temporal modeling, and convolutional feature hierarchy extraction:

```
[ Input Video ] ──► [ MediaPipe FaceMesh ] ──► [ Normalized Landmarks (300, 468, 2) ]
                                                            │
                       ┌────────────────────────────────────┼───────────────────────────────────┐
                       ▼                                    ▼                                   ▼
               [ ST-GCN Branch ]                     [ Bi-GRU Branch ]                  [ ConvNeXt1D Branch ]
          (Spatial-Temporal Graph)                (Bi-Directional RNN)                 (1D Depthwise Convs)
                       │                                    │                                   │
                       ▼                                    ▼                                   ▼
               [ Softmax Probs ]                    [ Softmax Probs ]                   [ Softmax Probs ]
                       │                                    │                                   │
                       └────────────────────────────────────┼───────────────────────────────────┘
                                                            ▼
                                                [ Average Probability ]
                                                            │
                                                            ▼
                                               [ Final Engagement Class ]
                                        (0: Disengaged, 1: Low, 2: Mid, 3: High)
```

### Key Models:
1. **Feature Extractor**: Processes videos to extract a sequence of `(300, 468, 2)` NumPy arrays tracking 468 2D facial landmarks over a normalized sequence length of 300 frames.
2. **ST-GCN**: Interprets the 468 landmarks as nodes of a graph, connecting them spatially based on facial structure and temporally across consecutive frames.
3. **Bi-GRU**: Flattens coordinates to 936 dimensions per frame and learns forward and backward sequential dependencies.
4. **ConvNeXt1D**: Employs deepwise convolutions, LayerNorm, and GELU activations to extract hierarchically grouped temporal features.
5. **Soft Voting Ensemble**: Combines the predictions of all three networks by averaging their output probability distributions, improving model generalizability and robustness.

---

## 🛠️ 2. Project Implementation History (What Happened)

Here is a summary of the implementation phases and optimization fixes:
* **Preprocessing & Landmark Extraction**: Videos from the DAiSEE dataset were processed using MediaPipe FaceMesh to generate structured `.npy` landmark files and a clean `labels.csv` index mapping.
* **Robust PyTorch Loader**: Created `utils/dataset.py` with custom, stratified splits. Added fallback logic to standard splits when handling limited datasets (avoiding minority class stratification crashes) and computed robust inverse-frequency class weights (ensuring class weights tensor shape is always `[4]`).
* **Optimized Training Loop**: Modified all training scripts to support training resume flags (`--resume <ckpt_path>`) which preserve epochs and optimizer state, and dataloader worker controls (`--num_workers`) for T4 GPU parallelization.
* **Interactive Evaluation**: Configured `evaluate.py` to output confusion matrices and compile comparative performance logs to `evaluation_results/final_evaluation_report.md`. Added support to load JSON training histories (`*_history.json`) and plot accuracy/loss curves.
* **Modern Web App**: Implemented absolute path resolution in `app/app.py` to run Streamlit from any working directory, with a fallback mechanism from `*_best.pth` checkpoints to `*_final.pth`, and updated the sidebar indicators to display the exact filename loaded.

---

## 📂 3. Repository Structure

```
student-engagement/
├── app/
│   └── app.py                  # Streamlit Web Dashboard (Visual UI & Real-time Inference)
├── checkpoints/                # Model checkpoints (stores weights & json history logs)
├── dataset/                    # DAiSEE metadata, e.g. CSVs
├── evaluation_results/         # Output directory for confusion matrices & report
├── models/
│   ├── stgcn.py                # ST-GCN Classifier Architecture
│   ├── bigru.py                # Bi-GRU Sequence Architecture
│   └── convnext1d.py           # ConvNeXt 1D Architecture
├── utils/
│   ├── dataset.py              # PyTorch Dataset Loader
│   └── extract_features.py     # MediaPipe FaceMesh Extractor
├── COLAB_TRAINING.ipynb        # GPU Training Notebook for Google Colab
├── ensemble.py                 # Soft-Voting Ensemble Logic
├── evaluate.py                 # Evaluation suite (plots curves & writes final reports)
├── preprocess.py               # Preprocessing script
├── requirements.txt            # Project dependencies
└── verify_and_dry_run.py       # One-batch dry-run verification script
```

---

## 🏃 4. Running the Project

### Local Installation & Verification
1. Clone the repository and install the dependencies:
   ```bash
   git clone https://github.com/Shivani367/student-engagement.git
   cd student-engagement
   pip install -r requirements.txt
   ```
2. Verify the pipelines and model shapes using the dry-run script:
   ```bash
   python verify_and_dry_run.py
   ```

### Full GPU Training on Google Colab
Since training on the full 5,358 samples requires a GPU, follow these steps:
1. Compress your local `processed/` directory (containing `.npy` files and `labels.csv`) into **`processed.zip`**.
2. Upload `processed.zip` to a folder named `student-engagement` in your Google Drive (`My Drive/student-engagement/processed.zip`).
3. Open Google Colab, upload [COLAB_TRAINING.ipynb](COLAB_TRAINING.ipynb), select the **T4 GPU** runtime, and run the cells.
4. The notebook will download your GitHub code, extract the dataset, train all three models in parallel, plot learning curves, run the ensemble, and back up your checkpoints to Google Drive results.

### Deploying the Streamlit Web Application
1. Download your best model checkpoints (`stgcn_best.pth`, `bigru_best.pth`, and `convnext_best.pth`) from your Google Drive backup directory.
2. Put these files into your local project's **`checkpoints/`** folder.
3. Start the dashboard:
   ```bash
   streamlit run app/app.py
   ```
4. Open the local address in your browser (usually `http://localhost:8501`) to upload student videos and view predicted engagement classifications.
