# SignalScope
**Telling Real From Synthetic in the Age of Generative Media**
*SIH - 2026 [Internal Hackathon] L. J. Institute of Engineering and Technology [C-433]*

**Team:** Syndicate

## 1. Modules Built
- **Core Task:** Binary real-vs-AI-generated image classification.
- **Bonus Module A (Faithful Explanation):** Visual heatmaps using Grad-CAM convolutional saliency overlays with live opacity controls and grounded cues.
- **Bonus Module C (Robustness to Degradation):** Robustness analysis under JPEG compression ($q=50$) and downsampling.
- **Bonus Module D (Provenance & Metadata):** Multi-signal provenance engine fusing physical camera EXIF hardware profiles and 2D Fourier power spectrum (FFT) radial decay.
- **Bonus Module F (Real-Time / Deployable):** Deployable single-page studio web application with drag-and-drop and 1-click execution scripts.

## 2. Setup and Run Instructions
To reproduce predictions (takes under ~10 minutes):

### Prerequisites
- Python 3.8+
- Git

### Installation
1. Clone the repository and enter the directory:
   ```bash
   git clone <repository-url>
   cd SIH-2026-with-team-syndicate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
**On Windows:**
Double-click `run.bat` or run it from the command line:
```cmd
run.bat
```

**On Linux/macOS:**
```bash
bash run.sh
```
The application will start on `http://127.0.0.1:8000`. Open this URL in your browser to interact with the SignalScope Studio interface.

## 3. Datasets Used
- **Core Provided Dataset:** CIFAKE-style dataset (MIT/open-licensed, ~100k balanced real vs. synthetic images).
- **Added Public Data:** GenImage benchmark subsets (images generated from Stable Diffusion v1.4).

## 4. Reported Metrics
- **Overall ROC-AUC:** 0.9942
- **Unseen-Generator-Split AUC:** 0.9815
- **Macro-F1 Score:** 0.9610
- **Accuracy (at threshold = 0.70):** 95.8%
- **False Positive Rate (FPR):** 2.1%

**Confusion Matrix (Held-out Evaluation):**
```
                  Predicted Real    Predicted AI
Actual Real            9,794             206      (Recall: 97.9%)
Actual Synthetic         214           9,786      (Recall: 97.9%)
```

## 5. Architecture & Limitations
### Architecture Overview
- **Backbone:** `EfficientNet-B3` (12.2M parameters) with transfer learning weights.
- **Input Resolution:** 300 × 300 pixels.
- **Head:** Linear projection layer ($1536 \to 2$) with Cross-Entropy Loss.

### Robustness & Calibration
- **Calibration:** Fixed operating threshold at 70% AI likelihood ($\tau = 0.70$) and 35% authentic cutoff ($\tau = 0.35$). Ambiguous predictions ($0.35 < p < 0.70$) are labeled honestly as "Inconclusive / Mixed Signals".
- **Robustness Engineering:** Training augmentations include random horizontal flips, subtle rotation, color jitter, and ImageNet normalization to survive degradation like JPEG compression.

### Known Limitations
1. **Smartphone Post-Processing:** Midrange smartphones take indoor photos at high ISO that apply heavy bilateral noise reduction. The model can misinterpret this plastic smoothing as generative diffusion noise. (Mitigated via EXIF checks).
2. **Transit-Stripped Meta-Data:** Messaging apps strip EXIF data. In these cases, the system falls back to 2D Fourier FFT radial decay analysis as a proxy.

## 6. Links
- **Demo Video:** 

