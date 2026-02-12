# Acoustic Analysis of Pathological Voices

Automatic detection of pathological voices using acoustic feature analysis and Support Vector Machine (SVM).

This project was developed as part of the M.Tech dissertation at Tezpur University.

---

## Problem Statement

Voice disorders affect vocal fold vibration patterns, producing measurable deviations in acoustic parameters.

This system classifies voice samples into:

- Healthy
- Pathological

using signal-derived acoustic features and machine learning.

---

## Dataset

- Total samples: 208
- Healthy: 57
- Pathological: 151
- Sampling Rate: 8000 Hz
- Recording: Sustained vowel /a/

Each sample produces a 12-dimensional feature vector.

---

## Feature Extraction

The following acoustic features are used:

### Jitter Parameters
- Jitt
- Jitta
- RAP
- PPQ5
- DDP

### Shimmer Parameters
- Shim
- ShdB
- APQ3
- APQ5
- APQ11
- DDA

### Harmonic Parameter
- mHNR (Mean Harmonics-to-Noise Ratio)

Total features per sample: 12

---

## Methodology

1. Load dataset
2. Normalize features
3. Train SVM classifier (Sigmoid kernel)
4. Evaluate using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

---

## Model

Support Vector Machine (SVM)

Kernel:
Sigmoid

Decision Function:
f(x) = sign(w · x + b)

---

## Results

Accuracy: 73.13%

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Unhealthy | 0.53 | 0.42 | 0.47 |
| Healthy | 0.79 | 0.85 | 0.82 |

---

## Future Work

- Feature selection optimization
- Cross-validation
- Deep learning-based classification
- Multi-class pathology detection
- Real-time deployment system

---

## Author

Shruti Khaklary  
M.Tech – Electronics Design and Technology  
Tezpur University
