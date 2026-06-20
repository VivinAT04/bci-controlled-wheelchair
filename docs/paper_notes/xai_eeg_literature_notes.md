# XAI and EEG Literature Review Notes

## Dissertation: BCI-Controlled Intelligent Wheelchair System

---

# Paper 1

## Exploring a Gradient-Based Explainable AI Technique for Time-Series Data: A Case Study of Assessing Stroke Rehabilitation Exercises

**Venue:** ICLR 2023 Workshop

### Objective

The paper investigates whether gradient-based saliency maps can explain neural network decisions on time-series healthcare data. The authors focus on stroke rehabilitation exercises and attempt to identify the specific temporal regions where compensatory movements occur.

### Dataset

| Item         | Details                                 |
| ------------ | --------------------------------------- |
| Participants | 15 post-stroke patients                 |
| Trials       | 300 rehabilitation exercise trials      |
| Labels       | Normal movement / Compensatory movement |
| Ground Truth | Frame-level annotations                 |

### Features

OpenPose was used to extract body-joint locations.

Tracked joints:

- Head
- Neck
- Shoulder
- Elbow
- Wrist

Movement was represented as displacement from the initial joint position.

### Model

**Feed-Forward Neural Network (FFNN)**

Architecture:

- Hidden Layer 1: 1024 neurons
- Hidden Layer 2: 512 neurons
- ReLU activation
- Softmax output

### Performance

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~98%  |

### Explainability Method

**Saliency Maps**

Procedure:

1. Compute gradients of model output with respect to input.
2. Convert gradients into frame-level importance scores.
3. Generate temporal saliency maps.
4. Highlight important regions associated with compensatory movement.

### Results

The saliency maps successfully highlighted:

- Head leaning
- Shoulder elevation
- Postural compensation

| Experiment               | Recall | F2 Score |
| ------------------------ | ------ | -------- |
| All Trials               | 0.44   | 0.44     |
| No Padding               | 0.44   | 0.44     |
| Compensatory Trials Only | 0.96   | 0.91     |

### Contributions

- Demonstrated saliency maps on time-series data.
- Localized important temporal regions.
- Reduced need for manual video review.
- Showed weakly-supervised temporal localization.

### Limitations

- Small dataset
- Single model architecture
- Threshold selected empirically
- Limited validation

### Relevance to Dissertation

Useful for demonstrating that:

- XAI can be applied to sequential signals.
- Temporal explanations are possible.
- Similar techniques could be applied to EEG classification.

**Potential adaptation:**

```text
EEG → Classifier → Saliency Map → Important EEG Segments
```

---

# Paper 2

## Towards Best Practice of Interpreting Deep Learning Models for EEG-Based Brain–Computer Interfaces

**Venue:** Frontiers in Computational Neuroscience, 2023

### Objective

To determine which explainability methods are trustworthy for EEG-based deep learning systems.

### Datasets

#### BCI Competition IV-2a

| Item     | Details                             |
| -------- | ----------------------------------- |
| Subjects | 9                                   |
| Channels | 22 EEG                              |
| Task     | Motor Imagery                       |
| Classes  | Left Hand, Right Hand, Feet, Tongue |

#### Error-Related Negativity Dataset

- 26 subjects

#### Driver Drowsiness Dataset

- 27 subjects

### Models Evaluated

#### EEGNet

Widely used CNN architecture for EEG.

#### InterpretableCNN

CNN designed specifically for interpretability.

### XAI Methods Compared

1. Saliency Maps
2. Deconvolution
3. Guided Backpropagation
4. Gradient × Input
5. Integrated Gradients
6. Layer-wise Relevance Propagation (LRP)
7. DeepLIFT

### Key Findings

#### Reliable Methods

- Gradient × Input
- Integrated Gradients
- LRP
- DeepLIFT

#### Unreliable Methods

- Saliency Maps
- Deconvolution
- Guided Backpropagation

### Neurophysiological Validation

#### Motor Imagery

- ERD
- ERS
- Sensorimotor rhythms

#### Error Detection

- Error-related negativity

#### Drowsiness

- Alpha spindle activity

### Major Contribution

The paper demonstrates that explanation quality must be validated rather than visually inspected.

Many popular EEG explanation techniques can be misleading.

### Recommendations

**Use:**

- LRP
- Integrated Gradients
- DeepLIFT
- Gradient × Input

**Avoid relying solely on:**

- Saliency Maps

### Relevance to Dissertation

Most important paper reviewed.

Provides evidence-based justification for using:

- LRP
- Integrated Gradients

for explaining EEG classifier decisions.

---

# Paper 3

## Explainable Artificial Intelligence (XAI) for EEG Analysis: A Survey on Recent Trends and Advancements

**Type:** Survey Paper (2026)

### Objective

Comprehensive survey of XAI methods used in EEG research.

Reviews:

- EEG datasets
- Machine learning
- Deep learning
- Explainability techniques
- Applications
- Challenges

### Evolution of EEG Classification

```text
Traditional ML
├── LDA
├── SVM
└── Random Forest

↓

Deep Learning
├── CNN
├── LSTM
├── Transformer
└── Foundation Models
```

### Categories of XAI

#### Feature Attribution

- SHAP
- LIME
- LRP
- DeepLIFT

Purpose:

> Identify influential features.

#### Visualization

- Grad-CAM
- Saliency Maps

Purpose:

> Identify important regions of input.

#### Rule-Based Methods

Interpretable decision rules.

#### Example-Based Methods

Representative examples used for explanation.

### Most Popular EEG XAI Methods

1. SHAP
2. LIME
3. Grad-CAM
4. LRP
5. DeepLIFT

### Applications

- Motor Imagery BCI
- Epilepsy Detection
- Sleep Analysis
- Emotion Recognition
- Depression Detection
- Stroke Prediction

### Challenges

- Lack of benchmark datasets
- Poor explanation validation
- Limited neuroscientific grounding
- Low robustness

### Major Contribution

Provides a taxonomy of EEG explainability methods.

Serves as a strong survey reference for the literature review.

### Relevance to Dissertation

Excellent source for:

**Chapter 2 → Explainable AI in EEG-Based Brain–Computer Interfaces**

---

# Paper 4

## EEG Seizure Detection using Convolutional Neural Network With Grad-CAM

**Venue:** TechRxiv Preprint, 2024

### Objective

Develop a CNN-based EEG seizure detector and explain predictions using Grad-CAM.

### Dataset

#### CHB-MIT EEG Dataset

| Item           | Details                    |
| -------------- | -------------------------- |
| Patients       | 22                         |
| EEG Recordings | 664                        |
| Seizures       | 198                        |
| Source         | Boston Children's Hospital |

### Preprocessing

| Step                   | Value     |
| ---------------------- | --------- |
| Original Sampling Rate | 256 Hz    |
| Resampled Rate         | 128 Hz    |
| Window Length          | 8 seconds |
| Overlap                | 4 seconds |

### CNN Architecture

Components:

- Convolution Layers
- Pooling Layers
- Dense Layers
- Dropout Layers

Training:

| Parameter     | Value                |
| ------------- | -------------------- |
| Optimizer     | Adam                 |
| Learning Rate | 0.0001               |
| Loss Function | Binary Cross Entropy |

### Explainability Method

#### Grad-CAM

Procedure:

1. Compute gradients from final convolution layer.
2. Weight feature maps.
3. Produce heatmap.
4. Highlight regions influencing seizure detection.

### Benefits of Grad-CAM

Provides:

- Transparency
- Interpretability
- Clinical trust

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Limitations

- Preprint (not peer-reviewed)
- Seizure detection only
- No comparison with EEGNet or modern architectures
- Limited XAI validation

### Relevance to Dissertation

Moderately relevant.

Useful because it demonstrates:

- EEG explainability
- Grad-CAM implementation
- CNN interpretation workflow

---

# Overall Dissertation Takeaways

## Most Important Paper

### Paper 2 (Frontiers 2023)

Reason:

- LRP
- Integrated Gradients
- DeepLIFT
- Gradient × Input

were shown to be more reliable than Saliency Maps.

---

## Best Survey

### Paper 3 (2026 Survey)

Reason:

Provides a comprehensive overview of EEG explainability research.

---

## Practical XAI Example

### Paper 4 (Grad-CAM Seizure Detection)

Reason:

Demonstrates real-world implementation of EEG explainability.

---

# Proposed Dissertation XAI Pipeline

```text
EEG Signals
      │
      ▼
Preprocessing
      │
      ▼
CSP / FBCSP / EEGNet
      │
      ▼
Classifier
      │
      ▼
XAI Module
(LRP / Integrated Gradients)
      │
      ▼
Explain Left / Right / Forward / Stop Decisions
```
