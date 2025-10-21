
# 🩺 AI Stethoscope: Heart Sound Classification using LSTM

This project presents an **AI

https://github.com/user-attachments/assets/d3645da0-6a5a-4f9a-b2b1-0d60f647e357

-powered digital stethoscope** capable of distinguishing between **heart** and **lung** sounds and classifying them into specific disease categories. Using **Long Short-Term Memory (LSTM)** networks, the system analyzes physiological audio signals to detect conditions such as **asthma, pneumonia, murmurs, stenosis, and more**.


### 🫁 Lung Sounds:

* **Source:** [PhysioNet Respiratory Sound Database](https://physionet.org/)
* **Categories:** URTI, Healthy, Asthma, LRTI, Bronchiectasis, Pneumonia, Bronchiolitis
* **Total recordings:** ~8000
* **Average length:** ~10 seconds
* **Epochs:** 125

### ❤️ Heart Sounds:

* **Source:** [PhysioNet Heart Sound Database](https://physionet.org/) and Cleveland Heart Sound Dataset
* **Categories:** Normal, Aortic Stenosis (AS), Mitral Regurgitation (MR), Mitral Stenosis (MS), Mitral Valve Replacement (MVR)
* **Total recordings:** ~8000
* **Average length:** ~10 seconds
* **Epochs:** 100

## 🔬 2. Exploratory Data Analysis (EDA)

Before model training, extensive **EDA** was performed to understand data structure, quality, and patterns.

Key tasks included:

* Checking **class distribution** for imbalance
* Visualizing **waveforms** and **spectrograms** for pattern identification
* Analyzing **audio duration**, **sampling rate**, and **noise levels**
* Identifying and removing **corrupted or missing** files
* Calculating **statistical features**: mean, variance, skewness
* Plotting **MFCCs** to observe differences between normal and abnormal sounds


## 🩹 3. Dataset Classes

### ❤️ Cardiac Classes:

| Class                           | Description                                                                                     |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Aortic Stenosis (AS)**        | Narrowing of the aortic valve causing left ventricular hypertrophy and potential heart failure. |
| **Mitral Regurgitation (MR)**   | Backflow of blood into the left atrium due to incomplete mitral valve closure.                  |
| **Mitral Stenosis (MS)**        | Restricted blood flow from left atrium to ventricle due to mitral valve narrowing.              |
| **Mitral Valve Prolapse (MVP)** | Floppy leaflets bulge back into the atrium, possibly causing regurgitation.                     |
| **Normal**                      | Healthy heart with proper blood flow and valve function.                                        |

### 🫁 Lung Classes:

* URTI
* Healthy
* Asthma
* LRTI
* Bronchiectasis
* Pneumonia
* Bronchiolitis

---

## 🎯 4. Feature Extraction

Audio signals are preprocessed and transformed into meaningful numerical representations suitable for deep learning:

* **Mel-Frequency Cepstral Coefficients (MFCCs):** 13–30 coefficients per frame
* **Spectrograms:** Time-frequency representation highlighting murmurs and wheezes
* **Chroma features:** Harmonic content for cardiac sounds
* **Spectral centroid and bandwidth:** Frequency domain features
* **Statistical features:** Mean, standard deviation, skewness
* **Time-domain features:** Signal duration, peak amplitude


## 🧠 5. Model Selection: LSTM

We selected a **Long Short-Term Memory (LSTM)** neural network due to its ability to model **time-series data** and capture **long-term dependencies** in sequential signals.

Key advantages:

* Handles sequential audio features effectively
* Overcomes vanishing gradient problems common in standard RNNs
* Proven success in biomedical sound analysis (arrhythmia and respiratory disease detection)
* Excellent balance between performance and architectural simplicity

---

## 🏋️‍♂️ 6. Model Training

The LSTM model was trained on ~5000 labeled recordings of heart and lung sounds.

### 🛠️ Training Pipeline:

1. **Data Preprocessing:**

   * Noise removal (Butterworth filters)
   * Normalization and segmentation of signals
   * Signal separation (heart vs. lung)

2. **Feature Extraction:**

   * MFCCs, spectral features, and STFT representations

3. **Data Augmentation:**

   * Pitch shifting
   * Time stretching
   * Background noise injection

4. **Model Input:** Sequential feature vectors

5. **Training Parameters:**

| Parameter        | Value                     |
| ---------------- | ------------------------- |
| Optimizer        | Adam                      |
| Loss Function    | Categorical Cross-Entropy |
| Epochs           | 50–100                    |
| Batch Size       | 32                        |
| Validation Split | 20%                       |
| Train/Test Split | 80/20                     |


## 📏 7. Model Evaluation

The model was evaluated using multiple metrics to ensure accuracy, reliability, and generalization.

### 📈 Metrics:

| Metric                      | Purpose                                               |
| --------------------------- | ----------------------------------------------------- |
| **Accuracy**                | Overall correct classification rate (>90% target)     |
| **Precision**               | Reduces false positives                               |
| **Recall (Sensitivity)**    | Ensures correct identification of true positives      |
| **F1-Score**                | Balances precision and recall                         |
| **Confusion Matrix**        | Visual representation of actual vs. predicted classes |
| **ROC Curve & AUC**         | Performance across decision thresholds                |
| **k-Fold Cross-Validation** | Ensures robustness across different data splits       |

✅ **Results:**

* **Heart sound classification accuracy:** ~92–95%
* **Lung sound classification accuracy:** ~94–97%
* **Real-time testing accuracy:** ~92%

The trained model was successfully integrated into the digital stethoscope system, enabling **real-time disease detection** from live audio inputs.

---

## 📁 Project Structure

```
├── data/
│   ├── heart_sounds/          # Heart sound recordings
│   ├── lung_sounds/           # Lung sound recordings
│   ├── labels.csv
├── src/
│   ├── preprocess.py          # Data preprocessing
│   ├── feature_extraction.py  # Feature extraction scripts
│   ├── model.py               # LSTM model architecture
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation metrics
├── notebooks/
│   └── EDA.ipynb              # Exploratory data analysis
├── requirements.txt
├── main.py                    # Real-time classification script
└── README.md
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/AI-Stethoscope.git
cd AI-Stethoscope
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Preprocess and extract features:

```bash
python src/preprocess.py
python src/feature_extraction.py
```

### 2. Train the LSTM model:

```bash
python src/train.py
```

### 3. Evaluate the model:

```bash
python src/evaluate.py
```

### 4. Real-time prediction (with digital stethoscope):

```bash
python main.py
```

---

## 📊 Results Summary

| Category       | Accuracy | F1-Score |
| -------------- | -------- | -------- |
| Heart Sounds   | 92–95%   | ~0.93    |
| Lung Sounds    | 94–97%   | ~0.94    |
| Real-Time Data | ~92%     | ~0.91    |

---

## 🩺 Applications

* Intelligent digital stethoscopes
* Early detection of heart and respiratory diseases
* Telemedicine and remote diagnostics
* Clinical decision support tools

---

## 📚 References

[14] PhysioNet Heart Sound Database – [https://physionet.org](https://physionet.org)
[15] PhysioNet Respiratory Sound Database – [https://physionet.org](https://physionet.org)
[21] Human Heart Anatomy Reference – Medical Illustrations Database

---

## 🧪 Future Work

* Integrate **CNN-LSTM hybrid models** to improve spatial feature learning
* Expand dataset diversity for rare diseases
* Deploy as a **mobile or web-based diagnostic application**
