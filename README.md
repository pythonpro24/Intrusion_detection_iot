
```markdown
# Efficient Predefined Time Adaptive Neural Network based Cryptography Framework espoused Intrusion Detection for Securing IoT Network

## Project Overview

This project focuses on implementing a secure intrusion detection system (IDS) for IoT networks using an **Efficient Predefined Time Adaptive Neural Network (PTANN)** and integrating **Black-Winged Kite** and **Wolf-Bird Optimizer algorithms** for optimization. Additionally, **Low-Complexity Elliptic Galois Cryptography (LCEGC)** is implemented to ensure secure data transmission.

The IDS is built using the **NSL-KDD dataset**, which classifies network traffic into **benign** and **malicious** categories. The system detects intrusions with high accuracy and secures data communication using cryptography.

### Key Features:
- **Efficient Predefined Time Adaptive Neural Network (PTANN)** for intrusion detection.
- **Black-Winged Kite Algorithm** and **Wolf-Bird Optimizer Algorithm** for model optimization.
- **Low-Complexity Elliptic Galois Cryptography (LCEGC)** for secure data transmission.
- Evaluation using metrics such as **accuracy**, **precision**, **F1-score**, and **ROC-AUC**.
- Visualization of the **Convergence Curve** for fitness during training.

---

## Project Structure

```bash
├── dataset
│   └── kdd_train.csv         # NSL-KDD dataset for training
├── models
│   └── eptann.h5             # Pre-trained model
├── cryptography
│   └── LCEGC.py              # Implementation of Elliptic Galois Cryptography
├── notebooks
│   └── EfficientPTANN.ipynb  # Main Jupyter Notebook for training and evaluating PTANN
├── results
│   └── performance_metrics.png  # Model performance plots (accuracy/loss)
├── README.md                 # Project documentation (this file)
```

---

## Dependencies

To install the dependencies for this project, use the following command:

```bash
pip install -r requirements.txt
```

### Main Libraries:
- `numpy`
- `pandas`
- `tensorflow` / `keras`
- `matplotlib`
- `seaborn`
- `sklearn`
- `cryptography`

---

## Data

The project uses the **NSL-KDD** dataset for network traffic classification, which is stored in `kdd_train.csv`. You can download this dataset from [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html).

The dataset contains the following columns:
- **Numeric features**: `duration`, `src_bytes`, `dst_bytes`, `wrong_fragment`, etc.
- **Categorical features**: `protocol_type`, `service`, `flag`, etc.
- **Label**: The target column `labels` indicates whether the traffic is normal or a specific attack type.

---

## Preprocessing

The data is preprocessed by encoding categorical features using **one-hot encoding** and normalizing numeric features using **StandardScaler**.

### Steps:
1. **Data Splitting**: The dataset is split into **training** and **test** sets (80% training, 20% testing).
2. **Scaling**: Features are scaled for better convergence using **StandardScaler**.
3. **3D Transformation**: The data is reshaped to a 3D format for input into the Convolutional Neural Network (CNN).

---

## Model Training

The **Efficient Predefined Time Adaptive Neural Network (PTANN)** is trained using the following configuration:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Activation**: ReLU for hidden layers, Sigmoid for the output layer
- **Callbacks**: Early Stopping (patience=5)

The model consists of:
- 2 **Conv1D** layers followed by **MaxPool1D**, **BatchNormalization**, and **Dropout** layers.
- A **Dense** layer for final classification with Sigmoid activation.

The **Black-Winged Kite Algorithm** and **Wolf-Bird Optimizer Algorithm** were used to further optimize the model during training.

---

## Cryptography: Secure Data Transmission

To ensure secure data transmission, **Low-Complexity Elliptic Galois Cryptography (LCEGC)** is implemented using AES encryption. Data is encrypted before transmission and decrypted upon reception to maintain confidentiality and detect potential attacks.

### Encryption/Decryption:
- **AES** with CBC mode is used for encryption and decryption.
- Random **IV** (Initialization Vector) is generated for each session to ensure encryption security.

---

## Model Performance

The model is evaluated using the following metrics:
- **Accuracy**: Measures the percentage of correctly classified samples.
- **F1-Score**: Balances precision and recall.
- **Precision**: Measures the proportion of true positives among predicted positives.
- **Recall**: Measures the proportion of true positives among actual positives.
- **ROC-AUC**: Represents the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

---

## Results

The model achieved **99.05% accuracy** on the test set, with a **0.0245 loss**. The performance of the model across epochs is visualized in the form of accuracy and loss plots.

![Model Performance](results/performance_metrics.png)

---

## Future Work

- Implement additional **cryptographic protocols** for enhanced data security.
- Apply the framework to other datasets for broader evaluation.
- Explore different deep learning architectures such as **Recurrent Neural Networks (RNN)** for temporal analysis.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

- **Jehu Jershone**
```

You can adapt this README file further to include more project-specific details if needed.
