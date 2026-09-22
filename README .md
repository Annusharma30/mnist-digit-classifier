# MNIST Handwritten Digit Classifier

A machine learning project that trains a **Logistic Regression** model to recognize handwritten digits (0-9) using the classic **MNIST dataset**, and tests it against a custom real-world handwritten digit image.

> Submitted as part of the **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship 2026**, in association with **BharatCares**.

## Project Description

This project walks through a complete supervised machine learning pipeline:

1. Load the MNIST dataset (70,000 handwritten digit images, 28x28 pixels each) directly from OpenML.
2. Explore and visualize sample digits.
3. Preprocess the data (normalize pixel values to 0-1, split into train/test sets).
4. Train a Logistic Regression classifier (`scikit-learn`) on 60,000 training images.
5. Evaluate the model on 10,000 held-out test images using accuracy, a confusion matrix, a full classification report, and one-vs-rest ROC/AUC curves.
6. Save the trained model to disk with `joblib`.
7. Load the saved model and test it on an independent, real-world handwritten digit image (preprocessed with PIL and OpenCV).

**Result:** The model achieves **92.57% accuracy** on the MNIST test set, with per-class AUC scores above 0.98 for every digit. Testing on a real-world image (outside the MNIST distribution) revealed a misclassification, highlighting the gap between benchmark accuracy and real-world generalization — discussed further in the project report.

## Dataset

- **Name:** MNIST (`mnist_784`)
- **Source:** [OpenML - mnist_784](https://www.openml.org/d/554) (fetched via `sklearn.datasets.fetch_openml`)
- **Original creators:** Yann LeCun, Corinna Cortes, Christopher J.C. Burges — [MNIST Database homepage](http://yann.lecun.com/exdb/mnist/)
- **Size:** 70,000 grayscale images, 28x28 pixels (784 features), labeled 0-9

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Environment | Jupyter Notebook |
| Numerical computing | NumPy |
| Machine learning | scikit-learn (LogisticRegression, metrics) |
| Visualization | Matplotlib, Seaborn |
| Image processing | Pillow (PIL), OpenCV |
| Model persistence | joblib |

## Project Structure

```
.
├── AnnuSharma_MNIST_Digit_Classifier.ipynb   # Main project notebook (code)
├── requirements.txt                          # Python dependencies
├── AnnuSharma_ProjectReport.docx              # Full project report
├── README.md                                  # This file
└── mnist_logistic_model.pkl                   # Saved trained model (generated on run)
```

## Setup & Run Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/<Annusharma30>/<mnist-digit-classifier>.git
   cd <mnist-digit-classifier>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook AnnuSharma_MNIST_Digit_Classifier.ipynb
   ```

5. **Run all cells** in order. The notebook will:
   - Download the MNIST dataset (requires an internet connection on first run)
   - Train the Logistic Regression model
   - Generate evaluation plots (confusion matrix, ROC curves)
   - Save the trained model as `mnist_logistic_model.pkl`
   - Load a custom test image (`OIP.jpg`) and predict its digit

   > **Note:** To test on your own handwritten digit image, place an image file named `OIP.jpg` in the same folder as the notebook, or update the filename in the image-loading cells.

## Key Results

| Metric | Value |
|---|---|
| Test Accuracy | **92.57%** |
| Weighted Avg Precision | 0.93 |
| Weighted Avg Recall | 0.93 |
| Weighted Avg F1-score | 0.93 |
| Avg per-class AUC | > 0.98 |

Full metrics, confusion matrix, ROC curves, and analysis of the real-world image test are available in `AnnuSharma_ProjectReport.docx`.

## Future Improvements

- Apply MNIST-style centering/normalization to custom images to improve real-world prediction accuracy.
- Try more powerful models (Random Forest, SVM, or a CNN) for higher accuracy.
- Add data augmentation for better generalization to varied handwriting styles.
- Build a simple web app for real-time digit prediction.

## Author

**Annu Sharma**
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 — BharatCares
