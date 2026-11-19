# Logistic Regression Pipeline

IEEE PUA · AI & Academic Learning Track (Level 2) — Practical Task, Part 1.

End-to-end Logistic Regression on the Wisconsin Breast Cancer Diagnostic dataset (see `data/dataset_details.md`): load → clean → scale/split → train → verify maths with custom Sigmoid / Softmax / Cross-Entropy → 2D decision boundary → probability histogram.

## Project structure
```
logistic_regression_pipeline/
├── notebook_1.ipynb      # full pipeline (executed, outputs included)
├── requirements.txt
├── README.md
├── data/dataset_details.md
└── figures/
    ├── class_distribution.png
    ├── feature_distributions.png
    ├── sigmoid_softmax_crossentropy.png
    ├── decision_boundary_2d.png
```

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebook_1.ipynb
```

## Results
- Test accuracy: **96.5%** (30-feature model); **95.6%** with only 2 features (`worst perimeter`, `worst concave points`) used for the boundary plot.
- Custom `sigmoid` matches `predict_proba`; custom binary cross-entropy (0.0773) matches `sklearn.metrics.log_loss`; custom `softmax` matches SciPy and reduces to sigmoid for 2 classes.

## Key takeaways
1. No missing values or duplicates; ~37% malignant, so a stratified split keeps class ratios stable.
2. The scaler is fit on training data only to avoid leakage; scaling helps the solver converge and makes coefficients comparable.
3. Cross-entropy punishes confident wrong predictions sharply (loss → ∞ as p → 0 for y = 1).
4. Logistic Regression's boundary is linear (`wᵀx + b = 0`); two features already separate the classes well.

Evaluation of this model (confusion matrix, ROC-AUC, thresholds) is in the companion repo `classification_metrics_analysis`.
