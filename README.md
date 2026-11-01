# An Efficient Botnet Attack Detection in IoT Environment using Advanced Machine Learning Models

This repository contains the project implementation associated with the IJERT paper *An Efficient Botnet Attack Detection in IoT Environment using Advanced Machine Learning Models* (IJERTV15IS080644, August 2026).

## Research paper

- [Published paper (PDF)](paper/IJERTV15IS080644.pdf)
- [IJERT publication page](https://www.ijert.org/an-efficient-botnet-attack-detection-in-iot-environment-using-advanced-machine-learning-models-ijertv15is080644)

## Repository contents

- `BotnetDetection.ipynb` — original notebook.
- `attention.py` — attention component used by the original implementation.
- `test.py` — original application/test script.
- `Dataset/` — supplied UNSW-NB15 dataset files.
- `model/` — existing model weights and training artifacts.
- `templates/`, `static/` — original web interface assets.
- `classical_ml/` — additional LR, SVM, Random Forest, and hard-majority-voting implementation aligned with the paper's abstract and Section III-E.
- `paper/` — published IJERT paper.

The original notebook, application files, and model artifacts are retained. The `classical_ml/` directory is an additional implementation; it does not replace the original work.

## Classical ML implementation

The paper describes Logistic Regression, Support Vector Machine, and Random Forest classifiers combined through majority voting. The added implementation follows that flow with preprocessing, correlation-based redundancy filtering, Random-Forest feature-importance selection, training-only SMOTE, and held-out evaluation.

See [`classical_ml/README.md`](classical_ml/README.md) for setup, methodology, measured results, and the distinction between this reproducible run and the approximate accuracies reported in the paper.

Run from the repository root:

```bash
python -m pip install -r classical_ml/requirements.txt
python classical_ml/train_evaluate.py
```

## Citation

Golanakonda, S., Nallamala, S., & Rajalingari, H. R. (2026). *An Efficient Botnet Attack Detection in IoT Environment using Advanced Machine Learning Models*. International Journal of Engineering Research & Technology (IJERT), 15(08), IJERTV15IS080644.

## Notes

- The dataset is included as supplied in the project archive. Check the dataset's terms before redistributing it independently.
- Do not commit credentials, local environment folders, or private data.
- The original root `requirements.txt` is retained for the original environment. The separate `classical_ml/requirements.txt` installs dependencies for the added classical-ML experiment.
