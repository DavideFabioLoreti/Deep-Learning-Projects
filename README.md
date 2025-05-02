# Classificazione-dei-Tumori-Cerebrali-con-il-Deep-Learning
Questo progetto utilizza tecniche di Deep Learning per la classificazione di tumori cerebrali attraverso immagini MRI. Il modello è in grado di distinguere quattro categorie:
* Glioma
* Meningioma
* Senza Tumore
* Tumore Pituitario

# 📂 Struttura Dataset

Il dataset è organizzato in due parti principali (training e test), ciascuna divisa nelle seguenti classi:

```bash
dataset/
├── training/
│   ├── glioma
│   ├── meningioma
│   ├── notumor
│   └── pituitary
└── test/
    ├── glioma
    ├── meningioma
    ├── notumor
    └── pituitary
```
[Scarica il dataset completo su Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

# ⚙️ Tecnologie e Librerie

TensorFlow

Keras

NumPy

Matplotlib

Seaborn

Scikit-learn

# 📊 Risultati

Nel notebook troverai grafici dettagliati di accuratezza e perdita, una matrice di confusione e il report di classificazione che evidenziano la performance del modello.

# CIFAR10
Aggiunta file di modello di classificazione di cifar10, sono stati implementati 2 modelli, uno basico e il secondo più profondo.
