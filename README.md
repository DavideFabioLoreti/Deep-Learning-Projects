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

## 🧠 Esempio Aggiuntivo: Classificazione CIFAR-10 con due Modelli CNN

Come ulteriore esempio didattico, è incluso un modello di classificazione immagini basato sul dataset **CIFAR-10**, che contiene 60.000 immagini a colori 32x32 suddivise in 10 classi (aerei, automobili, uccelli, gatti, cervi, cani, rane, cavalli, navi e camion).

Il modello utilizza una rete neurale convoluzionale semplice per dimostrare i principi base dell'elaborazione immagini:

- **Input**: immagini 32x32x3  
- **Architettura**: più layer Conv2D + MaxPooling + Dense  
- **Funzione di attivazione**: ReLU + Softmax  
- **Ottimizzatore**: Adam  
- **Output**: 10 classi  

Questa sezione è utile per confrontare un dataset generico con quello medico in termini di preprocessing, complessità del modello e accuratezza.

Per eseguirlo, consulta il notebook `DEEP LEARNING.ipynb` incluso nella repository, che contiene entrambi i modelli descritti di seguito.

### ✅ Primo modello (base)

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### ✅ Secondo modello (ottimizzato)

```python
model2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

🔍 Questi miglioramenti hanno permesso di **ridurre la loss** e **incrementare l'accuratezza** rispetto al primo modello.

