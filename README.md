# Introduction aux Modèles GAN (Generative Adversarial Networks)

[![CI](https://github.com/username/Introduction-to-GAN-model/actions/workflows/ci.yml/badge.svg)](https://github.com/username/Introduction-to-GAN-model/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

Ce projet implémente un DCGAN (Deep Convolutional Generative Adversarial Network) pour la génération d'images d'anime en utilisant TensorFlow 2.x.

## 🚀 Fonctionnalités

- Architecture DCGAN moderne et optimisée
- Génération d'images d'anime de haute qualité
- Sauvegarde automatique des checkpoints
- Visualisation des résultats en temps réel
- Interface simple d'utilisation

## 🏗️ Architecture

### Discriminateur
- Architecture convolutive profonde
- Couches de BatchNormalization pour la stabilité
- Dropout pour réduire le surapprentissage
- Activation LeakyReLU

### Générateur
- Couches de déconvolution (Conv2DTranspose)
- BatchNormalization
- Activation LeakyReLU
- Sortie avec activation tanh

## 📋 Prérequis

- Python 3.8 ou supérieur
- TensorFlow 2.4.0 ou supérieur
- CUDA (recommandé pour l'entraînement sur GPU)

## 🛠️ Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/username/Introduction-to-GAN-model.git
cd Introduction-to-GAN-model
```

2. Créez un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

1. Préparation des données :
   - Placez vos images dans le dossier `data/images`
   - Les images seront redimensionnées à 64x64 pixels

2. Entraînement du modèle :
```bash
python train.py
```

## ⚙️ Paramètres d'entraînement

- Taille du batch : 32
- Dimensions latentes : 100
- Taille des images : 64x64
- Époques : 100
- Taux d'apprentissage : 0.0002

## 📈 Résultats

Les résultats de l'entraînement sont sauvegardés dans :
- `training_checkpoints/` : Points de sauvegarde du modèle
- `results/` : Images générées pendant l'entraînement

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les lignes directrices.

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📚 Références

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [TensorFlow GAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.