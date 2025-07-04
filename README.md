# Hyperparameter tuning using Bayesian optimization

Bayesian optimization for hyperparameter tuning of a ResNet model on Fashion-MNIST.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

The script will:
1. Load Fashion-MNIST dataset
2. Use Bayesian optimization to find the best learning rate
3. Train ResNet with the optimized learning rate
4. Display optimization plots and final test accuracy
