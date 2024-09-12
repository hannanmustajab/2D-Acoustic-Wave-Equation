# 2D-Acoustic-Wave-Equation
# Physics-Informed Neural Networks (PINNs) for Wave Propagation

This project implements a Physics-Informed Neural Network (PINN) to solve the 2D wave equation using DeepXDE and PyTorch. The code models wave propagation in a 2D spatial domain over time and uses a Gaussian pulse as the initial condition. The solution is constrained by the physics of wave propagation encoded in the governing PDE.

## Features
- Solves the 2D wave equation using PINNs
- Supports resampling of training points during training for better accuracy
- Utilizes a custom neural network architecture with configurable depth
- Visualizes the loss history using scientific plots

## Requirements
To run the code, you need the following dependencies:

- Python 3.x
- PyTorch
- DeepXDE
- NumPy
- Pandas
- Matplotlib
- SciencePlots (for better plot styling)

You can install these dependencies using `pip`:

```bash
pip install torch deepxde numpy pandas matplotlib scienceplots
