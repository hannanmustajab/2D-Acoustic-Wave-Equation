import torch
import deepxde as dde
import numpy as np
from deepxde.backend import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib

# Set backend to PyTorch and device to CUDA if available
dde.backend.set_default_backend('pytorch')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input parameters (these will likely come from a command line interface or a config file)

t0, te = 0.0, 3                                         # Time range for the problem
sigmaX, sigmaY = 0.1, 0.1                               # Gaussian wave parameters (spread in X and Y)
XLocation, YLocation = 2,2                              # Location of Gaussian wave center
c = 1                                                   # Speed of wave propagation
hidden_layers = 5                                       # Number of hidden layers in the neural network

# Geometry parameters (spatial domain)
xmin, ymin = 0, 0
xmax, ymax = 4, 4

# Neural network parameters
num_hidden_layer = int(hidden_layers)                   # Number of hidden layers
hidden_layer_size = 64                                  # Number of neurons per hidden layer
num_domain = 100000                                     # Number of colocation points
epochs = 200000                                         # Number of training epochs

# Setup plotting style for scientific plots
plt.style.use(['science', 'no-latex'])
os.makedirs('models', exist_ok=True)

# Set matplotlib backend for non-interactive mode (useful for saving figures)
matplotlib.use('Agg')

# ===================== DEFINE FUNCTIONS =============================== #

# Define a Gaussian wave function as the initial condition
def gaussian_wave(x, y):
    """
    Returns a Gaussian wave centered at (XLocation, YLocation) with spread (sigmaX, sigmaY).
    """
    return torch.exp(-0.5 * (((x - XLocation) / sigmaX) ** 2 + ((y - YLocation) / sigmaY) ** 2))

# ===================== DEFINE THE PDE =============================== #

# Define the PDE (Wave equation in 2D with time)
def pde(X, U):
    """
    Defines the partial differential equation (PDE) for wave propagation.
    PDE: dU_tt = c^2 * (dU_xx + dU_yy)
    """
    # Second-order spatial derivatives
    dU_xx = dde.grad.hessian(U, X, j=0, i=0)  # ∂²U/∂x²
    dU_yy = dde.grad.hessian(U, X, j=1, i=1)  # ∂²U/∂y²
    # Second-order time derivative
    dU_tt = dde.grad.hessian(U, X, j=2, i=2)  # ∂²U/∂t²
    # Wave equation
    eq = dU_tt - c ** 2 * (dU_xx + dU_yy)
    return eq

# ===================== INITIAL CONDITIONS =============================== #

# Define the initial condition (Gaussian pulse)
def initial_condition(X):
    """
    Initial condition at t=0: Gaussian wave.
    """
    x, y = torch.tensor(X[:, 0:1]), torch.tensor(X[:, 1:2])
    return gaussian_wave(x, y)

# ===================== DEFINE GEOMETRY =============================== #

# Define the spatial domain (2D rectangle) and time domain
geom = dde.geometry.Rectangle([xmin, ymin], [xmax, ymax])                   # Spatial domain
timedomain = dde.geometry.TimeDomain(t0, te)                                # Time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain)                     # Combined space-time domain

# ===================== DATA & MODEL SETUP =============================== #

# Define the dataset for PINN (Physics-Informed Neural Network)
data = dde.data.TimePDE(
    geomtime, pde, [], num_domain=num_domain, train_distribution='pseudo'
)

# Neural network architecture
layer_size = [3] + [hidden_layer_size] * num_hidden_layer + [1]             # [input_dim, hidden_layer_sizes..., output_dim]

# Choose activation function based on resampling flag
activation_func = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
lr = 1e-3  # Learning rate

# Create a fully-connected neural network
net = dde.maps.FNN(layer_size, activation_func, initializer)

# Output transform to apply physical constraints or ansatz
def output_transform(x, y):
    """
    Apply an output transformation to enforce specific behavior for the solution.
    Reference: Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations.
               https://link.springer.com/article/10.1007/s10444-023-10065-9
    """
    x_in, y_in, t_in = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    term1 = torch.sigmoid(5 * (2 - (t_in/sigmaX))) * gaussian_wave(x_in, y_in)   # Controls initial wave shape
    term2 = torch.tanh(t_in / sigmaX) ** 2 * y                                   # Enforces smooth behavior in time
    return term1 + term2

# Apply the output transformation to the network
net.apply_output_transform(output_transform)

# Create the model with the dataset and network
model = dde.Model(data, net)

# Setup batch size for training
batch_size_ = 256

# PDEPointResampler is used to resample training points during training (if enabled)
resampler = dde.callbacks.PDEPointResampler(period=1000)

# Compile the model with ADAM optimizer and learning rate scheduling
model.compile(optimizer, lr=lr, decay=("inverse time", 20000, 0.9))                 # Inverse time decay

# Restore from a checkpoint if available (optional)
model.restore("models/model.ckpt-168000.pt", verbose=1)  # Restore from a previous checkpoint

# Setup a checkpoint to save the model during training
checker = dde.callbacks.ModelCheckpoint(
    "models/model.ckpt", save_better_only=True, period=2000
)

# Uncomment if resampling is enabled and for further training
losshistory, trainstate = model.train(iterations=epochs, batch_size=batch_size_, callbacks=[checker,resampler])

# ===================== PLOTTING LOSS =============================== #

# Save loss history and convert it to CSV format for visualization
dde.utils.dat_to_csv("loss.dat", "loss.csv", ["step", "pde_loss", "pde_loss_test"])

# Load the CSV file for plotting
df = pd.read_csv('loss.csv')

def plot_columns_pairs(df, columns):
    """
    Plot pairs of columns from the DataFrame.
    
    Parameters:
    - df: DataFrame containing the loss data.
    - columns: List of column names to plot (in pairs).
    """
    num_columns = len(columns)
    fig = plt.figure(figsize=(12, 2 * num_columns))

    # Plot each pair of columns
    for i in range(0, num_columns, 2):
        col1, col2 = columns[i], columns[i + 1]
        ax = fig.add_subplot(num_columns // 2, 2, i // 2 + 1)
        ax.plot(df['step'], df[col1], label=col1, color='blue', linestyle='-')
        ax.plot(df['step'], df[col2], label=col2, color='orange', linestyle='--')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')  # Log scale for the y-axis
        ax.set_title(f'{col1} vs {col2}')
        ax.legend()

    plt.tight_layout()
    fig.suptitle('Comparison of Various Loss Terms', y=1.02, fontsize=16)
    plt.savefig('scientific_plot.png', bbox_inches='tight', dpi=600)
    plt.show()

# Plot the loss comparison
columns_to_plot = ['pde_loss', 'pde_loss_test']
plot_columns_pairs(df, columns_to_plot)

# ===================== PLOTTING  =============================== #

# Load data

loaded_x = np.linspace(xmin,xmax,400)
loaded_y = np.linspace(xmin,xmax,400)
t = np.linspace(0, te, 1500)

nelx = loaded_x.shape[-1]
nely = loaded_y.shape[-1]
delta_t = t[1] - t[0]
xx, yy = np.meshgrid(loaded_x, loaded_y)

x_ = np.zeros(shape=((nelx) * (nely),))
y_ = np.zeros(shape=((nelx) * (nely),))
for c1, ycor in enumerate(loaded_y):
    for c2, xcor in enumerate(loaded_x):
        x_[c1 * (nelx) + c2] = xcor
        y_[c1 * (nelx) + c2] = ycor

# Create an empty 3D array for U
U = np.zeros((nelx, nely, len(t)))
Ts = []

# Loop over time points
for i, time in enumerate(t):
    t_ = np.ones((nelx) * (nely),) * (time)
    X = np.column_stack((x_, y_))
    X = np.column_stack((X, t_))
    
    # Make predictions using the model (assuming model is defined and loaded)
    T = model.predict(X)
    Ts.append(T)
    T = T.reshape(nelx, nely)
    
    # Assign the predicted values to the corresponding slice in U
    U[:, :, i] = T

# Create a folder named 'plots' if it doesn't exist
plots_folder = 'plots'
os.makedirs(plots_folder, exist_ok=True)
# Visualize a 2D slice of U at specific time points
time_steps_to_visualize = range(0, len(t), 100)

for time_index_to_visualize in time_steps_to_visualize:
    plt.figure(figsize=(15, 5))

    # Plot the original image
    plt.subplot(1, 1, 1)
    im1 = plt.imshow(U[:, :, time_index_to_visualize], origin="lower", cmap="seismic", vmin=-0.1, vmax=0.1, extent=(0, 4, 0, 4))
    plt.title(f'PINN at Time: {time_index_to_visualize * delta_t:.2f} seconds\n'
              f'Domain Points: {num_domain}, Hidden Layers: {hidden_layers}')
    plt.colorbar(im1, fraction=0.046, pad=0.04)

    # Save the plot in the 'plots' folder
    plot_filename = f'{plots_folder}/plot_{time_index_to_visualize:04d}.png'
    plt.savefig(plot_filename)
    plt.close()

print("Plots saved in the 'plots' folder.")
