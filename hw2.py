# %%

# I AM ADDING THIS BRUV DONT WORRY BUT GO AHEAD AND RESTORE
import pandas as pd 
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split # sklearn.model_selection use for splitting data between training and testing
from sklearn.preprocessing import StandardScaler # sklearn.preprocessing for scaling standards 
import os 

# %%
hdf = fetch_california_housing(as_frame=True)
hdf.DESCR

# %%
df = pd.DataFrame(hdf.data, columns = hdf.feature_names)

# %%
df

# %%
df.head()

# %%
df.describe()

# %%
# I have not made a read me before so I am using ChatGPT to help me write one! 

# This portion is updated near the bottom of the notebook to include everything needed for the assignment as is therefore overwritten. 

# I am just leaving it here for grading purposes.

with open("README.md","w") as f:

    f.write("# California Housing Dataset Exploration \n\n")
    f.write("This project performs an initial exploration of the"
            "California Housing dataset provided by Scikit-learn.\n\n")

    # Data description
    f.write('## Dataset Description \n\n')
    f.write("``` text \n")
    f.write(hdf.DESCR)
    f.write("\n```\n")

    f.write('## Frist 5 Rows\n\n')
    f.write("```text\n")
    f.write(df.head().to_string())
    f.write("\n```\n")

    f.write("## Summary Statistics\n\n")
    f.write("```text\n")
    f.write(df.describe().to_string())
    f.write("\n```\n")  

print("READ.md successfully created!")




# %% [markdown]
# ### We need inputs, X , and outputs, Y, for training a machine learning model. The dataset can be split between what we know and what we want to know. 

# %%
# seperate features known as X and targets known as Y 

df["MedHouseVal"] = hdf.target

X = df.drop(columns=["MedHouseVal"])
Y = df["MedHouseVal"]

# %%
# Train-test split
# split between 80% training and 20% testing 
# random_state = 0 for reproducibility 

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size = 0.20,
    random_state = 0
)

print("Training feature shape", x_train.shape)
print("Testing feature shape", x_test.shape)

print("Training target shape", y_train.shape)
print("Testing target shape", y_test.shape)

# %%
# feature scaling 

scaler = StandardScaler()

# fit only on training data and transform them 
# fit_transform will calculate the mean and standarad deviation of every feature using only the training set
# fit_transform will also use the scaled values to standardize the training data

x_train_scaled = scaler.fit_transform(x_train)

# use same scaler to transform the test data
# don't use fit_transform here, data leakage will occur, z = x - mu / sigma where mu & sigma are calculated from training data 
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled[:5])

# %%
# Convert scaled arrays back into DataFrames
x_train_scaled = pd.DataFrame(
    x_train_scaled,
    columns=X.columns,
    index=x_train.index
)

x_test_scaled = pd.DataFrame(
    x_test_scaled,
    columns=X.columns,
    index=x_test.index
)

print("\nScaled training data:")
print(x_train_scaled.head())

print("\nTraining feature means after scaling:")
print(x_train_scaled.mean())

print("\nTraining feature standard deviations after scaling:")
print(x_train_scaled.std())

# %%
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(x_train_scaled,y_train)

print("Linear Sklearn model trained!")

# %%
import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

x_train_tensor = torch.tensor(
    np.asarray(x_train_scaled),
    dtype = torch.float32
)

x_test_tensor = torch.tensor(
    np.asarray(x_test_scaled),
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    np.asarray(y_train).reshape(-1,1),
    dtype = torch.float32
)
y_test_tensor = torch.tensor(
    np.asarray(y_test).reshape(-1,1),
    dtype = torch.float32
)

print("X_train:", x_train_tensor.shape)
print("y_train:", y_train_tensor.shape)

print("X_test:", x_test_tensor.shape)
print("y_test:", y_test_tensor.shape)

# %%
class MLP(nn.Module):

    def __init__(self,input_size, hidden_size = 32):
        super(MLP,self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)

        self.relu = nn.ReLU()

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)

        return x

# %%
input_size = x_train_tensor.shape[1]

model = MLP(
    input_size = input_size,
    hidden_size = 32
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr = 0.01
)

print (model)

# %%
num_epochs = 100

train_losses = []

epoch_list = []

for epoch in range(1, num_epochs + 1):

    model.train()

    # computes y = f(X;theta) where theta represents all the nn weights and biases
    predictions = model(x_train_tensor)

    # calculates the mean squared error (MSEloss) MSE = 1/N SUM(y_i - y_mean_i)^2
    loss = criterion(predictions, y_train_tensor)

    # clears gradients from previous iteration
    optimizer.zero_grad()

    # performs backpropagation via gradient decent on every weight
    loss.backward()

    # updates the wights to reduce loss
    optimizer.step()

    # attaching losses found to view later
    train_losses.append(loss.item())
    epoch_list.append(epoch)

    # comparing epochs and losses, losses should move towards 0 after several epochs
    if epoch % 10 == 0:
        print(
            f"Epoch [{epoch}/{num_epochs}],"
            f"Loss: {loss.item():.4f}"
        )


# %%
import matplotlib.pyplot as plt

plt.plot(epoch_list,train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("# Epochs vs Loss Value")

plt.savefig(
    "loss_curve.png",
    dpi = 300,
    bbox_inches = 'tight'
)

# %%
from sklearn.metrics import mean_squared_error, r2_score

linear_predictions = linear_model.predict(x_test_scaled)

model.eval()

with torch.no_grad():
    nn_predictions_tensor = model(x_test_tensor)

nn_predictions = nn_predictions_tensor.numpy().flatten()

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rms = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

nn_mse = mean_squared_error(
    y_test,
    nn_predictions
)

nn_rmse = np.sqrt(nn_mse)

nn_r2 = r2_score(
    y_test,
    nn_predictions
)

print("LINEAR REGRESSION")
print("-------------------------------")
print(f"MSE: {linear_mse:.4f}")
print(f"RMSE: {linear_rms:.4f}")
print(f"R^2: {linear_r2:.4f}")

print()

print("NEURAL NETWORK")
print("---------------------------------")
print(f"MSE: {nn_mse:.4f}")
print(f"RMSE: {nn_rmse:.4f}")
print(f"R^2: {nn_r2:.4f}")

# %%
# ============================================================
# GENERATE README.md
# ============================================================

# Determine which model performed better
if nn_mse < linear_mse:
    better_model = "Neural Network"
    comparison_text = (
        "The Neural Network performed better than the Linear Regression "
        "model on the test data. It achieved a lower MSE and RMSE, meaning "
        "that its predictions were closer to the true median house values. "
        "It also achieved a higher R² score, indicating that it explained "
        "more of the variation in the target variable."
    )
else:
    better_model = "Linear Regression"
    comparison_text = (
        "The Linear Regression model performed better than the Neural "
        "Network on the test data. It achieved a lower MSE and RMSE, meaning "
        "that its predictions were closer to the true median house values. "
        "It also achieved a higher R² score, indicating that it explained "
        "more of the variation in the target variable."
    )


# Calculate how much the neural network training loss decreased
initial_loss = train_losses[0]
final_loss = train_losses[-1]

loss_reduction = (
    (initial_loss - final_loss) / initial_loss
) * 100


# Description of neural-network training
loss_analysis = (
    f"The neural network training loss decreased from "
    f"{initial_loss:.4f} at the beginning of training to "
    f"{final_loss:.4f} after {num_epochs} epochs. "
    f"This represents a reduction of approximately "
    f"{loss_reduction:.2f}% in the training loss. "
    "The decrease in loss indicates that the neural network learned "
    "patterns relating the input housing features to median house values. "
    "The largest improvements generally occur during the earlier epochs, "
    "while the rate of improvement becomes smaller later in training. "
    "This behavior indicates that the model is moving toward convergence."
)


# ============================================================
# WRITE README
# ============================================================

with open("README.md", "w", encoding="utf-8") as f:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    f.write("# Assignment 2: California Housing Regression\n\n")

    f.write(
        "This project explores the California Housing dataset and "
        "compares a baseline Linear Regression model with a shallow "
        "Neural Network implemented using PyTorch.\n\n"
    )

    f.write("---\n\n")


    # --------------------------------------------------------
    # HOW TO RUN
    # --------------------------------------------------------

    f.write("# Running the Code\n\n")

    f.write(
        "Create and activate the Conda environment using the "
        "required library versions:\n\n"
    )

    f.write("```bash\n")

    f.write(
        "conda create -n env_csci4425 python=3.12.2\n"
    )

    f.write(
        "conda activate env_csci4425\n"
    )

    f.write(
        "conda install numpy=2.4.1\n"
    )

    f.write(
        "conda install matplotlib=3.10.8\n"
    )

    f.write(
        "conda install pandas=2.3.3\n"
    )

    f.write(
        "conda install scikit-learn=1.8.0\n"
    )

    f.write(
        "conda install -c conda-forge scikit-datasets=0.2.5\n"
    )

    f.write(
        "conda install pytorch=2.5.1 "
        "torchvision=0.20.1 "
        "torchaudio=2.5.1\n"
    )

    f.write("```\n\n")

    f.write("Run the program using:\n\n")

    f.write("```bash\n")
    f.write("python hw2.py\n")
    f.write("```\n\n")

    f.write("---\n\n")


    # ========================================================
    # PART 1
    # ========================================================

    f.write("# Part 1: Dataset Exploration\n\n")

    f.write(
        "The California Housing dataset was loaded using "
        "`fetch_california_housing()` from Scikit-learn.\n\n"
    )


    # Dataset description
    f.write("## Dataset Description\n\n")

    f.write("```text\n")
    f.write(hdf.DESCR)
    f.write("\n```\n\n")


    # First five rows
    f.write("## First Five Rows\n\n")

    f.write("```text\n")
    f.write(df.head().to_string())
    f.write("\n```\n\n")


    # Summary statistics
    f.write("## Summary Statistics\n\n")

    f.write("```text\n")
    f.write(df.describe().to_string())
    f.write("\n```\n\n")

    f.write("---\n\n")


    # ========================================================
    # PART 2
    # ========================================================

    f.write("# Part 2: Data Preprocessing\n\n")

    f.write(
        "The dataset was separated into input features (`X`) and "
        "the target variable (`MedHouseVal`). The data was then split "
        "into an 80% training set and a 20% testing set using "
        "`random_state=0` for reproducibility.\n\n"
    )

    f.write(
        "The input features were standardized using Scikit-learn's "
        "`StandardScaler`. The scaler was fitted only on the training "
        "data (`X_train`) to prevent information from the test set from "
        "leaking into the training process. The fitted scaler was then "
        "used to transform both `X_train` and `X_test`.\n\n"
    )

    f.write(
        f"- Training samples: **{len(x_train)}**\n"
    )

    f.write(
        f"- Testing samples: **{len(x_test)}**\n"
    )

    f.write(
        f"- Number of input features: **{x_train.shape[1]}**\n\n"
    )

    f.write("---\n\n")


    # ========================================================
    # PART 3
    # ========================================================

    f.write("# Part 3: Model Building and Training\n\n")


    # Linear Regression
    f.write("## Model 1: Linear Regression\n\n")

    f.write(
        "A Scikit-learn `LinearRegression` model was used as the "
        "baseline regression model. The model was trained using the "
        "scaled training features and the corresponding median house "
        "values.\n\n"
    )


    # Neural Network
    f.write("## Model 2: PyTorch Neural Network\n\n")

    f.write(
        "The second model was a shallow Multi-Layer Perceptron (MLP) "
        "implemented using PyTorch.\n\n"
    )

    f.write("### Network Architecture\n\n")

    f.write(
        "- Input layer: **8 features**\n"
        "- Hidden layer: **32 neurons**\n"
        "- Activation function: **ReLU**\n"
        "- Output layer: **1 neuron**\n"
        "- Loss function: **Mean Squared Error (MSE)**\n"
        "- Optimizer: **Adam**\n"
        "- Learning rate: **0.01**\n"
        f"- Training epochs: **{num_epochs}**\n\n"
    )

    f.write(
        "The architecture can be summarized as:\n\n"
    )

    f.write(
        "**8 Inputs → 32 Hidden Neurons → ReLU → 1 Output**\n\n"
    )

    f.write(
        "During each training epoch, the model performed a forward "
        "pass, calculated the MSE loss, performed backpropagation, "
        "and updated its weights using the Adam optimizer.\n\n"
    )

    f.write("---\n\n")


    # ========================================================
    # PART 4
    # ========================================================

    f.write("# Part 4: Model Evaluation\n\n")

    f.write(
        "Both trained models were evaluated using the scaled testing "
        "dataset. The following regression metrics were calculated:\n\n"
    )

    f.write(
        "- Mean Squared Error (MSE)\n"
        "- Root Mean Squared Error (RMSE)\n"
        "- R-squared (R²)\n\n"
    )

    f.write(
        "Lower MSE and RMSE values indicate better predictive "
        "performance, while a higher R² value indicates that the "
        "model explains a larger fraction of the variation in the "
        "target variable.\n\n"
    )

    f.write("---\n\n")


    # ========================================================
    # PART 5
    # ========================================================

    f.write("# Part 5: Analysis and Interpretation\n\n")

    f.write("## Performance Comparison\n\n")


    # --------------------------------------------------------
    # Markdown Table
    # --------------------------------------------------------

    f.write(
        "| Model | MSE | RMSE | R² |\n"
    )

    f.write(
        "|---|---:|---:|---:|\n"
    )

    f.write(
        f"| Linear Regression | "
        f"{linear_mse:.4f} | "
        f"{linear_rms:.4f} | "
        f"{linear_r2:.4f} |\n"
    )

    f.write(
        f"| Neural Network | "
        f"{nn_mse:.4f} | "
        f"{nn_rmse:.4f} | "
        f"{nn_r2:.4f} |\n\n"
    )


    # Performance explanation
    f.write("### Performance Analysis\n\n")

    f.write(comparison_text)

    f.write("\n\n")

    f.write(
        f"Based on these evaluation metrics, **{better_model}** "
        "provided the better overall performance on the test dataset.\n\n"
    )


    # --------------------------------------------------------
    # LOSS CURVE
    # --------------------------------------------------------

    f.write("## Neural Network Training Loss\n\n")

    f.write(
        "The following figure shows the Mean Squared Error training "
        "loss as a function of epoch.\n\n"
    )

    f.write(
        "![Neural Network Training Loss](loss_curve.png)\n\n"
    )

    f.write("### Training Analysis\n\n")

    f.write(loss_analysis)

    f.write("\n\n")

    f.write("---\n\n")


    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    f.write("# Conclusion\n\n")

    f.write(
        "This project compared a baseline linear regression model "
        "with a shallow PyTorch neural network for predicting median "
        "California house values. Both models were trained using the "
        "same standardized training data and evaluated on the same "
        "held-out testing set. The comparison demonstrates how model "
        "performance can be quantitatively evaluated using MSE, RMSE, "
        "and R².\n"
    )


print("README.md successfully generated!")

# %%



