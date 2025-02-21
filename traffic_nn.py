{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6ad11d64-1edd-47ae-b353-1cf28be51e3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Data loaded: 288 samples ready for training.\n",
      "TrafficModel(\n",
      "  (fc1): Linear(in_features=4, out_features=16, bias=True)\n",
      "  (fc2): Linear(in_features=16, out_features=8, bias=True)\n",
      "  (out): Linear(in_features=8, out_features=1, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "\n",
    "# ✅ Load preprocessed data\n",
    "X_tensor = torch.load(\"X_tensor.pth\")\n",
    "y_tensor = torch.load(\"y_tensor.pth\")\n",
    "\n",
    "# ✅ Create DataLoader\n",
    "dataset = TensorDataset(X_tensor, y_tensor)\n",
    "train_loader = DataLoader(dataset, batch_size=32, shuffle=True)\n",
    "\n",
    "print(f\"✅ Data loaded: {len(dataset)} samples ready for training.\")\n",
    "\n",
    "# ✅ Define Neural Network Model\n",
    "class TrafficModel(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.fc1 = nn.Linear(X_tensor.shape[1], 16)  # Input layer\n",
    "        self.fc2 = nn.Linear(16, 8)  # Hidden layer\n",
    "        self.out = nn.Linear(8, 1)  # Output layer (binary classification)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = torch.relu(self.fc1(x))\n",
    "        x = torch.relu(self.fc2(x))\n",
    "        return torch.sigmoid(self.out(x))  # Sigmoid for binary classification\n",
    "\n",
    "# ✅ Initialize Model\n",
    "torch.manual_seed(42)\n",
    "model = TrafficModel()\n",
    "print(model)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7c6c5240-ed8d-4616-9f94-07013280449f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10, Loss: 0.5283\n",
      "Epoch 2/10, Loss: 0.5114\n",
      "Epoch 3/10, Loss: 0.4920\n",
      "Epoch 4/10, Loss: 0.4697\n",
      "Epoch 5/10, Loss: 0.4456\n",
      "Epoch 6/10, Loss: 0.4185\n",
      "Epoch 7/10, Loss: 0.3881\n",
      "Epoch 8/10, Loss: 0.3534\n",
      "Epoch 9/10, Loss: 0.3182\n",
      "Epoch 10/10, Loss: 0.2842\n",
      "✅ Training complete!\n",
      "✅ Model saved as 'network_model.pth'.\n"
     ]
    }
   ],
   "source": [
    "# ✅ Define Loss Function and Optimizer\n",
    "loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
    "\n",
    "# ✅ Training Loop\n",
    "epochs = 10\n",
    "for epoch in range(epochs):\n",
    "    total_loss = 0\n",
    "    for batch_X, batch_y in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "        y_pred = model(batch_X)\n",
    "        loss = loss_fn(y_pred, batch_y)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item()\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}\")\n",
    "\n",
    "print(\"✅ Training complete!\")\n",
    "\n",
    "# ✅ Save the trained model\n",
    "torch.save(model.state_dict(), \"network_model.pth\")\n",
    "print(\"✅ Model saved as 'network_model.pth'.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3be289b-9741-45cb-9828-6f142ba9529d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
