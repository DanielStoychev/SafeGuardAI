{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9ff4c0e9-6211-4c9d-8c43-9b97b02efdce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Preprocessing complete: 288 samples saved for training.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import torch\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv(\"data/network_traffic.csv\")\n",
    "\n",
    "# Convert protocol column to numeric values\n",
    "protocol_encoder = LabelEncoder()\n",
    "df[\"protocol\"] = protocol_encoder.fit_transform(df[\"protocol\"])  \n",
    "\n",
    "# Encode IP addresses\n",
    "ip_encoder = LabelEncoder()\n",
    "df[\"src_ip\"] = ip_encoder.fit_transform(df[\"src_ip\"])\n",
    "df[\"dst_ip\"] = ip_encoder.fit_transform(df[\"dst_ip\"])\n",
    "\n",
    "# Convert TCP flags from hex to decimal\n",
    "df[\"tcp_flags\"] = df[\"tcp_flags\"].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith(\"0x\") else 0)\n",
    "\n",
    "# Normalize numerical features\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(df.drop(columns=[\"timestamp\", \"attack\"]))  # Drop timestamp\n",
    "y = df[\"attack\"]  # Attack labels (0 = normal, 1 = attack)\n",
    "\n",
    "# ✅ Convert data to PyTorch tensors\n",
    "X_tensor = torch.tensor(X, dtype=torch.float32)\n",
    "y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)\n",
    "\n",
    "# ✅ Save tensors for training\n",
    "torch.save(X_tensor, \"X_tensor.pth\")\n",
    "torch.save(y_tensor, \"y_tensor.pth\")\n",
    "\n",
    "print(f\"✅ Preprocessing complete: {len(df)} samples saved for training.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7937714a-a210-41a6-b82d-4f979bc1f2ff",
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
