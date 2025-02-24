# 🚀 SafeGuardAI - AI-Powered Network Traffic Monitoring & Attack Prevention

## 📌 Overview

SafeGuardAI is an AI-powered security tool that:

- ✅ **Monitors live network traffic**
- ✅ **Detects and logs malicious activity**
- ✅ **Sends email alerts for detected attacks**
- ✅ **Automatically blocks malicious IPs**

## 🛠 Installation & Setup

### **1️⃣ Install Dependencies**

Run the following command to install required packages:

```bash
pip install pandas scikit-learn torch numpy smtplib email six
```

### **2️⃣ Configure Email Alerts**

Edit `live_detection.py` and replace:

```python
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "recipient_email@gmail.com"
```

> **⚠ Note:** If using Gmail, enable "Less Secure Apps" or generate an [App Password](https://support.google.com/accounts/answer/185833?hl=en).

### **3️⃣ Run the Program**

Start real-time monitoring:

```bash
python live_detection.py
```

---

## 📂 Project Structure

```
SafeGuardAI/
│── preprocess_network_data.py   # Prepares network traffic data for training
│── traffic_nn.py                # Trains the AI model for attack detection
│── live_detection.py            # Monitors traffic, detects attacks, and blocks IPs
│── attack_log.txt               # Logs detected attacks
│── blocked_ips.txt              # Stores blocked IPs
│── network_model.pth            # Saved AI model
│── README.md                    # Project documentation
```

---

## 📝 **File Descriptions**

### **1️⃣ `preprocess_network_data.py`**

- **Purpose:** Prepares network traffic data by encoding IPs and protocols.
- **Key Features:**
  - Converts TCP flags from hex to decimal.
  - Encodes IPs dynamically.
  - Normalizes numerical features.

### **2️⃣ `traffic_nn.py`**

- **Purpose:** Trains a neural network to classify network traffic.
- **Key Features:**
  - Uses PyTorch to train a model.
  - Saves the trained model as `network_model.pth`.

### **3️⃣ `live_detection.py`**

- **Purpose:** Captures network traffic, detects attacks, and takes action.
- **Key Features:**
  - Captures real-time network packets.
  - Logs detected attacks.
  - Sends email alerts.
  - Blocks malicious IPs using firewall rules.

---

## 🚀 **Future Enhancements**

- ✅ Add **automatic unblocking** after a timeout.
- ✅ Integrate with **Telegram/Slack for notifications.**
- ✅ Train a more **advanced AI model** using deep learning.

---

## 📧 **Support & Contact**

For questions or contributions, contact:
📩 Email: d.stoych96@gmail.com

