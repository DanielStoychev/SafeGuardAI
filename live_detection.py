import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import subprocess
import smtplib
import platform
from email.mime.text import MIMEText
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ Email Configuration (Use Your Own Email & App Password)
EMAIL_SENDER = "your_email@gmail.com"  # Replace with actual email sender
EMAIL_PASSWORD = "your_app_password"  # Replace with actual email password
EMAIL_RECEIVER = "recipient_email@gmail.com"  # Replace with recipient email

# ✅ Load Training Data to Fit Encoders and Scaler
df = pd.read_csv("network_traffic.csv")

# ✅ Convert TCP Flags from Hex to Integer
df["tcp_flags"] = df["tcp_flags"].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else 0)

# ✅ Fit Encoders
protocol_encoder = LabelEncoder()
protocol_encoder.fit(df["protocol"])

ip_encoder = LabelEncoder()
ip_encoder.fit(df["src_ip"].tolist() + df["dst_ip"].tolist())  # Fit on both source & destination IPs

# ✅ Encode IP Addresses
df["src_ip"] = ip_encoder.transform(df["src_ip"])
df["dst_ip"] = ip_encoder.transform(df["dst_ip"])

# ✅ Convert All Data to Numerical Format
X_train = df.drop(columns=["timestamp", "attack"])  # Exclude labels

# ✅ Fit the Scaler
scaler = StandardScaler()
scaler.fit(X_train)

# ✅ Define Neural Network Model (Same as Training)
class TrafficModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

# ✅ Load Trained Model
X_tensor = torch.load("X_tensor.pth")  # Load to get input size
model = TrafficModel(X_tensor.shape[1])
model.load_state_dict(torch.load("network_model.pth"))
model.eval()

print("✅ Model loaded. Monitoring live traffic...")

# ✅ Function to Block Malicious IPs
def block_ip(ip):
    os_type = platform.system()
    try:
        if os_type == "Windows":
            cmd = f'netsh advfirewall firewall add rule name="Blocked {ip}" dir=in action=block remoteip={ip}'
        else:
            cmd = f'sudo iptables -A INPUT -s {ip} -j DROP'  # For Linux

        subprocess.run(cmd, shell=True, check=True)
        print(f"🚫 Blocked IP: {ip}")
        
        # ✅ Log blocked IPs
        with open("blocked_ips.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} - 🚫 Blocked IP: {ip}\n")

    except Exception as e:
        print(f"❌ Failed to block {ip}: {e}")

# ✅ Function to Send Email Alert
def send_email_alert(src_ip, dst_ip, proto, pred):
    subject = "🚨 Alert: Potential Network Attack Detected!"
    body = f"""
    Time: {datetime.now()}
    Attack Detected!
    Source IP: {src_ip}
    Destination IP: {dst_ip}
    Protocol: {proto}
    Threat Score: {pred:.4f}

    🚫 IP {src_ip} has been blocked automatically!
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"📧 Email alert sent to {EMAIL_RECEIVER}!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ✅ Function to Log Attacks
def log_attack(src_ip, dst_ip, proto, pred):
    with open("attack_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()} - 🚨 Attack Detected! {src_ip} → {dst_ip} (Protocol {proto}) | Score: {pred:.4f}\n")
    
    block_ip(src_ip)  # ✅ Auto-block attacker
    send_email_alert(src_ip, dst_ip, proto, pred)  # ✅ Send email alert

# ✅ Function to Capture and Classify Live Packets
def capture_and_classify():
    cmd = f'tshark -i "Ethernet" -T fields -e ip.src -e ip.dst -e ip.proto -e tcp.flags -c 50'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            src_ip, dst_ip, proto = parts[:3]
            tcp_flags = parts[3] if len(parts) > 3 else "0x0000"

            # ✅ Dynamically encode IPs if unseen
            if src_ip not in ip_encoder.classes_:
                ip_encoder.classes_ = np.append(ip_encoder.classes_, src_ip)
            if dst_ip not in ip_encoder.classes_:
                ip_encoder.classes_ = np.append(ip_encoder.classes_, dst_ip)

            src_encoded = ip_encoder.transform([src_ip])[0]
            dst_encoded = ip_encoder.transform([dst_ip])[0]

            # ✅ Dynamically encode Protocols if unseen
            if proto not in protocol_encoder.classes_:
                protocol_encoder.classes_ = np.append(protocol_encoder.classes_, proto)

            proto_encoded = protocol_encoder.transform([proto])[0]

            # Convert TCP Flags from Hex to Integer
            tcp_flags_int = int(tcp_flags, 16) if tcp_flags.startswith("0x") else 0

            # Normalize Input (Fix StandardScaler Warning)
            X_input = np.array([[src_encoded, dst_encoded, proto_encoded, tcp_flags_int]])
            X_input_df = pd.DataFrame(X_input, columns=X_train.columns)  # Convert NumPy array to DataFrame
            X_input_scaled = scaler.transform(X_input_df)
            X_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)

            # Predict
            with torch.no_grad():
                pred = model(X_tensor).item()

            if pred > 0.5:
                print(f"🚨 Potential attack detected from {src_ip}!")
                log_attack(src_ip, dst_ip, proto, pred)  # ✅ Log, block, and send email
            else:
                print(f"✅ Normal traffic: {src_ip} → {dst_ip}")

    process.wait()

# ✅ Run Live Detection
capture_and_classify()
