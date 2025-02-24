{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4a40200d-012a-40ec-8eed-27bf31596a85",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🚀 Capturing network packets...\n",
      "✅ Data saved to network_traffic.csv\n"
     ]
    }
   ],
   "source": [
    "import subprocess\n",
    "import csv\n",
    "\n",
    "# Set up network interface\n",
    "interface = r'\\Device\\NPF_{B5A77C67-106C-4DC1-87AE-EB544BDCD7EA}'\n",
    "output_file = \"network_traffic.csv\"\n",
    "\n",
    "# Tshark command to capture packets\n",
    "cmd = f'tshark -i \"{interface}\" -c 500 -T fields -e frame.time_epoch -e ip.src -e ip.dst -e ip.proto -e tcp.flags'\n",
    "\n",
    "print(\"🚀 Capturing network packets...\")\n",
    "\n",
    "# Run tshark and capture output\n",
    "process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n",
    "\n",
    "# Open CSV file to save packets\n",
    "with open(output_file, \"w\", newline=\"\") as csvfile:\n",
    "    csv_writer = csv.writer(csvfile)\n",
    "    csv_writer.writerow([\"timestamp\", \"src_ip\", \"dst_ip\", \"protocol\", \"tcp_flags\", \"attack\"])  # Add label column\n",
    "\n",
    "    for line in process.stdout:\n",
    "        parts = line.strip().split(\"\\t\")\n",
    "        \n",
    "        if len(parts) >= 4:\n",
    "            timestamp, src_ip, dst_ip, protocol = parts[:4]\n",
    "            tcp_flags = parts[4] if len(parts) > 4 else \"0x0000\"\n",
    "            \n",
    "            # Label SYN packets as potential attacks (1 = attack, 0 = normal)\n",
    "            attack_label = 1 if tcp_flags == \"0x0002\" else 0\n",
    "            \n",
    "            # Save to CSV\n",
    "            csv_writer.writerow([timestamp, src_ip, dst_ip, protocol, tcp_flags, attack_label])\n",
    "\n",
    "process.wait()\n",
    "print(f\"✅ Data saved to {output_file}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e159be3-dcf8-45da-8bc4-bbf0255c3936",
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
