import os

# Add HuggingFace token here or import from env var
HF_TOKEN = os.getenv("HF_TOKEN")

# Define available support groups and their roles
SUPPORT_GROUPS = {
    "Hardware Support": "For issues related to physical devices like laptops, keyboards, and mice.",
    "Software Support": "For problems with applications, operating systems, and software licenses.",
    "Network Support": "For connectivity issues, including Wi-Fi, VPN, and internet access problems.",
    "User Access Management": "For requests related to password resets, account lockouts, and permissions."
}