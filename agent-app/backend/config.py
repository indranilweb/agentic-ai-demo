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

# Define prompt templates
PROMPT_1 = """
            You are an intelligent IT support ticket assignment agent. Your task is to analyze a new support ticket and assign it to the correct support group.

            Here are the available support groups and their responsibilities:
            {group_definitions}

            Analyze the following ticket and determine the most appropriate support group.

            Ticket Subject: {subject}
            Ticket Description: {description}

            Provide only the name of the correct support group as your answer. Do not add any other text. Do not add anything after your answer.
            """