import os
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from huggingface_hub import login

# --- 1. Define the Available Support Groups ---
SUPPORT_GROUPS = {
    "Hardware Support": "For issues related to physical devices like laptops, keyboards, and mice.",
    "Software Support": "For problems with applications, operating systems, and software licenses.",
    "Network Support": "For connectivity issues, including Wi-Fi, VPN, and internet access problems.",
    "User Access Management": "For requests related to password resets, account lockouts, and permissions."
}

def get_support_group_definitions():
    """Formats the support group dictionary into a string for the prompt."""
    definitions = ""
    for group, desc in SUPPORT_GROUPS.items():
        definitions += f"- {group}: {desc}\n"
    return definitions

# --- 2. Create a Mock Support Ticket ---
mock_ticket = {
    "ticket_id": "TKT-2025-12345",
    "user_email": "user@example.com",
    "subject": "Cannot connect to the internet",
    "description": "My laptop is showing no internet access and I can't access any websites. I've tried restarting my computer and the Wi-Fi router, but it's still not working. I have rebooted windows, maybe it is an issue with my access management. Please help."
    # "subject": "Unable to install MS Office",
    # "description": "I'm trying to install Microsoft Office, but the installation keeps failing with an error code. I've tried restarting my computer and checked my network connectivity, but it doesn't help."
    # "subject": "Forgot password for company email",
    # "description": "I tried logging into my company email but forgot my password. The reset link isn't working, and I need to login urgently."
}

# --- 3. The Agentic AI Core ---
class TicketAssignmentAgent:
    def __init__(self):
        # Get Hugging Face API token from environment variable
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face API token not found. Set HF_TOKEN environment variable or get one from https://huggingface.co/settings/tokens")

        # Log in to Hugging Face
        try:
            login(token=hf_token)
            print("Successfully logged in to Hugging Face")
        except Exception as e:
            raise ValueError(f"Failed to authenticate with Hugging Face: {e}")

        # Initialize a free LLM model using Hugging Face (Mistral)
        try:
            self.llm = HuggingFacePipeline(pipeline=pipeline(
                "text-generation",
                # model="mistralai/Mistral-7B-v0.3",
                model="google/gemma-2-2b-it",
                # model="facebook/bart-large-mnli",
                token=hf_token,  # Pass token explicitly
                max_new_tokens=5,  # Limit token generation for efficiency and response length
                # temperature=1,  # Reduces randomness
                # repetition_penalty=1.2,  # Reduces repeated parts of the prompt
                device=-1  # -1 to use CPU; change to 0 or higher for GPU if available
            ))
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}. Ensure the model is accessible and your API token has permission.")

        # Create a prompt template
        prompt_text = """
        You are an intelligent IT support ticket assignment agent. Your task is to analyze a new support ticket and assign it to the correct support group.

        Here are the available support groups and their responsibilities:
        {group_definitions}

        Analyze the following ticket and determine the most appropriate support group.
        Provide only the name of the support group as your answer.

        Ticket Subject: {subject}
        Ticket Description: {description}

        Provide only the name of the correct support group as your answer. Do not add any other text.
        """
        # print(f"\nPrompt --> {prompt_text}\n")
        self.prompt = PromptTemplate(
            input_variables=["group_definitions", "subject", "description"],
            template=prompt_text
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def assign_ticket(self, ticket):
        """
        Analyzes a ticket and returns the assigned group.
        """
        print(f"--- Analyzing Ticket: {ticket['ticket_id']} ---")
        print(f"Subject: {ticket['subject']}")
        print(f"Description: {ticket['description']}")
        print("------------------------------------------")

        group_definitions = get_support_group_definitions()

        try:
            response = self.chain.invoke({
                "group_definitions": group_definitions,
                "subject": ticket["subject"],
                "description": ticket["description"]
            })
            # print(f"\nResponse --> {response}\n")
            raw_response = response['text'].strip()
            print(f"\nRaw response --> {raw_response}\n")
            assigned_group = raw_response.split("\n")[-1].strip()
            # assigned_group = response['text'].strip()
            # print(f"\nAssigned_group --> {assigned_group}\n")
        except Exception as e:
            print(f"Error during model inference: {e}")
            assigned_group = "Unclassified"

        return assigned_group if assigned_group in SUPPORT_GROUPS else "Unclassified"

# --- 4. Running the POC ---
if __name__ == "__main__":
    try:
        agent = TicketAssignmentAgent()
        assigned_group = agent.assign_ticket(mock_ticket)
        print(f"\n✅ Agent's Decision: Ticket {mock_ticket['ticket_id']} assigned to --> {assigned_group}")
    except Exception as e:
        print(f"Failed to run agent: {e}")