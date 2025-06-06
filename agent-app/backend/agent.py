import os
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from huggingface_hub import login

# Define available support groups
SUPPORT_GROUPS = {
    "Hardware Support": "For issues related to physical devices like laptops, keyboards, and mice.",
    "Software Support": "For problems with applications, operating systems, and software licenses.",
    "Network Support": "For connectivity issues, including Wi-Fi, VPN, and internet access problems.",
    "User Access Management": "For requests related to password resets, account lockouts, and permissions."
}

def get_support_group_definitions():
    """Format support groups for the AI prompt."""
    return "\n".join([f"- {group}: {desc}" for group, desc in SUPPORT_GROUPS.items()])

class TicketAssignmentAgent:
    def __init__(self):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face API token not found. Set HF_TOKEN environment variable.")

        # Log in and initialize model
        login(token=hf_token)
        self.llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            # model="meta-llama/Llama-3.1-8B",
            token=hf_token,
            max_new_tokens=5,
            device=-1
        ))

        # Define AI prompt
        self.prompt = PromptTemplate(
            input_variables=["group_definitions", "subject", "description"],
            template="""
            You are an intelligent IT support ticket assignment agent. Your task is to analyze a new support ticket and assign it to the correct support group.

            Here are the available support groups and their responsibilities:
            {group_definitions}

            Analyze the following ticket and determine the most appropriate support group.
            Provide only the name of the support group as your answer.

            Ticket Subject: {subject}
            Ticket Description: {description}

            Provide only the name of the correct support group as your answer. Do not add any other text.
            """
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def assign_ticket(self, ticket):
        """Processes a support ticket and returns the assigned group."""
        response = self.chain.invoke({
            "group_definitions": get_support_group_definitions(),
            "subject": ticket["subject"],
            "description": ticket["description"]
        })

        raw_response = response['text'].strip()
        print(f"\nRaw response --> {raw_response}\n")
        assigned_group = raw_response.split("\n")[-1].strip()
        return assigned_group if assigned_group in SUPPORT_GROUPS else "Unclassified"