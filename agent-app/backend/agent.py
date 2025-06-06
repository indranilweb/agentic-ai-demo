import os
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from huggingface_hub import login
import config

# Define available support groups
SUPPORT_GROUPS = config.SUPPORT_GROUPS

def get_support_group_definitions():
    """Format support groups for the AI prompt."""
    return "\n".join([f"- {group}: {desc}" for group, desc in SUPPORT_GROUPS.items()])

class TicketAssignmentAgent:
    def __init__(self):
        hf_token = config.HF_TOKEN
        if not hf_token:
            raise ValueError("Hugging Face API token not found. Set HF_TOKEN environment variable.")

        # Log in to Hugging Face
        try:
            login(token=hf_token)
            print("Successfully logged in to Hugging Face")
        except Exception as e:
            raise ValueError(f"Failed to authenticate with Hugging Face: {e}")

        # Initialize a free LLM model using Hugging Face
        try:
            self.llm = HuggingFacePipeline(pipeline=pipeline(
                "text-generation",
                # model="meta-llama/Llama-3.1-8B",
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

        # Define AI prompt
        self.prompt = PromptTemplate(
            input_variables=["group_definitions", "subject", "description"],
            template=config.PROMPT_1
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def assign_ticket(self, ticket):
        """Processes a support ticket and returns the assigned group."""
        try:
            response = self.chain.invoke({
                "group_definitions": get_support_group_definitions(),
                "subject": ticket["subject"],
                "description": ticket["description"]
            })

            raw_response = response['text'].strip()
            print(f"\nRaw response --> {raw_response}\n")
            assigned_group = raw_response.split("\n")[-1].strip()
        except Exception as e:
            print(f"Error during model inference: {e}")
            assigned_group = "Unclassified"
            
        return assigned_group if assigned_group in SUPPORT_GROUPS else "Unclassified"