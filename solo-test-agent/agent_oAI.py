import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 1. Define the Available Support Groups ---
# This is the "knowledge" our agent has about the organization.
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
# In a real-world scenario, this would come from a ticketing system API.
mock_ticket = {
    "ticket_id": "TKT-2025-12345",
    "user_email": "user@example.com",
    "subject": "Cannot connect to the internet",
    "description": "My laptop is showing no internet access and I can't access any websites. I've tried restarting my computer and the Wi-Fi router, but it's still not working. I have rebooted windows, maybe it is an issue with my access management. Please help."
}

# --- 3. The Agentic AI Core ---
class TicketAssignmentAgent:
    def __init__(self):
        # Initialize the language model. gpt-3.5-turbo is a good balance of cost and performance.
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Create a prompt template that instructs the AI on its task.
        # This is where we give the agent its "goal".
        prompt_text = """
        You are an intelligent IT support ticket assignment agent. Your task is to analyze a new support ticket and assign it to the correct support group.

        Here are the available support groups and their responsibilities:
        {group_definitions}

        Analyze the following ticket and determine the most appropriate support group.
        Provide only the name of the support group as your answer.

        Ticket Subject: {subject}
        Ticket Description: {description}

        Correct Support Group:
        """
        self.prompt = PromptTemplate(
            input_variables=["group_definitions", "subject", "description"],
            template=prompt_text
        )

        # The LLMChain is the core of our agent, linking the prompt and the language model.
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def assign_ticket(self, ticket):
        """
        Analyzes a ticket and returns the assigned group.
        """
        print(f"--- Analyzing Ticket: {ticket['ticket_id']} ---")
        print(f"Subject: {ticket['subject']}")
        print(f"Description: {ticket['description']}")
        print("------------------------------------------")

        # Get the definitions of our support groups to provide context to the agent.
        group_definitions = get_support_group_definitions()

        # Run the agentic chain with the ticket information.
        response = self.chain.invoke({
            "group_definitions": group_definitions,
            "subject": ticket["subject"],
            "description": ticket["description"]
        })

        # The agent's decision is in the 'text' part of the response.
        assigned_group = response['text'].strip()

        # Final validation to ensure the agent's choice is valid.
        if assigned_group in SUPPORT_GROUPS:
            return assigned_group
        else:
            return "Unclassified" # Fallback if the agent hallucinates a group name.


# --- 4. Running the POC ---
if __name__ == "__main__":
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
    else:
        # Initialize our agent
        agent = TicketAssignmentAgent()

        # Use the agent to assign our mock ticket
        assigned_group = agent.assign_ticket(mock_ticket)

        # Print the result
        print(f"\n✅ Agent's Decision: Ticket {mock_ticket['ticket_id']} assigned to --> {assigned_group}")