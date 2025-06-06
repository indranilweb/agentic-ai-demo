Pre-requisites: Install Python and the associated py modules

pip install langchain transformers accelerate sentence-transformers langchain-community huggingface_hub


Section 1: Solo test agent - https://github.com/indranilweb/agentic-ai-demo/tree/main/solo-test-agent
Can be used for testing the capabilities of the LLM models and fine-tuning the prompts.
All test support groups and prompts are hard-coded in the agent py files.

Option 1:  agent_oAI.py - OpenAI implementation
Maybe the intended way, but requires paid OpenAI token.
If token is available, set in env variable or hardcode in file and execute py file.

set OPENAI_API_KEY={{valid_open_ai_token}}
python agent_oAI.py

Option 2:  agent_HF.py - Free LLM model from HuggingFace implementation
Can be used for POC purpose, might not give the best results. Also need to try multiple options like - Llama, Mistral, Gemma, and others to find the best model for us.

Note: This method will download and run the LLM model (spanning 5 - 20 GB) in local system, and its performance will be completely dependent on local system CPU, GPU, RAM.

Visit https://huggingface.co and create an auth token for your account.
Next visit the LLM page you want to use and accept their terms and conditions to gain access to the gated repository.
Once token is available, set in env variable or hardcode in file and execute py file.

set HF_TOKEN={{valid_hugging_face_token}}
python agent_HF.py


Section 2: Agent app - https://github.com/indranilweb/agentic-ai-demo/tree/main/agent-app
Incorporates a backend in FastAPI and a frontend to allow users to submit test tickets.
The API connects to the agent and fetches the response to be displayed in UI.

Note: This implementation currently uses google/gemma from HuggingFace. So setup mentioned in Option: 2 is required.

Backend: To serve the API install below py modules and execute below command.

pip install fastapi[all] uvicorn

uvicorn api-server:app --host 0.0.0.0 --port 8000 --reload


Frontend: To serve the UI execute below command.

python -m http.server 8080

The UI should be available at http://localhost:8080/ for testing.