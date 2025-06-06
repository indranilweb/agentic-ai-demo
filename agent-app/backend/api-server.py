# uvicorn api-server:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from agent import TicketAssignmentAgent  # Import AI agent class

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; update with specific domains for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

agent = TicketAssignmentAgent()  # Initialize AI agent

@app.post("/assign-ticket")
async def assign_ticket(request: Request):
    ticket = await request.json()
    assigned_group = agent.assign_ticket(ticket)
    return {"support_group": assigned_group}