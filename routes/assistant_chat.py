from fastapi import APIRouter
from pydantic import BaseModel

from rag.rag_service import search_similar_incidents
from services.openclaw_service import ask_openclaw

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@router.post("/assistant/chat")
def assistant_chat(request: ChatRequest):
    similar_cases = search_similar_incidents(request.message)

    rag_context = "\n\n".join([
        f"Cas similaire {i+1}: {case}"
        for i, case in enumerate(similar_cases)
    ])

    prompt = f"""
Tu es un assistant technique intelligent pour une plateforme de gestion des tests électroniques.

Message du technicien/ingénieur :
{request.message}

Cas similaires trouvés par ChromaDB :
{rag_context}

Objectif :
- Comprendre le problème
- Utiliser les cas similaires
- Donner une réponse structurée si c'est le premier diagnostic
- Puis guider l'utilisateur étape par étape
- Répondre naturellement comme un chat Discord
- À la fin, demander s'il veut plus de clarification ou une assistance étape par étape
"""

    result = ask_openclaw(
        session_id=request.user_id,
        prompt=prompt
    )

    return {
        "user_id": request.user_id,
        "message": request.message,
        "rag_context": rag_context,
        "openclaw_result": result
    }
