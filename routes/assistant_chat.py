from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json
import re

from database import SessionLocal
from models.models import AssistantConversation, AssistantMessage, Incident, Solution
from rag.rag_service import search_similar_incidents
from services.openclaw_service import ask_openclaw


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ChatRequest(BaseModel):
    user_id: str
    message: str


def extract_json_from_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def detect_resolution_intent(user_message: str, history: str):
    msg = user_message.lower().strip()

    obvious_resolved = [
        "resolu",
        "résolu",
        "solved",
        "fixed",
        "c bon",
        "c'est bon",
        "ca marche",
        "ça marche",
        "test ok",
        "le test passe",
        "probleme regle",
        "problème réglé"
    ]

    if msg in obvious_resolved:
        return {
            "is_resolved": True,
            "cause": "Cause à compléter par l'ingénieur.",
            "solution": user_message,
            "summary": "Le technicien indique que le problème est résolu."
        }

    prompt = f"""
Tu es un classificateur pour une application de diagnostic technique industrielle.

Ton rôle est de déterminer si le dernier message de l'ingénieur/technicien indique que le problème est résolu.

Historique de la conversation :
{history}

Dernier message utilisateur :
{user_message}

Réponds uniquement en JSON valide, sans texte autour.

Format exact :
{{
  "is_resolved": true ou false,
  "cause": "cause probable ou confirmée si disponible, sinon chaîne vide",
  "solution": "solution appliquée si disponible, sinon chaîne vide",
  "summary": "résumé court de la résolution si disponible, sinon chaîne vide"
}}

Règles :
- Mets true si le message indique que le problème est résolu, même avec des fautes.
- Mets true si le technicien dit que le test passe, que ça marche, que c'est corrigé, ou que la solution a été validée.
- Mets false si le message est seulement une étape de diagnostic ou une réponse intermédiaire.
- Comprends le langage informel et les fautes d'orthographe.

Exemples résolus :
- ça marche maintenant
- ca march mtn
- test OK
- c bon
- c bon mtn
- probleme reglé
- le test passe
- le test pass
- fixed
- solution validée
- après activation de la terminaison ça marche
- c bon maintenant le test passe après activation de la terminaison
"""

    result = ask_openclaw(
        session_id="resolution_detector",
        prompt=prompt
    )

    if not result.get("success"):
        return {
            "is_resolved": False,
            "cause": "",
            "solution": "",
            "summary": ""
        }

    data = extract_json_from_text(result.get("response", ""))

    if data is None:
        return {
            "is_resolved": False,
            "cause": "",
            "solution": "",
            "summary": ""
        }

    return {
        "is_resolved": bool(data.get("is_resolved", False)),
        "cause": data.get("cause", ""),
        "solution": data.get("solution", ""),
        "summary": data.get("summary", "")
    }


def detect_new_problem_intent(user_message: str):
    msg = user_message.lower().strip()

    obvious_not_problem = [
        "ok",
        "oui",
        "non",
        "merci",
        "thanks",
        "thank you",
        "resolu",
        "résolu",
        "solved",
        "fixed",
        "c bon",
        "c'est bon",
        "ca marche",
        "ça marche",
        "test ok",
        "le test passe",
        "probleme regle",
        "problème réglé"
    ]

    if msg in obvious_not_problem:
        return {
            "is_new_problem": False,
            "reason": "Message de confirmation ou de clôture, pas un nouveau problème."
        }

    prompt = f"""
Tu es un classificateur pour une application de diagnostic technique industrielle.

Ton rôle est de déterminer si le message utilisateur doit créer un NOUVEL incident technique.

Message utilisateur :
{user_message}

Réponds uniquement en JSON valide, sans texte autour.

Format exact :
{{
  "is_new_problem": true ou false,
  "reason": "raison courte"
}}

Règles :
- Mets true si le message décrit un nouveau problème technique, une erreur, une panne, un test échoué, un timeout, une anomalie, un défaut de communication, un problème matériel ou logiciel.
- Mets false si le message est seulement une confirmation, un remerciement, une fermeture, une phrase vague, ou indique que le problème est résolu.
- Mets false si le message est trop court ou ne décrit pas clairement un problème.
- Comprends les fautes d'orthographe et le langage informel.

Exemples true :
- le test moteur donne CAN timeout
- erreur LIN capteur pression
- le banc ne démarre pas
- j'ai un problème avec discord, l'app reste bloquée au démarrage
- aucune trame CAN reçue
- le rapport ne se génère pas
- problème alimentation ECU
- test NOK sur banc A1
- l'ECU ne répond pas

Exemples false :
- ok
- merci
- c bon
- ça marche
- ca marche
- resolu
- résolu
- le test passe maintenant
- fixed
- d'accord
- oui
- non
- continue
- explique
- c bon mtn
"""

    result = ask_openclaw(
        session_id="new_problem_detector",
        prompt=prompt
    )

    if not result.get("success"):
        return {
            "is_new_problem": False,
            "reason": "Impossible de classifier le message."
        }

    data = extract_json_from_text(result.get("response", ""))

    if data is None:
        return {
            "is_new_problem": False,
            "reason": "Réponse de classification invalide."
        }

    return {
        "is_new_problem": bool(data.get("is_new_problem", False)),
        "reason": data.get("reason", "")
    }


@router.post("/assistant/chat")
def assistant_chat(request: ChatRequest, db: Session = Depends(get_db)):
    message_text = request.message.strip()

    if not message_text:
        return {
            "success": False,
            "response": "Message vide."
        }

    conversation = db.query(AssistantConversation).filter(
        AssistantConversation.discord_user_id == request.user_id,
        AssistantConversation.status == "active"
    ).first()

    if conversation is not None:
        old_messages = db.query(AssistantMessage).filter(
            AssistantMessage.conversation_id == conversation.id
        ).order_by(AssistantMessage.created_at.asc()).all()

        history = "\n\n".join([
            f"{msg.role.upper()} : {msg.message}"
            for msg in old_messages
        ])

        resolution = detect_resolution_intent(
            user_message=message_text,
            history=history
        )

        if resolution.get("is_resolved"):
            incident = None

            if conversation.incident_id is not None:
                incident = db.query(Incident).filter(
                    Incident.id == conversation.incident_id
                ).first()

            cause_text = resolution.get("cause", "")
            solution_text = resolution.get("solution", "")
            summary_text = resolution.get("summary", "")

            if not solution_text:
                solution_text = message_text

            if not cause_text:
                cause_text = "Cause déduite à partir de la conversation technique."

            if incident is not None:
                incident.statut = "resolu"
                incident.cause = cause_text
                incident.solution = solution_text

                solution_record = Solution(
                    titre=f"Solution validée - Incident {incident.id}",
                    description=solution_text or summary_text,
                    type_probleme=incident.type_probleme,
                    equipement=incident.equipement,
                    efficacite=1,
                    id_incident=incident.id,
                    id_user=None
                )

                db.add(solution_record)

            conversation.status = "solved"

            user_msg = AssistantMessage(
                conversation_id=conversation.id,
                role="user",
                message=message_text
            )
            db.add(user_msg)

            assistant_response = (
                "Problème marqué comme résolu. "
                "La cause et la solution validée ont été sauvegardées dans la base de connaissances."
            )

            assistant_msg = AssistantMessage(
                conversation_id=conversation.id,
                role="assistant",
                message=assistant_response
            )
            db.add(assistant_msg)

            db.commit()

            return {
                "success": True,
                "resolved": True,
                "conversation_id": conversation.id,
                "incident_id": conversation.incident_id,
                "cause": cause_text,
                "solution": solution_text,
                "response": assistant_response
            }

    if conversation is None:
        msg = message_text.lower().strip()

        not_problem_words = [
            "ok", "oui", "non", "merci", "thanks", "thank you",
            "resolu", "résolu", "solved", "fixed",
            "c bon", "c'est bon", "ca marche", "ça marche",
            "test ok", "le test passe"
        ]

        technical_words = [
            "test", "timeout", "erreur", "error", "nok", "problem", "problème",
            "marche pas", "doesn't work", "not work", "blocked", "stuck",
            "can", "lin", "ecu", "banc", "capteur", "moteur", "communication",
            "trame", "bus", "rapport", "database", "discord"
        ]

        if msg in not_problem_words:
            return {
                "success": False,
                "response": "Aucune conversation active. Envoyez un nouveau problème technique clair pour démarrer un nouvel incident."
            }

        if not any(word in msg for word in technical_words):
            return {
                "success": False,
                "response": "Je n'ai pas détecté un problème technique clair. Décrivez le test, l'erreur ou l'équipement concerné."
            }

    similar_cases = search_similar_incidents(message_text)

    rag_context = "\n\n".join([
        f"Cas similaire {i + 1}: {case}"
        for i, case in enumerate(similar_cases)
    ])

    if conversation is None:
        incident = Incident(
            titre=message_text[:100],
            description=message_text,
            statut="ouvert",
            type_probleme="Assistant AI",
            equipement="Inconnu"
        )

        db.add(incident)
        db.commit()
        db.refresh(incident)

        conversation = AssistantConversation(
            discord_user_id=request.user_id,
            incident_id=incident.id,
            initial_question=message_text,
            rag_context=rag_context,
            status="active"
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    prompt = f"""
Tu es un assistant technique intelligent pour une plateforme de gestion des tests électroniques.

Message du technicien/ingénieur :
{message_text}

Cas similaires trouvés par ChromaDB :
{rag_context}

Objectif :
- Comprendre le problème
- Utiliser les cas similaires
- Donner une réponse structurée si c'est le premier diagnostic
- Puis guider l'utilisateur étape par étape
- Répondre naturellement comme un chat Discord
- À la fin, demander s'il veut plus de clarification ou une assistance étape par étape

Important :
- Réponse maximum 1200 caractères.
- Ne donne pas toutes les étapes à la fois.
- Donne seulement l'étape actuelle et une seule question finale.
- Si c'est une réponse de suivi, ne répète pas toute l'analyse initiale.
- Réponds toujours dans la même langue que le dernier message utilisateur.
- Si l'utilisateur écrit en français, réponds en français.
- Si l'utilisateur écrit en anglais, réponds en anglais.
- Si l'utilisateur écrit en arabe, réponds en arabe.
- Si l'utilisateur mélange plusieurs langues, réponds dans la langue dominante.
- Si l'utilisateur écrit en darija, réponds en darija simple ou en arabe simple compréhensible.
"""

    result = ask_openclaw(
        session_id=f"conv_{conversation.id}",
        prompt=prompt
    )

    response_text = result.get("response", "")

    user_msg = AssistantMessage(
        conversation_id=conversation.id,
        role="user",
        message=message_text
    )
    db.add(user_msg)

    assistant_msg = AssistantMessage(
        conversation_id=conversation.id,
        role="assistant",
        message=response_text
    )
    db.add(assistant_msg)

    db.commit()

    return {
        "user_id": request.user_id,
        "conversation_id": conversation.id,
        "incident_id": conversation.incident_id,
        "message": message_text,
        "response": response_text,
        "success": result.get("success", False),
        "resolved": False
    }


@router.post("/assistant/report/{conversation_id}")
def generate_assistant_report(conversation_id: int, db: Session = Depends(get_db)):
    conversation = db.query(AssistantConversation).filter(
        AssistantConversation.id == conversation_id
    ).first()

    if conversation is None:
        return {
            "success": False,
            "error": "Conversation introuvable"
        }

    messages = db.query(AssistantMessage).filter(
        AssistantMessage.conversation_id == conversation_id
    ).order_by(AssistantMessage.created_at.asc()).all()

    history = "\n\n".join([
        f"{msg.role.upper()} : {msg.message}"
        for msg in messages
    ])

    prompt = f"""
Tu es un assistant technique chargé de générer un rapport d'incident professionnel.

Le rapport doit être très lisible pour un humain.

Règles de formatage :
- Utilise des titres clairs
- Laisse des lignes vides entre sections
- Utilise des bullet points quand nécessaire
- Réponse bien espacée
- Style professionnel industriel
- Pas de texte compact

Problème initial :
{conversation.initial_question}

Contexte RAG :
{conversation.rag_context}

Historique complet :
{history}

Génère ce format EXACT :

# RAPPORT D'INCIDENT

## 1. Titre de l'incident

## 2. Description du problème

## 3. Symptômes observés

## 4. Analyse des causes probables

## 5. Étapes de diagnostic réalisées

## 6. Solution finale ou recommandée

## 7. Statut

## 8. Recommandations futures
"""

    result = ask_openclaw(
        session_id=f"report_{conversation_id}",
        prompt=prompt
    )

    return {
        "success": result.get("success", False),
        "conversation_id": conversation_id,
        "incident_id": conversation.incident_id,
        "report": result.get("response", "")
    }

