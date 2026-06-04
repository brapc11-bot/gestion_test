from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal
from models.models import AssistantConversation, AssistantMessage, Incident, Solution, Category
from rag.rag_service import search_similar_incidents, add_solved_incident_to_chroma
from services.openclaw_service import ask_openclaw
import json
import re
from datetime import datetime

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


# ---------------------------------------------------------
# LANGUAGE DETECTION
# ---------------------------------------------------------

def detect_language_local(text: str) -> str:
    """
    Simple local language detection.
    Returns: fr, en, ar, darija
    """

    msg = text.lower().strip()

    has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)

    darija_arabic_markers = [
        "دابا", "واش", "مزيان", "خدام", "بزاف", "ديال",
        "شنو", "كيفاش", "علاش", "صافي", "راه", "حيت",
        "باقي", "ما خدمش", "مازال"
    ]

    if has_arabic:
        if any(word in msg for word in darija_arabic_markers):
            return "darija"
        return "ar"

    darija_latin_markers = [
        "safi", "daba", "mzyan", "kay", "dyal", "mlli",
        "wach", "bghit", "kifach", "hiya", "daz", "rah",
        "chno", "3lach", "hit", "kayn", "makaynch",
        "baqi", "ba9i", "mazal", "makhdamch", "khdam"
    ]

    if any(word in msg for word in darija_latin_markers):
        return "darija"

    words = (
        msg.replace(",", " ")
        .replace(".", " ")
        .replace("?", " ")
        .replace("'", " ")
        .replace("’", " ")
        .split()
    )

    french_markers = [
        "le", "la", "les", "un", "une", "des",
        "donne", "pendant", "maintenant", "apres", "après",
        "demarrage", "démarrage", "coupure", "alimentation",
        "tension", "chute", "capteur", "moteur", "banc",
        "erreur", "problème", "probleme", "marche", "pas",
        "j", "ai", "c", "bon", "ça", "ca", "est"
    ]

    english_markers = [
        "the", "motor", "gives", "give",
        "fixed", "passes", "passed", "after", "enabling", "enable",
        "between", "working", "solved", "issue",
        "problem", "error", "now", "still", "not",
        "does", "work", "failed", "failure",
        "during", "startup", "supply"
    ]

    french_score = sum(1 for word in words if word in french_markers)
    english_score = sum(1 for word in words if word in english_markers)

    # Strong French indicators
    if (
        " le " in f" {msg} "
        or " la " in f" {msg} "
        or " une " in f" {msg} "
        or " donne " in f" {msg} "
        or " pendant " in f" {msg} "
        or " alimentation " in f" {msg} "
        or " démarrage" in msg
        or " demarrage" in msg
    ):
        if french_score >= english_score:
            return "fr"

    if english_score >= 2 and english_score > french_score:
        return "en"

    return "fr"



def language_name_for_prompt(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "English"
    if lang == "ar":
        return "Arabic"
    if lang == "darija":
        return "Moroccan Darija"

    return "French"


def empty_message_response(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "Empty message."
    if lang == "ar":
        return "الرسالة فارغة."
    if lang == "darija":
        return "الرسالة خاوية."

    return "Message vide."


def no_active_conversation_response(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "No active conversation. Send a clear new technical problem to start a diagnostic."
    if lang == "ar":
        return "لا توجد محادثة نشطة. أرسل مشكلا تقنيا واضحا لبدء التشخيص."
    if lang == "darija":
        return "ما كايناش محادثة نشطة. صيفط مشكل تقني واضح باش نبداو التشخيص."

    return "Aucune conversation active. Envoyez un nouveau problème technique clair pour démarrer le diagnostic."


def unclear_problem_response(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return "I did not detect a clear technical problem. Describe the test, the error, the equipment, and the symptom."
    if lang == "ar":
        return "لم أكتشف مشكلا تقنيا واضحا. اذكر الاختبار، الخطأ، الجهاز، والأعراض."
    if lang == "darija":
        return "ما بانليش مشكل تقني واضح. شرح ليا التست، الخطأ، الجهاز، وشنو كايوقع."

    return "Je n'ai pas détecté un problème technique clair. Décrivez le test, l'erreur, l'équipement et le symptôme."


def solved_response_by_language(text: str) -> str:
    lang = detect_language_local(text)

    if lang == "en":
        return (
            "Problem marked as resolved. "
            "The validated cause and solution have been saved in the knowledge base."
        )

    if lang == "ar":
        return "تم تحديد المشكل كمحلول، وتم حفظ السبب والحل في قاعدة المعرفة."

    if lang == "darija":
        return "تسجل المشكل كمحلول، وتخزن السبب والحل فقاعدة المعرفة."

    return (
        "Problème marqué comme résolu. "
        "La cause et la solution validée ont été sauvegardées dans la base de connaissances."
    )

def build_clean_solution_text(incident, cause_text: str, solution_text: str, summary_text: str, history: str) -> str:
    """
    Build a clean structured solution for MariaDB and ChromaDB.
    This improves future RAG quality.
    """

    problem = incident.description if incident and incident.description else "Problème technique signalé par le technicien."

    full_context = (history or "").lower()
    problem_lower = (problem or "").lower()
    cause_lower = (cause_text or "").lower()
    solution_lower = (solution_text or "").lower()

    symptoms = []

    is_can_case = (
        "can" in problem_lower
        or "can" in cause_lower
        or "can" in solution_lower
    )

    is_lin_case = (
        "lin" in problem_lower
        or "lin" in cause_lower
        or "lin" in solution_lower
    )

    is_power_case = (
        "alimentation" in problem_lower
        or "power" in problem_lower
        or "tension" in problem_lower
        or "alimentation" in cause_lower
        or "power" in cause_lower
        or "tension" in cause_lower
        or "alimentation" in solution_lower
        or "power" in solution_lower
        or "tension" in solution_lower
    )

    # CAN symptoms
    if is_can_case and "timeout" in full_context:
        symptoms.append("Timeout de communication CAN observé pendant le test.")

    if is_can_case and ("120" in full_context or "120 ohm" in full_context):
        symptoms.append("Résistance CAN-H / CAN-L mesurée à environ 120 ohm.")

    if is_can_case and ("60" in full_context or "60 ohm" in full_context):
        symptoms.append("Après correction, résistance CAN-H / CAN-L mesurée à environ 60 ohm.")

    # LIN symptoms
    if is_lin_case:
        symptoms.append("Erreur de communication LIN observée pendant le test.")

    if is_lin_case and ("aucune trame" in full_context or "aucune trame lin" in full_context):
        symptoms.append("Aucune trame LIN reçue malgré l’alimentation du capteur.")

    if is_lin_case and ("12v" in full_context or "12 v" in full_context):
        symptoms.append("Capteur alimenté en 12V.")

    if is_lin_case and (
        "mal branche" in full_context
        or "mal branché" in full_context
        or "connecteur" in full_context
        or "fil lin" in full_context
        or "cablage lin" in full_context
        or "câblage lin" in full_context
    ):
        symptoms.append("Mauvais branchement détecté au niveau du connecteur ou du fil de communication.")

    # Power symptoms
    if is_power_case and (
        "chute de tension" in full_context
        or "tension chute" in full_context
        or "tension instable" in full_context
        or "alimentation instable" in full_context
        or "chute de 12v" in full_context
        or "chute de 12 v" in full_context
        or "12v à 9v" in full_context
        or "12v a 9v" in full_context
        or "12 v à 9 v" in full_context
        or "12 v a 9 v" in full_context
    ):
        symptoms.append("Tension d’alimentation instable observée pendant le test.")

    if not symptoms:
        symptoms.append("Symptômes décrits dans la conversation de diagnostic.")

    symptoms_text = "\n".join([f"- {s}" for s in symptoms])

    return f"""
Problème:
{problem}

Symptômes observés:
{symptoms_text}

Cause validée:
{cause_text or "Cause déduite à partir de la conversation technique."}

Solution appliquée:
{solution_text or "Solution validée par le technicien."}

Résultat:
{summary_text or "Le technicien a confirmé que le problème est résolu."}
""".strip()


def improve_cause_solution_from_history(cause_text: str, solution_text: str, summary_text: str, history: str, user_message: str):
    """
    Improve generic cause/solution using the conversation history.
    This keeps resolution local without calling OpenClaw.
    """

    full_context = (history + "\n" + user_message).lower()

    # LIN wiring case
    if (
        "lin" in full_context
        and (
            "mal branche" in full_context
            or "mal branché" in full_context
            or "mauvais branchement" in full_context
            or "cablage" in full_context
            or "câblage" in full_context
            or "connecteur" in full_context
            or "fil lin" in full_context
        )
    ):
        return {
            "cause": "Erreur de communication LIN causée par un mauvais branchement du fil LIN ou du connecteur.",
            "solution": "Correction du câblage LIN et remise du fil LIN sur le bon connecteur/pin.",
            "summary": "Le test capteur pression passe après correction du câblage LIN."
        }

    # Power supply instability case
    if (
        ("alimentation" in full_context or "power" in full_context)
        and (
            "chute" in full_context
            or "instable" in full_context
            or "9v" in full_context
            or "12v" in full_context
            or "tension" in full_context
        )
    ):
        return {
            "cause": "Alimentation instable provoquant une chute de tension pendant le test.",
            "solution": "Remplacement ou correction de l’alimentation afin de stabiliser la tension à 12V.",
            "summary": "Le test passe après stabilisation ou remplacement de l’alimentation."
        }

    # CAN termination case
    if (
        "can" in full_context
        and ("120" in full_context or "120 ohm" in full_context)
        and ("60" in full_context or "60 ohm" in full_context)
        and ("terminaison" in full_context or "termination" in full_context)
    ):
        return {
            "cause": (
                "Le timeout CAN était causé par une terminaison CAN manquante ou incorrecte. "
                "La résistance mesurée entre CAN-H et CAN-L était de 120 ohm au lieu d'environ 60 ohm."
            ),
            "solution": (
                "Activation de la terminaison CAN côté banc/interface afin d'obtenir environ 60 ohm "
                "entre CAN-H et CAN-L."
            ),
            "summary": "Le test moteur passe après activation de la terminaison CAN."
        }

    return {
        "cause": cause_text,
        "solution": solution_text,
        "summary": summary_text
    }


# ---------------------------------------------------------
# LOCAL RESOLUTION DETECTION
# No OpenClaw call here.
# ---------------------------------------------------------

def detect_resolution_intent(user_message: str, history: str):
    msg = user_message.lower().strip()
    full_context = (history + "\n" + user_message).lower()

    negative_phrases = [
        # English
        "not solved", "not fixed", "not working",
        "still timeout", "still error", "still problem",
        "does not work", "doesn't work", "not work",
        "same problem", "same error",

        # French
        "pas résolu", "pas resolu", "pas réglé", "pas regle",
        "ne marche pas", "ça ne marche pas", "ca ne marche pas",
        "encore timeout", "toujours timeout",
        "toujours erreur", "toujours problème", "toujours probleme",
        "ça marche pas", "ca marche pas",

        # Arabic / Darija Arabic
        "مازال", "ما زال", "ما خدامش", "ما خدمش",
        "ما تصلحش", "باقي", "باقي نفس المشكل",

        # Darija Latin
        "makhdamch", "ma khdamch", "mazal", "baqi", "ba9i",
        "baqi timeout", "ba9i timeout"
    ]

    new_problem_phrases = [
        # English
        "but now", "another problem", "another issue", "new problem",

        # French
        "mais maintenant", "autre problème", "autre probleme",
        "nouveau problème", "nouveau probleme",

        # Arabic / Darija
        "دابا عندي", "ولكن دابا", "مشكل آخر"
    ]

    resolution_triggers = [
        # French
        "resolu", "résolu", "c bon", "c'est bon",
        "ca marche", "ça marche", "test ok", "test passe",
        "test pass", "le test passe", "le test pass",
        "probleme regle", "problème réglé",
        "probleme corrige", "problème corrigé",
        "plus de timeout", "plus d'erreur", "plus erreur",
        "tout est ok", "tout est normal",
        "ca fonctionne", "ça fonctionne",

        # English
        "solved", "fixed", "problem solved", "issue solved",
        "it works", "working now", "no timeout", "timeout gone",
        "test passed", "the test passes", "the test passed",

        # Arabic / Darija
        "المشكل تحل", "تحل المشكل", "خدام دابا", "دابا خدام",

        # Darija Latin
        "safi", "daz mzyan", "khdam daba", "mzyan daba"
    ]

    if any(phrase in msg for phrase in negative_phrases):
        return {
            "is_resolved": False,
            "cause": "",
            "solution": "",
            "summary": ""
        }

    if any(phrase in msg for phrase in new_problem_phrases):
        return {
            "is_resolved": False,
            "cause": "",
            "solution": "",
            "summary": ""
        }

    if any(trigger in msg for trigger in resolution_triggers):

        # Specific CAN termination case
        if (
            "can" in full_context
            and ("120" in full_context or "120 ohm" in full_context)
            and ("60" in full_context or "60 ohm" in full_context)
            and ("terminaison" in full_context or "termination" in full_context)
        ):
            return {
                "is_resolved": True,
                "cause": (
                    "Le timeout CAN était causé par une terminaison CAN manquante ou incorrecte. "
                    "La résistance mesurée entre CAN-H et CAN-L était de 120 ohm au lieu d'environ 60 ohm, "
                    "ce qui indique qu'une seule terminaison était présente sur le bus."
                ),
                "solution": (
                    "Activation de la terminaison CAN côté banc/interface afin d'obtenir environ 60 ohm "
                    "entre CAN-H et CAN-L. Après correction, le test moteur est passé correctement."
                ),
                "summary": (
                    "Le problème de timeout CAN a été résolu après activation de la terminaison CAN."
                )
            }

        return {
            "is_resolved": True,
            "cause": "Cause déduite à partir de la conversation technique.",
            "solution": user_message,
            "summary": "Le technicien indique que le problème est résolu."
        }

    return {
        "is_resolved": False,
        "cause": "",
        "solution": "",
        "summary": ""
    }

def suggest_category_name(text: str) -> tuple[str, str]:
    """
    Suggest a general category name from the incident text.
    The goal is to create reusable categories, not one category per incident.
    """

    msg = text.lower()

    if "can" in msg:
        return (
            "Communication CAN",
            "Incidents liés au bus CAN, timeout CAN, trames CAN, câblage CAN, terminaison CAN ou configuration CAN."
        )

    if "lin" in msg:
        return (
            "Communication LIN",
            "Incidents liés au bus LIN, trames LIN, câblage LIN, capteurs LIN ou erreurs de communication LIN."
        )

    if (
        "alimentation" in msg
        or "tension" in msg
        or "12v" in msg
        or "9v" in msg
        or "power" in msg
        or "voltage" in msg
    ):
        return (
            "Alimentation",
            "Incidents liés à l'alimentation électrique, tension instable, chute de tension, courant ou bloc d'alimentation."
        )

    if "capteur" in msg or "sensor" in msg:
        return (
            "Capteurs",
            "Incidents liés aux capteurs, mesures, signaux capteurs ou comportement anormal d'un capteur."
        )

    if "ecu" in msg or "flash" in msg or "calibration" in msg or "firmware" in msg:
        return (
            "ECU / Flash / Calibration",
            "Incidents liés à l'ECU, au flash, au firmware, aux fichiers de calibration ou à la configuration calculateur."
        )

    if "rapport" in msg or "report" in msg or "pdf" in msg:
        return (
            "Rapports",
            "Incidents liés à la génération, l'export ou la consultation des rapports."
        )

    if "database" in msg or "mariadb" in msg or "mysql" in msg or "base de données" in msg:
        return (
            "Base de données",
            "Incidents liés à la base de données, requêtes SQL, connexion ou stockage."
        )

    if "banc" in msg or "bench" in msg:
        return (
            "Banc de test",
            "Incidents liés au banc de test, configuration banc, interface ou équipement de test."
        )

    return (
        "Autre",
        "Incidents techniques ne correspondant pas encore à une catégorie spécialisée."
    )


def get_or_create_category(db: Session, text: str):
    """
    Find an existing category or create a new one if needed.
    """

    category_name, description = suggest_category_name(text)

    category = db.query(Category).filter(
        Category.nom == category_name
    ).first()

    if category is None:
        category = Category(
            nom=category_name,
            description=description
        )
        db.add(category)
        db.commit()
        db.refresh(category)

    return category

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


def generate_validated_knowledge_with_openclaw(
    initial_problem: str,
    user_history: str,
    current_message: str
):
    """
    Ask OpenClaw to generate clean structured knowledge after the technician confirms the solution works.
    This is called only after local resolution detection.
    """

    prompt = f"""
You are a technical knowledge extraction assistant for an industrial test incident management system.

The technician has confirmed that the problem is solved.

Your task:
Analyze ONLY the technician's messages and extract the validated knowledge that should be saved in MariaDB and ChromaDB.

Do not invent facts.
Do not use assistant suggestions unless the technician confirmed them.
Use the technician's observations, measurements, actions, and final confirmation.

Initial problem:
{initial_problem}

Technician conversation history:
{user_history}

Final technician message:
{current_message}

Return ONLY valid JSON, no markdown, no explanation.

JSON format:
{{
  "category_name": "short general category name",
  "category_description": "short category description",
  "cause": "validated technical cause",
  "solution": "validated solution applied",
  "symptoms": [
    "observed symptom 1",
    "observed symptom 2"
  ],
  "result": "final confirmed result",
  "structured_solution": "clean human-readable structured solution",
  "chroma_document": "clean searchable document for future RAG"
}}

Rules:
- category_name must be general, not too specific.
  Good examples:
  "Communication CAN", "Communication LIN", "Alimentation", "Pneumatique / Étanchéité", "Température / Refroidissement", "Câblage / Connectique", "Capteurs", "Banc de test", "Logiciel".
- If the problem is about temperature, cooling, fan, overheating, use "Température / Refroidissement".
- If the problem is about pressure leak, sealing, pneumatic tube, use "Pneumatique / Étanchéité".
- If the problem is about voltage drop, 12V, 9V, power supply, use "Alimentation".
- If the problem is about CAN, CAN-H, CAN-L, termination, use "Communication CAN".
- If the problem is about LIN, LIN frame, LIN wire, use "Communication LIN".
- structured_solution must include:
  Problème:
  Symptômes observés:
  Cause validée:
  Solution appliquée:
  Résultat:
- chroma_document must be concise but complete for future similarity search.
"""

    result = ask_openclaw(
        session_id="knowledge_extractor",
        prompt=prompt
    )

    if not result.get("success"):
        return None

    data = extract_json_from_text(result.get("response", ""))

    if not data:
        return None

    return data

def get_or_create_category_by_name(db: Session, category_name: str, category_description: str = ""):
    """
    Get an existing category by name or create it if it does not exist.
    Used after OpenClaw extracts the validated category.
    """

    if not category_name:
        category_name = "Autre"

    category_name = category_name.strip()

    category = db.query(Category).filter(
        Category.nom == category_name
    ).first()

    if category is None:
        category = Category(
            nom=category_name,
            description=category_description or "Catégorie créée automatiquement par l'assistant."
        )
        db.add(category)
        db.commit()
        db.refresh(category)

    return category


# ---------------------------------------------------------
# MAIN CHAT ROUTE
# ---------------------------------------------------------

@router.post("/assistant/chat")
def assistant_chat(request: ChatRequest, db: Session = Depends(get_db)):
    message_text = request.message.strip()

    if not message_text:
        return {
            "success": False,
            "response": empty_message_response(request.message)
        }

    # ---------------------------------------------------------
    # GLOBAL NORMALIZATION
    # ---------------------------------------------------------

    msg_lower = message_text.lower().strip()

    # ---------------------------------------------------------
    # GLOBAL COMMAND: CANCEL
    # Must be checked before CASE 1 and CASE 2
    # ---------------------------------------------------------

    cancel_words = [
        "cancel", "stop", "abort",
        "annuler", "annule",
        "إلغاء", "الغاء",
        "safi", "stoppe", "stopper"
    ]

    if any(word in msg_lower for word in cancel_words):
        return {
            "success": True,
            "cancelled": True,
            "response": "Request cancelled. No data was saved."
        }

    # Find active conversation for this Discord user
    conversation = (
        db.query(AssistantConversation)
        .filter(
            AssistantConversation.discord_user_id == request.user_id,
            AssistantConversation.status == "active"
        )
        .order_by(AssistantConversation.updated_at.desc())
        .first()
    )

    # -----------------------------------------------------
    # CASE 1: Existing active conversation
    # Check if user says the problem is solved.
    # -----------------------------------------------------

    if conversation is not None:
        old_messages = db.query(AssistantMessage).filter(
            AssistantMessage.conversation_id == conversation.id
        ).order_by(AssistantMessage.created_at.asc()).all()
        
        if not old_messages:
            old_messages = []

        history = "\n\n".join([
            f"{msg.role.upper()} : {msg.message}"
            for msg in old_messages
            if msg.message
        ])

        user_history = "\n\n".join([
            msg.message
            for msg in old_messages
            if msg.role == "user" and msg.message
        ])        

        msg_lower = message_text.lower().strip()

        resolution_triggers = [
            # French
            "resolu",
            "résolu",
            "c bon",
            "c'est bon",
            "ca marche",
            "ça marche",
            "test ok",
            "test passe",
            "test pass",
            "le test passe",
            "le test pass",
            "probleme regle",
            "problème réglé",
            "plus de timeout",
            "plus d'erreur",
            "plus erreur",

            # English
            "solved",
            "fixed",
            "problem solved",
            "issue solved",
            "it works",
            "working now",
            "no timeout",
            "timeout gone",
            "test passed",
            "the test passes",
            "the test passed",

            # Arabic / Darija
            "المشكل تحل",
            "تحل المشكل",
            "خدام دابا",
            "دابا خدام",

            # Darija Latin
            "safi",
            "daz mzyan",
            "khdam daba",
            "mzyan daba"
        ]

        negative_phrases = [
            # English
            "not solved", "not fixed", "not working", "still timeout",
            "still error", "still problem", "does not work", "doesn't work",

            # French
            "pas résolu", "pas resolu", "ne marche pas",
            "encore timeout", "toujours timeout",
            "ça marche pas", "ca marche pas",

            # Arabic / Darija
            "مازال", "ما خدامش", "ما خدمش", "باقي",

            # Darija Latin
            "makhdamch", "ma khdamch", "mazal", "baqi", "ba9i"
        ]

        new_problem_phrases = [
            # English
            "but now", "another problem", "another issue", "new problem",

            # French
            "mais maintenant", "autre problème", "autre probleme",
            "nouveau problème", "nouveau probleme",

            # Arabic / Darija
            "دابا عندي", "ولكن دابا", "مشكل آخر"
        ]

        resolution = {
            "is_resolved": False,
            "cause": "",
            "solution": "",
            "summary": ""
        }

        if (
            any(trigger in msg_lower for trigger in resolution_triggers)
            and not any(phrase in msg_lower for phrase in negative_phrases)
            and not any(phrase in msg_lower for phrase in new_problem_phrases)
        ):
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
                        
            knowledge = generate_validated_knowledge_with_openclaw(
                initial_problem=incident.description if incident else conversation.initial_question,
                user_history=user_history,
                current_message=message_text
            )

            category_from_ai = None

            if knowledge:
                cause_text = knowledge.get("cause") or cause_text
                solution_text = knowledge.get("solution") or solution_text
                summary_text = knowledge.get("result") or summary_text
                clean_solution_text = knowledge.get("structured_solution") or solution_text

                category_name = knowledge.get("category_name") or ""
                category_description = knowledge.get("category_description") or ""

                if category_name:
                    category_from_ai = get_or_create_category_by_name(
                        db=db,
                        category_name=category_name,
                        category_description=category_description
                    )
            else:
                clean_solution_text = build_clean_solution_text(
                    incident=incident,
                    cause_text=cause_text,
                    solution_text=solution_text,
                    summary_text=summary_text,
                    history=user_history
                )


            if incident is not None:
                incident.statut = "resolu"
                incident.cause = cause_text
                incident.solution = clean_solution_text

                if category_from_ai is not None:
                    incident.category_id = category_from_ai.id

                solution_record = Solution(
                    titre=f"Solution validée - Incident {incident.id}",
                    description=clean_solution_text,
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

            assistant_response = solved_response_by_language(message_text)

            assistant_msg = AssistantMessage(
                conversation_id=conversation.id,
                role="assistant",
                message=assistant_response
            )
            db.add(assistant_msg)

            conversation.updated_at = datetime.utcnow() 

            db.commit()

            if incident is not None:
                db.refresh(incident)

                try:
                    add_solved_incident_to_chroma(incident)
                    print(f"Incident {incident.id} ajouté/mis à jour dans ChromaDB.")
                except Exception as e:
                    print("Erreur ajout incident résolu dans ChromaDB:", e)

            return {
                "success": True,
                "resolved": True,
                "conversation_id": conversation.id,
                "incident_id": conversation.incident_id,
                "cause": cause_text,
                "solution": clean_solution_text,
                "response": assistant_response
            }

    # -----------------------------------------------------
    # CASE 2: No active conversation
    # Check if message is a clear technical problem.
    # -----------------------------------------------------

    if conversation is None:

        # ==============================
        # LEVEL 1: Normalize input
        # ==============================

        msg = message_text.lower().strip()

        def is_new_problem(msg: str) -> bool:
            keywords = [
                "new problem", "another problem", "another issue",
                "autre problème", "autre souci", "nouveau problème",
                "nouveau souci", "مشكل آخر", "دابا عندي"
            ]
            return any(k in msg for k in keywords)

        # ==============================
        # LEVEL 2: Non-problem messages
        # ==============================

        not_problem_words = [
            "ok", "oui", "non", "merci", "thanks", "thank you",
            "resolu", "résolu", "solved", "fixed",
            "c bon", "c'est bon", "ca marche", "ça marche",
            "test ok", "le test passe",
            "safi", "daba safi"
        ]

        if msg in not_problem_words:
            return {
                "success": False,
                "response": no_active_conversation_response(message_text)
            }

        # ==============================
        # LEVEL 3: Technical filtering
        # ==============================

        technical_words = [
            # General
            "test", "timeout", "erreur", "error", "nok",
            "problem", "problème", "probleme",

            # Behavior
            "marche pas", "doesn't work", "does not work", "not work",
            "blocked", "stuck",

            # Technical domain
            "can", "lin", "ecu", "banc", "capteur", "moteur",
            "motor", "communication", "trame", "bus", "rapport",
            "database", "discord", "résistance", "resistance", "ohm",
            "alimentation", "power", "sensor", "relay", "relais",
            "firmware", "calibration", "flash", "voltage", "current",

            # Arabic
            "اختبار", "خطأ", "مشكل", "محرك", "كان", "تواصل",

            # Darija
            "kay", "dyal"
        ]

        if not any(word in msg for word in technical_words):
            return {
                "success": False,
                "response": unclear_problem_response(message_text)
            }

        # ==============================
        # LEVEL 4: New problem detection
        # ==============================

        if is_new_problem(msg):
            return {
                "success": False,
                "response": unclear_problem_response(message_text)
            }


    # -----------------------------------------------------
    # RAG search
    # Search local knowledge only once when a new problem starts.
    # If conversation already exists, reuse the saved rag_context.
    # -----------------------------------------------------

    if conversation is None:
        print("RAG search: new problem, searching ChromaDB...")

        
        similar_cases = search_similar_incidents(message_text)

        rag_context = "\n\n".join([
            f"Similar case {i + 1}: {case}"
            for i, case in enumerate(similar_cases)
        ])
    else:
        print("RAG search skipped: active conversation, reusing saved context.")

        rag_context = conversation.rag_context or ""


    # -----------------------------------------------------
    # Create incident + conversation if new problem
    # -----------------------------------------------------

    if conversation is None:
        category = get_or_create_category(db, message_text)

        incident = Incident(
            titre=message_text[:100],
            description=message_text,
            statut="ouvert",
            type_probleme="Assistant AI",
            equipement="Inconnu",
            category_id=category.id
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

    # -----------------------------------------------------
    # Build OpenClaw prompt
    # -----------------------------------------------------

    is_follow_up = False

    if conversation is not None:
        old_messages_count = db.query(AssistantMessage).filter(
            AssistantMessage.conversation_id == conversation.id
        ).count()
        is_follow_up = old_messages_count > 0

    prompt = f"""
You are an intelligent technical diagnostic assistant for an electronic test management platform.

CRITICAL LANGUAGE RULE:
You must determine the language directly from the user's latest message.
Answer entirely in the same language as the user's latest message.

Very important:
- If the latest user message is in French, answer fully in French.
- If the latest user message is in English, answer fully in English.
- If the latest user message is in Arabic, answer fully in Arabic.
- If the latest user message is in Moroccan Darija, answer in simple Moroccan Darija.
- If internal similar cases are written in another language, translate and reformulate them into the user's language.
- Do not copy the language of the internal case.
- The opening sentence, section titles, step labels, and final question must also be in the user's language.
- Do not write "This matches", "Step", or "Question" if the user wrote in French. Use "J’ai trouvé...", "Étape actuelle :", and "Question :".

Technician/engineer latest message:
{message_text}

Similar internal cases found by ChromaDB:
{rag_context}

Conversation state:
Is follow-up message: {is_follow_up}

Internal case relevance rule:
- The internal cases provided by ChromaDB are only candidate matches, not guaranteed correct matches.
- Before using an internal case, compare it with the user's latest problem.
- Use an internal case only if it belongs to the same technical family and explains the same kind of symptom.
- Do not force a weak match.
- If the candidate case is about alimentation/tension but the user problem is about pressure leak/étanchéité, ignore it.
- If the candidate case is about CAN but the user problem is about LIN, ignore it.
- If the candidate case is about LIN but the user problem is about alimentation, ignore it.
- If no candidate is truly relevant, say in the user's language that no validated internal solution was found, then propose one diagnostic path based on technical reasoning.

Conversation behavior:
- If this is a follow-up message, do not repeat whether an internal similar case was found.
- If this is a follow-up message, do not say again "J’ai trouvé un cas interne similaire", "I found a similar case", or "This matches".
- Mention the internal case only in the first answer of a new incident.
- Continue directly from the technician's latest result.
- Do not restart the analysis.
- Do not repeat the same introduction.
- Only give the next useful step and one question.

Core mission:
You are not a simple search bot.
You are an interactive diagnostic assistant for technicians and engineers.

Internal knowledge behavior:
- Always analyze the internal similar cases first.
- If a similar internal case contains a useful solution, treat it as company knowledge.
- Do not copy the internal solution directly.
- Understand it, translate it if needed, reformulate it clearly, and turn it into practical guidance.
- Start with only the first useful diagnostic step.
- Ask the technician for the result of that step.
- Continue step by step until the technician confirms that the solution works.

Mandatory wording rule:
- If the user writes in French, use:
  "J’ai trouvé un cas interne similaire déjà résolu."
  "Étape actuelle :"
  "Question :"
- If the user writes in English, use English labels.
- If the user writes in Arabic, use Arabic labels.
- If the user writes in Moroccan Darija, use simple Darija labels.

If no useful internal solution is found:
- Say that no validated internal solution was found.
- Propose one reasonable diagnostic path based on technical knowledge.
- Do not present it as validated company knowledge.
- Ask the technician to test one step and report the result.

If the technician does not understand:
- Explain more simply.
- Break the action into smaller steps.
- Avoid long theory.
- Give one clear action to do now.

If the technician says the proposed solution did not work:
- Do not repeat the same solution.
- Move to another possible cause or another diagnostic path.
- Ask one clear question to continue.

If the technician confirms the solution works:
- The backend will close the incident locally.
- Do not generate a long final message.

Tool usage rule:
- Do not use external tools unless absolutely necessary.
- First use the provided internal context and the conversation history.
- If the internal context is sufficient, answer directly.
- Do not search the web for every follow-up message.
- Only search externally if no internal solution exists or if the previous proposed solution failed.

Important:
- Maximum response length: 1200 characters.
- Do not give all diagnostic steps at once.
- Give only the current diagnostic step and one final question.
- If this is a follow-up answer, do not repeat the whole initial analysis.
- Keep the answer practical and technical.
- Use simple language suitable for a technician working in the field.
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


# ---------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------

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
