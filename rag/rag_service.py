import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy.orm import Session

from models.models import Incident
from database import SessionLocal


# Persistent ChromaDB storage
# This creates/uses a local folder named chroma_db in your project.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="incidents",
    embedding_function=embedding_function
)


def format_incident_for_chroma(inc: Incident) -> str:
    """
    Convert an incident into a clean text document for semantic search.
    Only useful data should be indexed in ChromaDB.
    """

    return f"""
Titre: {inc.titre or ""}
Description: {inc.description or ""}
Type: {inc.type_probleme or ""}
Equipement: {inc.equipement or ""}
Statut: {inc.statut or ""}
Cause: {inc.cause or ""}
Solution: {inc.solution or ""}
"""


def load_incidents_to_chroma():
    """
    Reload all incidents from MariaDB into ChromaDB.

    Useful when initializing or rebuilding the vector database.
    """

    db: Session = SessionLocal()

    try:
        incidents = db.query(Incident).all()

        documents = []
        ids = []
        metadatas = []

        for inc in incidents:
            documents.append(format_incident_for_chroma(inc))
            ids.append(str(inc.id))
            metadatas.append({
                "incident_id": inc.id,
                "statut": inc.statut or "",
                "type_probleme": inc.type_probleme or "",
                "equipement": inc.equipement or "",
            })

        if not ids:
            return 0

        # Safer than delete + add:
        # If the id exists, it updates it.
        # If the id does not exist, it creates it.
        collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        return len(incidents)

    finally:
        db.close()


def add_solved_incident_to_chroma(incident: Incident):
    """
    Add or update one validated solved incident in ChromaDB.

    This should be called only after the technician confirms
    that the solution works and the incident is saved as resolved in MariaDB.
    """

    if incident is None:
        return False

    if incident.id is None:
        return False

    if incident.statut != "resolu":
        return False

    document = format_incident_for_chroma(incident)

    metadata = {
        "incident_id": incident.id,
        "statut": incident.statut or "",
        "type_probleme": incident.type_probleme or "",
        "equipement": incident.equipement or "",
        "validated": "true"
    }

    collection.upsert(
        documents=[document],
        ids=[str(incident.id)],
        metadatas=[metadata]
    )

    return True


def extract_important_terms(text: str):
    """
    Extract important technical keywords.
    Generic words like test/probleme/erreur are ignored.
    """

    if not text:
        return set()

    text = text.lower()

    keywords = [
        # Communication
        "can", "lin", "timeout", "trame", "communication",

        # Power
        "alimentation", "tension", "12v", "9v", "courant", "power", "voltage",

        # ECU / software
        "ecu", "flash", "firmware", "calibration",

        # Sensors
        "capteur", "sensor", "pression",

        # Pneumatic / leakage
        "étanchéité", "etancheite", "fuite", "pneumatique", "tuyau",
        "raccord", "bar", "maintien",

        # Wiring
        "câblage", "cablage", "connecteur", "fil", "pin", "masse",

        # Bench
        "banc", "interface"
    ]

    return {kw for kw in keywords if kw in text}


def search_similar_incidents(query: str, n_results: int = 3):
    """
    Search similar incidents in ChromaDB and return full data from MariaDB.
    Weak or unrelated matches are filtered out.
    """

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    incident_ids = results["ids"][0]
    distances = results["distances"][0]

    query_terms = extract_important_terms(query)

    db: Session = SessionLocal()

    response = []

    try:
        for incident_id, distance in zip(incident_ids, distances):
            incident = db.query(Incident).filter(
                Incident.id == int(incident_id)
            ).first()

            if not incident:
                continue

            incident_text = f"""
            {incident.titre or ""}
            {incident.description or ""}
            {incident.type_probleme or ""}
            {incident.equipement or ""}
            {incident.cause or ""}
            {incident.solution or ""}
            """

            incident_terms = extract_important_terms(incident_text)

            shared_terms = query_terms.intersection(incident_terms)

            print(
                f"RAG candidate incident {incident.id} | "
                f"distance={distance} | "
                f"query_terms={query_terms} | "
                f"incident_terms={incident_terms} | "
                f"shared={shared_terms}"
            )

            # Filter unrelated cases.
            # If there are no strong shared technical terms, ignore the result.
            if len(shared_terms) < 2:
                continue

            response.append({
                "id": incident.id,
                "titre": incident.titre,
                "description": incident.description,
                "type_probleme": incident.type_probleme,
                "equipement": incident.equipement,
                "statut": incident.statut,
                "cause_probable": incident.cause,
                "solution_proposee": incident.solution,
                "distance": distance,
                "shared_terms": list(shared_terms)
            })

    finally:
        db.close()

    return response


def generate_smart_response(query: str, n_results: int = 3):
    """
    Generate a structured summary from similar incidents.
    """

    incidents = search_similar_incidents(query, n_results)

    causes = []
    solutions = []
    types = []

    for inc in incidents:
        if inc["type_probleme"]:
            types.append(inc["type_probleme"])

        if inc["cause_probable"]:
            causes.append(inc["cause_probable"])

        if inc["solution_proposee"]:
            solutions.append(inc["solution_proposee"])

    return {
        "probleme_recu": query,
        "type_probable": list(set(types)),
        "causes_probables": list(set(causes)),
        "solutions_recommandees": list(set(solutions)),
        "incidents_similaires": incidents
    }
