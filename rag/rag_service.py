import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy.orm import Session
from models.models import Incident
from database import SessionLocal

chroma_client = chromadb.Client()
embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="incidents",
    embedding_function=embedding_function
)


def load_incidents_to_chroma():
    db: Session = SessionLocal()
    incidents = db.query(Incident).all()

    documents = []
    ids = []

    for inc in incidents:
        text = f"""
        Titre: {inc.titre}
        Description: {inc.description}
        Type: {inc.type_probleme}
        Equipement: {inc.equipement}
        Cause: {inc.cause}
        Solution: {inc.solution}
        """
        documents.append(text)
        ids.append(str(inc.id))

    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(
        documents=documents,
        ids=ids
    )

    db.close()

    return len(incidents)


def search_similar_incidents(query: str, n_results: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    incident_ids = results["ids"][0]
    distances = results["distances"][0]

    db: Session = SessionLocal()

    response = []

    for incident_id, distance in zip(incident_ids, distances):
        incident = db.query(Incident).filter(Incident.id == int(incident_id)).first()

        if incident:
            response.append({
                "id": incident.id,
                "titre": incident.titre,
                "description": incident.description,
                "type_probleme": incident.type_probleme,
                "equipement": incident.equipement,
                "statut": incident.statut,
                "cause_probable": incident.cause,
                "solution_proposee": incident.solution,
                "distance": distance
            })

    db.close()

    return response

def generate_smart_response(query: str, n_results: int = 3):
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
