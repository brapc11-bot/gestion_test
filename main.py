from fastapi import FastAPI
from database import engine
from models.models import Base
from routes.users import router as user_router
from routes import tests
from routes import incidents

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user_router)
app.include_router(tests.router)
app.include_router(incidents.router)


@app.get("/")
def root():
    return {"message": "API PFE fonctionne"}

