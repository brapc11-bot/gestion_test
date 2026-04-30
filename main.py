from fastapi import FastAPI
from database import engine
from models.models import Base
from routes.users import router as user_router

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user_router)

@app.get("/")
def root():
    return {"message": "API PFE fonctionne"}

