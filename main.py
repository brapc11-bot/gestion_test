from fastapi import FastAPI
from database import engine
from routes.users import router as user_router

app = FastAPI()

app.include_router(user_router)

@app.get("/")
def root():
    return {"message": "API PFE fonctionne"}

