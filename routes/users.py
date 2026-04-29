from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models.models import User
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException

router = APIRouter()

# CREATE USER
@router.post("/users")
def create_user(nom: str, email: str, mot_de_passe: str, role: str = "technicien", db: Session = Depends(get_db)):
    user = User(
        nom=nom,
        email=email,
        mot_de_passe=mot_de_passe,
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# GET ALL USERS
@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@router.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.query(User).filter(User.id == user_id).delete(synchronize_session=False)
    db.commit()

    return {"message": "User deleted successfully"}

