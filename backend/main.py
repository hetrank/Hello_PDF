from fastapi import FastAPI
from database import Base, engine
from auth import router as auth_router
from chat import router as chat_router

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "Backend running"}