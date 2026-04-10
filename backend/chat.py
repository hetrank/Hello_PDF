from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Chat, Document, Message
from auth import get_current_user
from generation import generate_answer
import os

# RAG Imports:
from ocr_utils_old import pdf_to_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/chat/create")
def create_chat(name: str, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    chat = Chat(name=name, user_id=current_user.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("/chat/list")
def list_chats(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Chat).filter(Chat.user_id == current_user.id).all()


@router.post("/chat/{chat_id}/upload")
def upload_pdf(chat_id: int, file: UploadFile = File(...), current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Store metadata
    doc = Document(file_name=file.filename, file_path=file_path, chat_id=chat_id)
    db.add(doc)
    db.commit()

    #convert pdf -> documents
    documents = pdf_to_doc(file_path)

    #chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    #vector store per chat
    persist_dir = os.path.join(VECTOR_DIR, f"chat_{chat_id}")

    if os.path.exists(persist_dir):
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )
        vectordb.add_documents(chunks)
    else:
        Chroma.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            persist_directory=persist_dir
        )

    return {"message": "Uploaded and processed"}


@router.post("/chat/{chat_id}/query")
def query_chat(chat_id: int, question: str, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    
    def generator():
        for token in generate_answer(chat_id, question):
            yield token

    return StreamingResponse(generator(), media_type="text/plain")


@router.post("/chat/{chat_id}/message")
def save_message(chat_id: int, data: dict, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    
    user_msg = Message(chat_id=chat_id, role="user", content=data["question"])
    bot_msg = Message(chat_id=chat_id, role="assistant", content=data["answer"])

    db.add(user_msg)
    db.add(bot_msg)
    db.commit()

    return {"message": "Saved"}


@router.get("/chat/{chat_id}/messages")
def get_messages(chat_id: int, current_user=Depends(get_current_user), db: Session = Depends(get_db)):

    messages = db.query(Message).filter(Message.chat_id == chat_id).all()

    return [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]

