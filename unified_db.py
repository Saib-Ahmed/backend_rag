import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import bcrypt
import uuid
from pymongo import MongoClient
import logging

logging.getLogger("pymongo").setLevel(logging.WARNING)
logger = logging.getLogger("unified_db")

MONGO_URI = "mongodb+srv://shashankdev9745_db_user:hWfZ5o3dxY96axQL@cluster0.8igmf03.mongodb.net/?appName=Cluster0"

try:
    import certifi
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client.rag_database
    # Check connection
    client.admin.command('ping')
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    db = None

def get_db_connection():
    return db

def init_db():
    if db is not None:
        try:
            db.users.create_index("username", unique=True)
            db.users.create_index("email", unique=True)
            db.sessions.create_index("session_id", unique=True)
            db.sessions.create_index("user_id")
            db.history.create_index("session_id")
            db.document_metadata.create_index("file_name", unique=True)
            
            # Check and seed admin user saib@gmail.com / 123456
            admin_email = "saib@gmail.com"
            admin_user = db.users.find_one({"$or": [{"email": admin_email}, {"username": admin_email}]})
            if not admin_user:
                logger.info("Initializing admin user saib@gmail.com")
                user_id = str(uuid.uuid4())
                db.users.insert_one({
                    "user_id": user_id,
                    "username": "saib@gmail.com",
                    "email": admin_email,
                    "hashed_password": get_password_hash("123456"),
                    "role": "admin",
                    "created_at": datetime.utcnow()
                })
            else:
                # Ensure the user has the admin role
                db.users.update_one(
                    {"user_id": admin_user["user_id"]},
                    {"$set": {"role": "admin"}}
                )
        except Exception as e:
            logger.error(f"Error creating indexes or seeding admin user: {e}")

# Initialize DB on import
init_db()

# Password Hashing
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

# --- USER MANAGEMENT ---
def get_user(identifier: str) -> Optional[Dict[str, Any]]:
    if db is None: return None
    user = db.users.find_one({"$or": [{"email": identifier}, {"username": identifier}]})
    return dict(user) if user else None

def create_user(username: str, email: str, password: str) -> Optional[str]:
    if db is None: return None
    if get_user(username) or get_user(email):
        return None
    user_id = str(uuid.uuid4())
    db.users.insert_one({
        "user_id": user_id,
        "username": username,
        "email": email,
        "hashed_password": get_password_hash(password),
        "created_at": datetime.utcnow()
    })
    return user_id

# --- SESSION MANAGEMENT ---
def create_session(user_id: str, title: str = "New Chat") -> str:
    if db is None: return str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    db.sessions.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "title": title,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    return session_id

def get_user_sessions(user_id: str) -> List[Dict[str, Any]]:
    if db is None: return []
    if user_id == "default_user":
        sessions = list(db.sessions.find().sort("updated_at", -1))
    else:
        sessions = list(db.sessions.find({"user_id": user_id}).sort("updated_at", -1))
    
    # ensure consistent return format
    return [{"session_id": s["session_id"], "title": s["title"], "updated_at": s.get("updated_at")} for s in sessions]

def update_session_title(session_id: str, new_title: str):
    if db is None: return
    db.sessions.update_one(
        {"session_id": session_id},
        {"$set": {"title": new_title, "updated_at": datetime.utcnow()}}
    )

def delete_session(session_id: str):
    if db is None: return
    db.sessions.delete_many({"session_id": session_id})
    db.history.delete_many({"session_id": session_id})

# --- HISTORY MANAGEMENT ---
def append_message(session_id: str, role: str, content: str, rag_version: str = "unknown", sources: list = None, metrics: dict = None, user_id: str = "default_user"):
    if db is None: return
    db.history.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "rag_version": rag_version,
        "sources": sources or [],
        "metrics": metrics or {},
        "timestamp": datetime.utcnow()
    })
    db.sessions.update_one(
        {"session_id": session_id},
        {
            "$set": {"updated_at": datetime.utcnow()},
            "$setOnInsert": {
                "session_id": session_id,
                "user_id": user_id,
                "title": content[:30] + "..." if role == "user" else "New Chat",
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )

def get_chat_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if db is None: return []
    docs = list(db.history.find({"session_id": session_id}).sort("timestamp", 1).limit(limit))
    
    result = []
    for d in docs:
        result.append({
            "session_id": d["session_id"],
            "role": d["role"],
            "content": d["content"],
            "rag_version": d.get("rag_version", "unknown"),
            "sources": d.get("sources", []),
            "metrics": d.get("metrics", {}),
            "timestamp": d["timestamp"]
        })
    return result

def get_chat_history_formatted_for_llm(session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    if db is None: return []
    docs = list(db.history.find({"session_id": session_id}).sort("timestamp", -1).limit(limit))
    docs.reverse()
    formatted = []
    for d in docs:
        formatted.append({
            "role": d["role"],
            "content": d["content"]
        })
    return formatted

# --- DOCUMENT METADATA MANAGEMENT ---
def save_document_metadata(
    file_name: str,
    doc_type: str = "PDF",
    source: str = "public",
    source_description: str = "",
    creation_date: str = "",
    rag_version: str = "version1",
):
    """Save or update document metadata in MongoDB with version history tracking."""
    if db is None:
        return
    
    existing = db.document_metadata.find_one({"file_name": file_name})
    update_item = {
        "date": datetime.utcnow().isoformat(),
        "version": rag_version
    }
    
    if existing:
        update_count = existing.get("update_count", 0) + 1
        update_history = existing.get("update_history", [])
        if not isinstance(update_history, list):
            update_history = []
        update_history.append(update_item)
        
        doc = {
            "doc_type": doc_type,
            "source": source,
            "source_description": source_description,
            "creation_date": creation_date,
            "ingestion_date": datetime.utcnow(),
            "rag_version": rag_version,
            "update_count": update_count,
            "update_history": update_history,
        }
        db.document_metadata.update_one(
            {"file_name": file_name},
            {"$set": doc}
        )
    else:
        doc = {
            "file_name": file_name,
            "doc_type": doc_type,
            "source": source,
            "source_description": source_description,
            "creation_date": creation_date,
            "ingestion_date": datetime.utcnow(),
            "rag_version": rag_version,
            "update_count": 0,
            "update_history": [update_item]
        }
        db.document_metadata.insert_one(doc)
        
    logger.info(f"Saved metadata for '{file_name}' (updates: {update_count if existing else 0})")


def get_document_metadata(file_name: str) -> Optional[Dict[str, Any]]:
    """Return metadata for a single document."""
    if db is None:
        return None
    doc = db.document_metadata.find_one({"file_name": file_name}, {"_id": 0})
    if doc and isinstance(doc.get("ingestion_date"), datetime):
        doc["ingestion_date"] = doc["ingestion_date"].isoformat()
    return doc


def get_all_document_metadata() -> List[Dict[str, Any]]:
    """Return metadata for all documents."""
    if db is None:
        return []
    docs = list(db.document_metadata.find({}, {"_id": 0}).sort("ingestion_date", -1))
    for doc in docs:
        if isinstance(doc.get("ingestion_date"), datetime):
            doc["ingestion_date"] = doc["ingestion_date"].isoformat()
    return docs


def delete_document_metadata(file_name: str) -> bool:
    """Delete metadata for a document. Returns True if a record was deleted."""
    if db is None:
        return False
    res = db.document_metadata.delete_one({"file_name": file_name})
    return res.deleted_count > 0


def update_document_metadata(
    file_name: str,
    doc_type: Optional[str] = None,
    source: Optional[str] = None,
    source_description: Optional[str] = None,
    creation_date: Optional[str] = None,
    ingestion_date: Optional[str] = None,
    rag_version: Optional[str] = None,
) -> bool:
    """Update fields for a document's metadata in MongoDB."""
    if db is None:
        return False
    
    updated_fields = {}
    if doc_type is not None:
        updated_fields["doc_type"] = doc_type
    if source is not None:
        updated_fields["source"] = source
    if source_description is not None:
        updated_fields["source_description"] = source_description
    if creation_date is not None:
        updated_fields["creation_date"] = creation_date
    if rag_version is not None:
        updated_fields["rag_version"] = rag_version
        
    if ingestion_date is not None:
        try:
            clean_str = ingestion_date.replace("Z", "") if ingestion_date.endswith("Z") else ingestion_date
            updated_fields["ingestion_date"] = datetime.fromisoformat(clean_str)
        except Exception:
            updated_fields["ingestion_date"] = datetime.utcnow()
            
    if not updated_fields:
        return False
        
    db.document_metadata.update_one(
        {"file_name": file_name},
        {"$set": updated_fields},
        upsert=True
    )
    return True
