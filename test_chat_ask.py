import asyncio
from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
from rag_app.core.config import get_settings
from rag_app.db.models import User, Department
from rag_app.retrieval.service import RetrievalService
from rag_app.generation.service import GenerationService
from rag_app.research.graph import research_graph

async def main():
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with async_session() as session:
        # We assume a valid user and maybe departments are set up
        # We'll just create a mock user for now
        user_id = uuid4()
        user = User(id=user_id, email="test@example.com")
        
        retrieval_service = RetrievalService(session)
        generation_service = GenerationService(session)
        
        config = {
            "configurable": {
                "retrieval_service": retrieval_service,
                "generation_service": generation_service,
                "db_session": session,
            }
        }
        
        print("=== Test 1: Conversation Routing (Greeting) ===")
        state1 = {
            "query": "hi",
            "top_k": 5,
            "rerank": False,
            "user": user,
            "allowed_departments": [],
        }
        result1 = await research_graph.ainvoke(state1, config)
        print("Response:", result1.get("final_response").answer)
        print()
        
        print("=== Test 2: Knowledge Query (Insufficient context) ===")
        state2 = {
            "query": "What is the secret code to the mainframe?",
            "top_k": 5,
            "rerank": False,
            "user": user,
            "allowed_departments": [],
        }
        result2 = await research_graph.ainvoke(state2, config)
        print("Response:", result2.get("final_response").answer)
        print()
        
        print("Testing Complete.")

if __name__ == "__main__":
    asyncio.run(main())
