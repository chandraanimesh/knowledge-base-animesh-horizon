


# from fastmcp import FastMCP
#
# from rag_app.db.session import AsyncSessionLocal
# from rag_app.generation.service import GenerationService
# from rag_app.schemas.rag import GenerateRequest
#
# mcp = FastMCP("production-rag")
#
#
# @mcp.tool()
# async def query_rag(query: str, top_k: int = 5, rerank: bool = True) -> dict:
#     async with AsyncSessionLocal() as session:
#         service = GenerationService(session)
#         response = await service.generate_answer(
#             GenerateRequest(query=query, top_k=top_k, rerank=rerank)
#         )
#         return response.model_dump()
