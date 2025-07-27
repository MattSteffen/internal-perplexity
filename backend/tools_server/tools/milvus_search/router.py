from fastapi import APIRouter, HTTPException
from .schemas import SearchInput, QueryInput
from . import tools

router = APIRouter()

@router.post("/search")
async def search_endpoint(input_data: SearchInput):
    results = tools.search(input_data)
    if "error" in results:
        raise HTTPException(status_code=503, detail=results["error"])
    return results

@router.post("/query")
async def query_endpoint(input_data: QueryInput):
    results = tools.query(input_data)
    if "error" in results:
        raise HTTPException(status_code=503, detail=results["error"])
    return results
