from fastapi import APIRouter, HTTPException
from .schemas import CombInput
from .tools import comb

router = APIRouter()


@router.post("/comb")
async def comb_endpoint(input_data: CombInput):
    try:
        return comb(input_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error communicating with LLM: {e}"
        )
