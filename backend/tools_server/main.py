import uvicorn
from fastapi import FastAPI
from .tools.milvus_search import router as milvus_router
from .tools.comb import router as comb_router
from .tools.milvus_search import SearchInputSchema, QueryInputSchema
from .tools.comb import CombInputSchema

app = FastAPI()

app.include_router(milvus_router, prefix="/milvus", tags=["milvus"])
app.include_router(comb_router, prefix="/comb", tags=["comb"])

@app.get("/openapi.json")
async def get_openapi_spec():
    return {
        "search": SearchInputSchema,
        "query": QueryInputSchema,
        "comb": CombInputSchema,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
