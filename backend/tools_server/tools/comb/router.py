from fastapi import APIRouter, HTTPException
from .schemas import CombInput
from ..milvus_search.utils import connect_milvus
import ollama

router = APIRouter()


@router.post("/comb")
async def comb_endpoint(input_data: CombInput):
    milvus_client = connect_milvus()
    if not milvus_client:
        raise HTTPException(
            status_code=503, detail="Could not connect to Milvus database."
        )

    # 1. Query for all documents with a given filter
    query_results = milvus_client.query(
        collection_name="test_collection",
        filter=" and ".join(input_data.filters) if input_data.filters else "",
        output_fields=["text"],  # We only need the text for the LLM
        limit=1000,  # Get a large number of documents to comb through
    )

    # 2. Ask LLM to read and give its thoughts
    combined_text = "\n".join([res["text"]] for res in query_results["results"])
    prompt = f"From the perspective of {input_data.perspective}, read the following documents and provide your thoughts and insights:\n\n{combined_text}"

    try:
        client = ollama.Client(host="http://localhost:11434")
        response = client.chat(
            model="o1-preview",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return {"thoughts": response["message"]["content"]}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error communicating with LLM: {e}"
        )
