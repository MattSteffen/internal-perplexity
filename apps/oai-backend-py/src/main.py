"""Main FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.auth import init_oauth
from src.config import app_config, settings
from src.milvus_client import MilvusClientPool, get_milvus_uri
from src.routes import include_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool_config = app_config.milvus_pool
    app.state.milvus_pool = MilvusClientPool(
        uri=get_milvus_uri(),
        ttl_seconds=pool_config.ttl_seconds,
        max_size=pool_config.max_size,
    )
    yield
    # Clean up the Milvus client pool and release the resources
    app.state.milvus_pool.close_all()


app = FastAPI(
    title="OpenAI-Compatible Backend",
    description="OpenAI-compatible API proxy for Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS to allow frontend requests
cors_config = app_config.cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.allow_origins,
    allow_credentials=cors_config.allow_credentials,
    allow_methods=cors_config.allow_methods,
    allow_headers=cors_config.allow_headers,
)

# Initialize OAuth with the FastAPI app
init_oauth(app)

include_routes(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
