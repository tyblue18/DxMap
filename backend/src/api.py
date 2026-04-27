"""FastAPI app exposing the coding pipeline."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .coder.pipeline import run
from .coder.schema import CodingRequest, CodingResponse

app = FastAPI(
    title="Clinical Coder",
    description="ICD-10-CM and CPT code suggestion with span attribution",
    version="0.1.0",
)

# CORS for local development — tighten this for production deploys.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/code", response_model=CodingResponse)
def code(request: CodingRequest) -> CodingResponse:
    try:
        return run(request)
    except FileNotFoundError as exc:
        # Most likely cause: indices weren't built.
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not initialized: {exc}. Run `python -m src.data.build_indices`.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(exc).__name__}: {exc}") from exc
