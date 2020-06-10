from fastapi import FastAPI, Form
from starlette.middleware.cors import CORSMiddleware

from werewolf.core.config import settings

app = FastAPI()

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.post("/ml-api/mnist")
async def mnist(*, img: bytes = Form(...)):
    return {
        'result': 1,
        'prob': 0.96
    }
