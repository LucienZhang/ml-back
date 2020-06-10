from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
import requests
from PIL import Image
import numpy as np
import json

from ml.core.config import settings

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
async def mnist(*, img: UploadFile = File(...)):
    image = Image.open(img.file).resize((28, 28)).convert('L')
    image = np.array(image, dtype='float32').reshape((-1, 28, 28, 1))
    image /= 255.0
    data = json.dumps({
        "instances": image.tolist()
    })
    headers = {"content-type": "application/json"}
    url = settings.TF_PREDICT_URL.format('mnist')
    json_response = await requests.post(url, data=data, headers=headers)
    predictions = np.array(json.loads(json_response.text)['predictions'])
    result = np.argmax(predictions, axis=-1)[0].item()
    prob = '{:.2%}'.format(np.max(predictions, axis=-1)[0].item())
    return json.dumps(dict(result=result, prob=prob))
