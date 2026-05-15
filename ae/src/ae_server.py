"""Runs the AE FastAPI server.

Expected endpoint:
POST /ae
{
  "instances": [{"observation": {...}}]
}

Response:
{
  "predictions": [{"action": 0}]
}
"""

from __future__ import annotations

from fastapi import FastAPI, Request

from ae_manager import AEManager

app = FastAPI()
manager = AEManager()


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]]:
    """Return one action prediction for each supplied instance."""
    input_json = await request.json()
    predictions: list[dict[str, int]] = []

    for instance in input_json.get("instances", []):
        observation = instance.get("observation", {})
        predictions.append({"action": int(manager.ae(observation))})

    return {"predictions": predictions}


@app.get("/reset")
def reset() -> dict[str, str]:
    """Clear persistent episode state between rounds."""
    manager.reset()
    return {"message": "reset ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""
    return {"message": "health ok"}
