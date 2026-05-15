# TIL-AI 2026 AE Agent

This repository contains a fast rule-based Autonomous Exploration (AE) agent for the TIL-AI 2026 challenge.

## What the AE agent does

The evaluator repeatedly sends observations to `POST /ae` on port `5005`. The server must return one integer action.

Action meanings:

- `0`: move forward
- `1`: move backward
- `2`: turn left
- `3`: turn right
- `4`: stay
- `5`: place bomb

The current agent is in `ae/src/ae_manager.py`.

It is not a neural network yet. It is a fast planner that:

1. reads the 7 x 5 agent viewcone and 5 x 5 base viewcone;
2. builds an internal map of known walls, destructible walls, items, bombs, enemies, and enemy bases;
3. prioritises missions, resources, recon tokens, and enemy-base bombing opportunities;
4. avoids known bomb blast zones;
5. explores unknown cells when no high-value target is visible;
6. returns actions very quickly, which helps the speed score.

## Run locally

From the `ae` folder:

```bash
pip install -r requirements.txt
uvicorn src.ae_server:app --port 5005 --host 0.0.0.0
```

Health check:

```bash
curl http://localhost:5005/health
```

Reset endpoint:

```bash
curl http://localhost:5005/reset
```

Smoke test:

```bash
python test_smoke.py
```

## Docker

From the `ae` folder:

```bash
docker build -t til-ae .
docker run -p 5005:5005 til-ae
```

## Next improvement path

The rule-based agent is the safe first submission because it is fast, understandable, and does not need GPU inference. The next step is to use the A100 to train a policy with the official `til_environment`, then ensemble or replace this planner only if the trained model beats it in actual rollouts.
