# MissionMind

**AI-powered autonomous mission planning for a simulated Mars rover.**

Built for the **IBM Bob AI Builders Challenge** using **IBM Bob** and **IBM Granite**.

---

## Overview

MissionMind is an autonomous Mars rover mission commander that plans scientific exploration routes while balancing:

- Scientific value
- Battery and energy limits
- Mission duration
- Terrain risk
- Communication constraints
- Environmental conditions
- Safe return-to-base requirements

Instead of relying on a single AI model for every decision, MissionMind uses multiple specialist AI agents together with a deterministic safety layer.

The system can also dynamically replan when unexpected mission events occur.

---

## Key Features

### Multi-Agent Mission Planning

MissionMind uses four AI roles:

- **Science Agent** — evaluates and prioritizes scientifically valuable targets.
- **Resource Agent** — evaluates battery, energy usage, and mission time.
- **Safety Agent** — evaluates terrain, environmental, and communication risks.
- **Mission Commander** — combines the specialist recommendations into a final mission plan.

### Deterministic Safety Validation

AI recommendations do not have the final say on critical safety constraints.

Every mission plan is checked by a deterministic `SafetyValidator` before it is accepted.

Hard constraints include resource limits and safe-return requirements.

Critical situations such as severe battery failure and explicit Return-to-Base commands use deterministic emergency behavior rather than relying on LLM judgment.

### Autonomous Replanning

MissionMind can react to mission events including:

- Battery Failure
- Terrain Hazard
- New Scientific Discovery
- Communication Loss
- Return to Base

The rover's mission plan, route, confidence, and reasoning update dynamically in response.

### Interactive Mission Control Dashboard

The React dashboard displays:

- Mars mission map
- Rover status
- Battery level
- Current route
- AI mission decision
- Mission confidence
- Energy and time estimates
- Mission event controls
- Live event log
- Terrain hazards
- Newly discovered waypoints

---

## IBM Technology

### IBM Bob

**IBM Bob** was used as the primary AI-assisted development environment for building, debugging, testing, and integrating MissionMind.

### IBM Granite

The live demo uses:

**IBM Granite 4.2 3B**

running locally through **Ollama**.

Local inference is especially relevant to the Mars rover scenario because mission intelligence should not depend entirely on a continuous connection to a remote AI service.

MissionMind also contains a `WatsonxClient` abstraction so the LLM backend can be swapped without changing the agent architecture.

---

## Architecture

                    ┌─────────────────────┐
                    │   MissionMind UI    │
                    │    React + Vite     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │   Mission API       │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Mission Planner          │
              └───────────────┬────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │ Science Agent  │  │ Resource Agent │  │  Safety Agent  │
 └────────┬───────┘  └────────┬───────┘  └────────┬───────┘
          └───────────────────┼───────────────────┘
                              ▼
                  ┌──────────────────────┐
                  │  Mission Commander   │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Deterministic Safety │
                  │      Validator       │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │   Safe Mission Plan  │
                  └──────────────────────┘
