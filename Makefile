.PHONY: install lint format test cov eval web run serve docker-build docker-run deploy

IMAGE ?= equitymind:local
REGION ?= us-central1
SERVICE ?= equitymind

install:        ## Install all deps (incl. dev)
	uv sync

lint:           ## Static checks
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check . --fix
	uv run ruff format .

test:           ## Offline unit tests
	uv run pytest -q

cov:            ## Tests with coverage report
	uv run pytest -q --cov --cov-report=term-missing --cov-report=xml

eval:           ## Live agent evals (needs GOOGLE_API_KEY)
	uv run python evals/run_eval.py --min-pass-rate 0.8

web:            ## ADK dev UI with traces at http://localhost:8000
	uv run adk web --session_service_uri sqlite+aiosqlite:///./data/sessions.db

run:            ## Terminal chat
	uv run adk run equitymind

serve:          ## Production server locally on :8080 with persistent sessions
	mkdir -p data && EQUITYMIND_SESSION_URI=sqlite+aiosqlite:///./data/sessions.db uv run python server.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8080:8080 --env-file equitymind/.env $(IMAGE)

deploy:         ## Manual Cloud Run deploy from source (CI does this on main)
	gcloud run deploy $(SERVICE) --source . --region $(REGION) \
	  --set-secrets GOOGLE_API_KEY=equitymind-google-api-key:latest \
	  --set-env-vars EQUITYMIND_TRACE_TO_CLOUD=true,GOOGLE_GENAI_USE_VERTEXAI=FALSE
