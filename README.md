# BADA: Business-aware Desktop automation Agent

BADA is an agentic framework to interact with computer iterfaces that leverages the vasts ammount of information already available on bussiness contexts to improve results when executing computer tasks via agents.

It also stores all past execution in a vectorized memory and uses it to improve similar plans.

## Prequisites

- Cuda 12.4
- [uv](https://docs.astral.sh/uv)

## Instalation

Clone the repository
```bash
git clone https://github.com/RPA-US/BADA.git
cd BADA
```

Start environment with [uv](https://docs.astral.sh/uv)
```bash
uv python install 3.10 # If not available
uv venv
uv sync
```