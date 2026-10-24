# Model Pricing Sync

Synchronizes model prices from OpenRouter to Langfuse.

Requires Python 3.10+. No external dependencies.

## Setup

```bash
cp .env.example .env
chmod 600 .env
```

Configure `.env` with OpenRouter/Langfuse credentials and model mappings.

Example:

```dotenv
MODEL_MAPPINGS_JSON={"gpt-6-astra":"openai/gpt-6-astra","gpt-6-sol":"openai/gpt-6-sol","gpt-6-luna-fast":"openai/gpt-6-luna"}
```

All supported variables are documented in `.env.example`.

## Usage

```bash
python3 pricing_sync.py --show
python3 pricing_sync.py --check
python3 pricing_sync.py --sync
```

`--show` displays OpenRouter prices.

`--check` detects Langfuse pricing drift and exits with code `2` when drift exists.

`--
