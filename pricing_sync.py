\
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DRIFT = 2


class SyncError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    openrouter_base_url: str
    openrouter_api_key: str
    openrouter_timeout: float

    langfuse_base_url: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_timeout: float
    langfuse_model_name_prefix: str
    langfuse_allow_recreate: bool
    langfuse_page_size: int

    model_mappings: dict[str, str]
    price_field_map: dict[str, str]
    required_openrouter_price_fields: tuple[str, ...]
    price_compare_tolerance: Decimal


@dataclass(frozen=True)
class DesiredModel:
    opencode_model: str
    openrouter_model: str
    langfuse_model_name: str
    match_pattern: str
    prices: dict[str, Decimal]


def fail(message: str) -> None:
    raise SyncError(message)


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        fail(f"env file not found: {path}")

    result: dict[str, str] = {}

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        if "=" not in line:
            fail(f"{path}:{line_number}: expected KEY=VALUE")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            fail(f"{path}:{line_number}: invalid variable name: {key}")

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            quote = value[0]
            value = value[1:-1]

            if quote == '"':
                value = bytes(value, "utf-8").decode("unicode_escape")

        result[key] = value

    return result


def require(env: dict[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        fail(f"{name} is not configured")
    return value


def env_float(env: dict[str, str], name: str, default: str) -> float:
    raw = env.get(name, default).strip()
    try:
        value = float(raw)
    except ValueError:
        fail(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        fail(f"{name} must be greater than zero")
    return value


def env_int(env: dict[str, str], name: str, default: str) -> int:
    raw = env.get(name, default).strip()
    try:
        value = int(raw)
    except ValueError:
        fail(f"{name} must be an integer")
    if value <= 0:
        fail(f"{name} must be greater than zero")
    return value


def env_bool(env: dict[str, str], name: str, default: str = "0") -> bool:
    raw = env.get(name, default).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    fail(f"{name} must be one of: 0, 1, true, false, yes, no, on, off")


def parse_json_object(env: dict[str, str], name: str) -> dict[str, str]:
    raw = require(env, name)

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        fail(f"{name} is not valid JSON: {error}")

    if not isinstance(value, dict) or not value:
        fail(f"{name} must be a non-empty JSON object")

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            fail(f"{name} contains an invalid key")
        if not isinstance(item, str) or not item.strip():
            fail(f"{name}[{key!r}] must be a non-empty string")
        result[key.strip()] = item.strip()

    return result


def load_config(path: Path) -> Config:
    env = parse_env_file(path)

    required_fields = tuple(
        item.strip()
        for item in env.get(
            "REQUIRED_OPENROUTER_PRICE_FIELDS",
            "prompt,completion",
        ).split(",")
        if item.strip()
    )

    if not required_fields:
        fail("REQUIRED_OPENROUTER_PRICE_FIELDS must not be empty")

    tolerance_raw = env.get(
        "PRICE_COMPARE_TOLERANCE",
        "0.000000000001",
    ).strip()

    try:
        tolerance = Decimal(tolerance_raw)
    except InvalidOperation:
        fail("PRICE_COMPARE_TOLERANCE must be a decimal number")

    if tolerance < 0:
        fail("PRICE_COMPARE_TOLERANCE must not be negative")

    return Config(
        openrouter_base_url=require(env, "OPENROUTER_BASE_URL").rstrip("/"),
        openrouter_api_key=require(env, "OPENROUTER_API_KEY"),
        openrouter_timeout=env_float(
            env, "OPENROUTER_TIMEOUT_SECONDS", "20"
        ),
        langfuse_base_url=require(env, "LANGFUSE_BASE_URL").rstrip("/"),
        langfuse_public_key=require(env, "LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=require(env, "LANGFUSE_SECRET_KEY"),
        langfuse_timeout=env_float(
            env, "LANGFUSE_TIMEOUT_SECONDS", "20"
        ),
        langfuse_model_name_prefix=env.get(
            "LANGFUSE_MODEL_NAME_PREFIX",
            "openrouter-sync:",
        ),
        langfuse_allow_recreate=env_bool(
            env, "LANGFUSE_ALLOW_RECREATE", "0"
        ),
        langfuse_page_size=env_int(
            env, "LANGFUSE_PAGE_SIZE", "100"
        ),
        model_mappings=parse_json_object(env, "MODEL_MAPPINGS_JSON"),
        price_field_map=parse_json_object(env, "PRICE_FIELD_MAP_JSON"),
        required_openrouter_price_fields=required_fields,
        price_compare_tolerance=tolerance,
    )


def http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: Any = None,
    timeout: float,
    accepted_statuses: set[int] | None = None,
) -> tuple[int, Any]:
    data: bytes | None = None

    request_headers = {
        "Accept": "application/json",
        "User-Agent": "model-pricing-sync/1.0",
    }

    if headers:
        request_headers.update(headers)

    if payload is not None:
        data = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = raw
    except urllib.error.URLError as error:
        fail(f"HTTP request failed: {method} {url}: {error}")

    if accepted_statuses is None:
        accepted_statuses = {200}

    if status not in accepted_statuses:
        fail(
            f"HTTP {status}: {method} {url}: "
            f"{json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body}"
        )

    return status, body


def openrouter_models(config: Config) -> dict[str, dict[str, Any]]:
    url = f"{config.openrouter_base_url}/models"

    _, body = http_json(
        "GET",
        url,
        headers={
            "Authorization": f"Bearer {config.openrouter_api_key}",
        },
        timeout=config.openrouter_timeout,
    )

    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        fail("unexpected OpenRouter /models response")

    result: dict[str, dict[str, Any]] = {}

    for item in body["data"]:
        if not isinstance(item, dict):
            continue

        model_id = item.get("id")
        if isinstance(model_id, str):
            result[model_id] = item

    return result


def decimal_price(value: Any, *, field: str, model_id: str) -> Decimal:
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError):
        fail(
            f"OpenRouter model {model_id}: pricing.{field} "
            f"is not a decimal: {value!r}"
        )

    if price < 0:
        fail(
            f"OpenRouter model {model_id}: pricing.{field} "
            "must not be negative"
        )

    return price


def desired_models(
    config: Config,
    catalog: dict[str, dict[str, Any]],
) -> list[DesiredModel]:
    result: list[DesiredModel] = []

    for opencode_model, openrouter_model in config.model_mappings.items():
        source = catalog.get(openrouter_model)

        if source is None:
            fail(
                f"OpenRouter model not found: {openrouter_model} "
                f"(mapped from {opencode_model})"
            )

        pricing = source.get("pricing")
        if not isinstance(pricing, dict):
            fail(f"OpenRouter model {openrouter_model} has no pricing object")

        for required_field in config.required_openrouter_price_fields:
            if required_field not in pricing:
                fail(
                    f"OpenRouter model {openrouter_model} is missing "
                    f"required pricing field: {required_field}"
                )

        prices: dict[str, Decimal] = {}

        for openrouter_field, langfuse_usage_key in config.price_field_map.items():
            value = pricing.get(openrouter_field)

            if value is None or value == "":
                continue

            price = decimal_price(
                value,
                field=openrouter_field,
                model_id=openrouter_model,
            )

            prices[langfuse_usage_key] = price

        if not prices:
            fail(
                f"OpenRouter model {openrouter_model}: no mapped prices found"
            )

        result.append(
            DesiredModel(
                opencode_model=opencode_model,
                openrouter_model=openrouter_model,
                langfuse_model_name=(
                    f"{config.langfuse_model_name_prefix}{opencode_model}"
                ),
                match_pattern=f"(?i)^{re.escape(opencode_model)}$",
                prices=prices,
            )
        )

    return result


def langfuse_auth(config: Config) -> str:
    raw = (
        f"{config.langfuse_public_key}:"
        f"{config.langfuse_secret_key}"
    ).encode("utf-8")

    return "Basic " + base64.b64encode(raw).decode("ascii")


def langfuse_headers(config: Config) -> dict[str, str]:
    return {
        "Authorization": langfuse_auth(config),
    }


def list_langfuse_models(config: Config) -> list[dict[str, Any]]:
    page = 1
    models: list[dict[str, Any]] = []

    while True:
        query = urllib.parse.urlencode(
            {
                "page": page,
                "limit": config.langfuse_page_size,
            }
        )

        url = (
            f"{config.langfuse_base_url}"
            f"/api/public/models?{query}"
        )

        _, body = http_json(
            "GET",
            url,
            headers=langfuse_headers(config),
            timeout=config.langfuse_timeout,
        )

        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            fail("unexpected Langfuse models response")

        batch = [
            item
            for item in body["data"]
            if isinstance(item, dict)
        ]

        models.extend(batch)

        meta = body.get("meta")
        if not isinstance(meta, dict):
            break

        total_pages = meta.get("totalPages")
        if isinstance(total_pages, int):
            if page >= total_pages:
                break
            page += 1
            continue

        total_items = meta.get("totalItems")
        if isinstance(total_items, int):
            if len(models) >= total_items:
                break
            page += 1
            continue

        if len(batch) < config.langfuse_page_size:
            break

        page += 1

    return models


def managed_langfuse_models(
    config: Config,
    models: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for model in models:
        name = model.get("modelName")

        if (
            isinstance(name, str)
            and name.startswith(config.langfuse_model_name_prefix)
        ):
            result[name] = model

    return result


def desired_payload(model: DesiredModel) -> dict[str, Any]:
    return {
        "modelName": model.langfuse_model_name,
        "matchPattern": model.match_pattern,
        "unit": "TOKENS",
        "pricingTiers": [
            {
                "name": "OpenRouter",
                "isDefault": True,
                "priority": 0,
                "conditions": [],
                "prices": {
                    key: float(value)
                    for key, value in sorted(model.prices.items())
                },
            }
        ],
    }


def existing_prices(model: dict[str, Any]) -> dict[str, Decimal]:
    tiers = model.get("pricingTiers")

    if not isinstance(tiers, list):
        return {}

    default_tier = None

    for tier in tiers:
        if isinstance(tier, dict) and tier.get("isDefault") is True:
            default_tier = tier
            break

    if default_tier is None:
        return {}

    prices = default_tier.get("prices")
    if not isinstance(prices, dict):
        return {}

    result: dict[str, Decimal] = {}

    for key, value in prices.items():
        try:
            result[str(key)] = Decimal(str(value))
        except (InvalidOperation, ValueError):
            continue

    return result


def decimals_equal(
    left: Decimal,
    right: Decimal,
    tolerance: Decimal,
) -> bool:
    return abs(left - right) <= tolerance


def model_matches(
    config: Config,
    desired: DesiredModel,
    existing: dict[str, Any],
) -> bool:
    if existing.get("matchPattern") != desired.match_pattern:
        return False

    current_prices = existing_prices(existing)

    if set(current_prices) != set(desired.prices):
        return False

    for key, expected in desired.prices.items():
        actual = current_prices[key]
        if not decimals_equal(actual, expected, config.price_compare_tolerance):
            return False

    return True


def create_langfuse_model(
    config: Config,
    desired: DesiredModel,
) -> None:
    url = f"{config.langfuse_base_url}/api/public/models"

    http_json(
        "POST",
        url,
        headers=langfuse_headers(config),
        payload=desired_payload(desired),
        timeout=config.langfuse_timeout,
        accepted_statuses={200, 201},
    )


def update_langfuse_model(
    config: Config,
    desired: DesiredModel,
    existing: dict[str, Any],
) -> None:
    model_id = existing.get("id")

    if not isinstance(model_id, str) or not model_id:
        fail(
            f"Langfuse model {desired.langfuse_model_name} "
            "has no valid id"
        )

    url = (
        f"{config.langfuse_base_url}"
        f"/api/public/models/{urllib.parse.quote(model_id, safe='')}"
    )

    status, body = http_json(
        "PUT",
        url,
        headers=langfuse_headers(config),
        payload=desired_payload(desired),
        timeout=config.langfuse_timeout,
        accepted_statuses={200, 201, 204, 404, 405},
    )

    if status in {200, 201, 204}:
        return

    if not config.langfuse_allow_recreate:
        fail(
            "Langfuse model update endpoint is unavailable "
            f"(HTTP {status}) for {desired.langfuse_model_name}. "
            "Set LANGFUSE_ALLOW_RECREATE=1 to allow delete + recreate."
        )

    delete_langfuse_model(config, model_id)
    create_langfuse_model(config, desired)


def delete_langfuse_model(
    config: Config,
    model_id: str,
) -> None:
    url = (
        f"{config.langfuse_base_url}"
        f"/api/public/models/{urllib.parse.quote(model_id, safe='')}"
    )

    http_json(
        "DELETE",
        url,
        headers=langfuse_headers(config),
        timeout=config.langfuse_timeout,
        accepted_statuses={200, 204},
    )


def format_per_million(price: Decimal) -> str:
    value = price * Decimal(1_000_000)
    normalized = value.normalize()
    return f"${normalized}/M"


def print_desired_table(models: list[DesiredModel]) -> None:
    headers = [
        "OpenCode model",
        "OpenRouter model",
        "Langfuse prices",
    ]

    rows: list[list[str]] = []

    for model in models:
        price_text = ", ".join(
            f"{key}={format_per_million(value)}"
            for key, value in sorted(model.prices.items())
        )

        rows.append(
            [
                model.opencode_model,
                model.openrouter_model,
                price_text,
            ]
        )

    widths = [
        max(
            len(headers[index]),
            *(len(row[index]) for row in rows),
        )
        for index in range(len(headers))
    ]

    print(
        "  ".join(
            headers[index].ljust(widths[index])
            for index in range(len(headers))
        )
    )

    print(
        "  ".join("-" * width for width in widths)
    )

    for row in rows:
        print(
            "  ".join(
                row[index].ljust(widths[index])
                for index in range(len(headers))
            )
        )


def check_drift(
    config: Config,
    desired: list[DesiredModel],
    existing: dict[str, dict[str, Any]],
) -> list[tuple[DesiredModel, str]]:
    drift: list[tuple[DesiredModel, str]] = []

    for model in desired:
        current = existing.get(model.langfuse_model_name)

        if current is None:
            drift.append((model, "missing"))
            continue

        if not model_matches(config, model, current):
            drift.append((model, "different"))

    return drift


def run_show(config: Config) -> int:
    catalog = openrouter_models(config)
    desired = desired_models(config, catalog)
    print_desired_table(desired)
    return EXIT_OK


def run_check(config: Config) -> int:
    catalog = openrouter_models(config)
    desired = desired_models(config, catalog)

    all_existing = list_langfuse_models(config)
    existing = managed_langfuse_models(config, all_existing)

    drift = check_drift(config, desired, existing)

    if not drift:
        print("pricing check: OK")
        return EXIT_OK

    for model, state in drift:
        print(
            f"DRIFT: {model.opencode_model}: {state} "
            f"({model.langfuse_model_name})"
        )

    return EXIT_DRIFT


def run_sync(config: Config) -> int:
    catalog = openrouter_models(config)
    desired = desired_models(config, catalog)

    all_existing = list_langfuse_models(config)
    existing = managed_langfuse_models(config, all_existing)

    created = 0
    updated = 0
    unchanged = 0

    for model in desired:
        current = existing.get(model.langfuse_model_name)

        if current is None:
            create_langfuse_model(config, model)
            print(
                f"created: {model.langfuse_model_name} "
                f"<- {model.openrouter_model}"
            )
            created += 1
            continue

        if model_matches(config, model, current):
            print(
                f"unchanged: {model.langfuse_model_name} "
                f"<- {model.openrouter_model}"
            )
            unchanged += 1
            continue

        update_langfuse_model(config, model, current)
        print(
            f"updated: {model.langfuse_model_name} "
            f"<- {model.openrouter_model}"
        )
        updated += 1

    print()
    print(
        "pricing sync: OK "
        f"(created={created}, updated={updated}, unchanged={unchanged})"
    )

    return EXIT_OK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize OpenRouter model prices into "
            "custom Langfuse model definitions."
        )
    )

    parser.add_argument(
        "--env",
        default=".env",
        help="Path to env file (default: .env)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--show",
        action="store_true",
        help="Show mapped OpenRouter prices without changing Langfuse",
    )

    mode.add_argument(
        "--check",
        action="store_true",
        help="Check whether managed Langfuse model prices are synchronized",
    )

    mode.add_argument(
        "--sync",
        action="store_true",
        help="Create/update managed Langfuse model definitions",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config(Path(args.env))

        if args.show:
            return run_show(config)

        if args.check:
            return run_check(config)

        if args.sync:
            return run_sync(config)

        return EXIT_ERROR

    except SyncError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
