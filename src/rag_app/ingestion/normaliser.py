import re
from typing import Any


FIELD_ALIASES = {
    "company_id": {
        "id",
        "company_id",
        "companyid",
        "company id",
    },
    "company_name": {
        "company",
        "company_name",
        "companyname",
        "company name",
    },
    "funding_status": {
        "funding",
        "funding_status",
        "fundingstatus",
        "funding status",
    },
    "city": {
        "city",
        "location",
        "location_city",
        "location city",
    },
    "country": {
        "country",
    },
}


def normalize_column_name(name: str) -> str:
    value = name.strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def canonical_field_name(name: str) -> str:
    normalized = normalize_column_name(name)

    for canonical, aliases in FIELD_ALIASES.items():
        normalized_aliases = {
            normalize_column_name(alias)
            for alias in aliases
        }

        if normalized in normalized_aliases:
            return canonical

    return normalized


def normalize_company_id(value: Any) -> str:
    value = str(value).strip()

    if value.isdigit():
        return str(int(value))

    return value
