from rag_app.schemas.rag import CompanyFilters


CITY_ALIASES = {
    "ahemdabad": "Ahmedabad",
    "ahmedabad": "Ahmedabad",
}


def normalize_company_filters(
    filters: CompanyFilters,
) -> CompanyFilters:

    city = filters.city

    if city:
        normalized_city = CITY_ALIASES.get(
            city.strip().lower(),
            city.strip(),
        )
    else:
        normalized_city = None

    return filters.model_copy(
        update={
            "city": normalized_city,
        }
    )