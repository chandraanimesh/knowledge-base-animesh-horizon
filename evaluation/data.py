EVAL_CASES = [
    {
        "name": "exact_company_id",
        "query": "What is company 0008?",
        "expected": {
            "company_id": "8",
        },
    },
    {
        "name": "funded_titan_ahmedabad",
        "query": (
            "Find a funded company in Ahmedabad "
            "whose name contains Titan."
        ),
        "expected": {
            "city": "Ahmedabad",
            "funding_status": "Funded",
            "company_name_contains": "Titan",
        },
    },
    {
        "name": "funded_ahmedabad",
        "query": "Find funded companies in Ahmedabad.",
        "expected": {
            "city": "Ahmedabad",
            "funding_status": "Funded",
        },
    },
    {
        "name": "analytics_ahmedabad",
        "query": (
            "Find companies with Analytics "
            "in their name in Ahmedabad."
        ),
        "expected": {
            "city": "Ahmedabad",
            "company_name_contains": "Analytics",
        },
    },
    {
        "name": "nonexistent_company",
        "query": "What is company 9999?",
        "expected": {
            "company_id": "9999",
        },
        "expect_no_results": True,
    },
    {
        "name": "nonexistent_city",
        "query": "Find funded companies in Atlantis.",
        "expected": {
            "city": "Atlantis",
            "funding_status": "Funded",
        },
        "expect_no_results": True,
    },
    {
        "name": "city_typo",
        "query": "Find funded companies in Ahemdabad.",
        "expected": {
            "city": "Ahmedabad",
            "funding_status": "Funded",
        },
    },
]