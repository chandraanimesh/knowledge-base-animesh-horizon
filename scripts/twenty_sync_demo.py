from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TWENTY_BASE_URL = (
    __import__("os").environ.get("TWENTY_BASE_URL", "http://localhost:3000")
    .rstrip("/")
)
TWENTY_API_KEY = __import__("os").environ.get("TWENTY_API_KEY")
RAG_BASE_URL = (
    __import__("os").environ.get("RAG_BASE_URL", "http://localhost:8000")
    .rstrip("/")
)

COMPANY_ID = "7a0e5540-c5cc-4fbd-950d-6416093e194f"
COMPANY_NAME = "RAG Integration Demo"
DEPARTMENT_CODE = "UNCLASSIFIED"
VERIFICATION_MARKER = "CRMKB-DEMO-7a0e5540"


def fetch_company() -> dict:
    """Fetch companies from Twenty and select the exact known record."""
    if not TWENTY_API_KEY:
        raise RuntimeError("TWENTY_API_KEY is missing from the project .env")

    response = requests.get(
        f"{TWENTY_BASE_URL}/rest/companies",
        params={"order_by": "name", "limit": 100},
        headers={"Authorization": f"Bearer {TWENTY_API_KEY}"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    companies = payload.get("data", {}).get("companies", [])

    for company in companies:
        if company.get("id") == COMPANY_ID:
            return company

    raise RuntimeError(
        f"Company {COMPANY_ID} was not found in the returned page. "
        "No RAG upload was attempted."
    )


def company_to_markdown(company: dict) -> str:
    """Build a text document with source identity and provenance."""
    domain_data = company.get("domainName") or {}
    address = company.get("address") or {}

    domain = domain_data.get("primaryLinkUrl") or "Not provided"
    address_parts = [
        address.get("addressStreet1"),
        address.get("addressStreet2"),
        address.get("addressCity"),
        address.get("addressState"),
        address.get("addressPostcode"),
        address.get("addressCountry"),
    ]
    full_address = ", ".join(
        str(part).strip() for part in address_parts if part and str(part).strip()
    )

    return f"""# CRM Company: {company.get("name", "Unknown")}

## Verification
- Verification marker: {VERIFICATION_MARKER}

## Company details
- Company name: {company.get("name", "Not provided")}
- Domain: {domain}
- Address: {full_address or "Not provided"}

## Source provenance
- Source system: Twenty CRM
- Source object: company
- Source record ID: {company.get("id", "Not provided")}
- Created at: {company.get("createdAt", "Not provided")}
- Last updated at: {company.get("updatedAt", "Not provided")}
- Deleted: {str(company.get("deletedAt") is not None).lower()}

## Sync notes
- This document was generated from the Twenty CRM API.
- CRM record ID is the source identity; it is not a RAG document ID.
"""


def upload_to_rag(markdown: str, filename: str, department_code: str = DEPARTMENT_CODE) -> dict:
    """Authenticate and upload through the existing RAG API."""
    email = input("RAG admin email: ").strip()
    password = getpass.getpass("RAG admin password: ")

    if not email or not password:
        raise RuntimeError("RAG email and password are required.")

    with requests.Session() as session:
        login_response = session.post(
            f"{RAG_BASE_URL}/auth/login",
            data={"username": email, "password": password},
            timeout=30,
        )
        login_response.raise_for_status()

        token = login_response.json().get("access_token")
        if not token:
            raise RuntimeError("RAG login succeeded but no access_token was returned.")

        upload_response = session.post(
            f"{RAG_BASE_URL}/documents/upload",
            params={"department_code": department_code},
            headers={"Authorization": f"Bearer {token}"},
            files={
                "file": (
                    filename,
                    markdown.encode("utf-8"),
                    "text/markdown",
                )
            },
            timeout=120,
        )
        upload_response.raise_for_status()
        return upload_response.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync one verified Twenty CRM company into the existing RAG API."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually upload to RAG. Without this flag, run a dry-run only.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Use the native Twenty MCP client adapter (/mcp) instead of legacy REST.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Company name to look up and sync (defaults to 'Airbnb' for --mcp or 'RAG Integration Demo' for legacy).",
    )
    parser.add_argument(
        "--department",
        type=str,
        default=DEPARTMENT_CODE,
        help="Target department code in RAG (defaults to UNCLASSIFIED).",
    )
    args = parser.parse_args()

    try:
        dept_code = args.department or DEPARTMENT_CODE
        company_target = args.company or ("Airbnb" if args.mcp else COMPANY_NAME)

        if args.mcp:
            import asyncio
            from rag_app.crm.twenty_mcp_client import TwentyMCPClient
            from rag_app.crm.sync_service import normalize_crm_company_to_markdown

            mcp_client = TwentyMCPClient(enabled=True)
            print(f"Fetching company '{company_target}' via Twenty MCP adapter ({mcp_client.mcp_url})...")
            crm_state = asyncio.run(mcp_client.get_live_company_state(company_target))

            if not crm_state.is_connected or not crm_state.company:
                raise RuntimeError(
                    f"Twenty MCP fetch failed for '{company_target}': {crm_state.error_message or 'not found'}"
                )

            markdown, doc_meta = normalize_crm_company_to_markdown(
                company_state=crm_state,
                verification_marker=f"CRMKB-MCP-{crm_state.company.id[:8]}",
            )
            cid = crm_state.company.id
            filename = f"twenty_company_{cid[:8]}.md"
            comp_display = {
                "name": crm_state.company.name,
                "id": crm_state.company.id,
                "domain": crm_state.company.domain,
            }
        else:
            company = fetch_company()

            if company.get("deletedAt") is not None:
                raise RuntimeError("The CRM company is deleted; refusing to upload it.")

            if company.get("name") != COMPANY_NAME:
                raise RuntimeError(
                    f"Expected {COMPANY_NAME!r}, received {company.get('name')!r}."
                )

            markdown = company_to_markdown(company)
            filename = f"twenty_company_{COMPANY_ID[:8]}.md"
            comp_display = {
                "name": company["name"],
                "id": company["id"],
                "domain": (company.get("domainName") or {}).get("primaryLinkUrl"),
            }

        print("\nCompany verified:")
        print(f"  Name: {comp_display['name']}")
        print(f"  CRM ID: {comp_display['id']}")
        print(f"  Domain: {comp_display['domain']}")
        print(f"  RAG department: {dept_code}")
        print(f"  Markdown filename: {filename}")
        print("\n--- Generated Markdown ---")
        print(markdown)
        print("--- End Markdown ---")

        if not args.upload:
            print("\nDRY RUN ONLY. Nothing was uploaded to RAG.")
            print("Review the content before running again with --upload.")
            return 0

        print(
            "\nWARNING: the existing RAG deduplication uses file content hashes. "
            "Because this Markdown may differ from the manually uploaded file, "
            "this upload may create another RAG document."
        )
        confirmation = input(
            "Type UPLOAD to proceed with the RAG upload: "
        ).strip()
        if confirmation != "UPLOAD":
            print("Cancelled. No RAG upload was attempted.")
            return 0

        result = upload_to_rag(markdown, filename, department_code=dept_code)
        print("\nRAG API response:")
        print(result)
        print("\nCheck the response's skipped/document_id fields and RAG logs.")
        return 0

    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        print(f"HTTP request failed (status {status_code}).", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text[:1000], file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Sync failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
