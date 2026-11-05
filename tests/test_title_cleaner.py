import pytest
from rag_app.reports.title_cleaner import derive_clean_title


def test_derive_clean_title_strips_conversational_prefixes():
    cases = [
        ("Can you please give me a summary of Q3 revenue?", "Q3 Revenue"),
        ("Please export as docx the employee travel policies", "Employee Travel Policies"),
        ("Could you prepare a report on executive compensation in pdf format?", "Executive Compensation"),
        ("I want to see the 2026 platform architecture roadmap", "2026 Platform Architecture Roadmap"),
        ("Summarize the customer churn risk factors", "Customer Churn Risk Factors"),
        ("Generate a report for Q1 financial results", "Q1 Financial Results"),
    ]
    for raw_prompt, expected_title in cases:
        actual = derive_clean_title(raw_prompt)
        assert actual == expected_title, f"Failed for '{raw_prompt}': got '{actual}', expected '{expected_title}'"


def test_derive_clean_title_sanitizes_illegal_chars():
    prompt = "Can you export: Revenue / Costs * 2026 <Confidential> | Draft?"
    title = derive_clean_title(prompt)
    for bad_char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        assert bad_char not in title
    assert "Revenue Costs 2026 Confidential Draft" in title


def test_derive_clean_title_fallbacks():
    assert derive_clean_title("") == "Report"
    assert derive_clean_title("   ") == "Report"
    assert derive_clean_title("please export this") == "Report"
    assert derive_clean_title("can you download as pdf?") == "Report"

