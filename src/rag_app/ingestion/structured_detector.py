import csv
from io import StringIO


def looks_like_csv_text(
    text: str,
) -> bool:

    sample = text[:5000]
    if not sample.strip():
        return False
    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|",
        )
        reader = csv.reader(
            StringIO(sample),
            dialect,
        )
        rows = [
            row
            for row in reader
            if any(
                cell.strip()
                for cell in row
            )
        ]

    except csv.Error:
        return False
    if len(rows) < 2:
        return False
    column_count = len(rows[0])
    if column_count < 2:
        return False
    checked_rows = rows[:10]
    return all(
        len(row) == column_count
        for row in checked_rows
    )