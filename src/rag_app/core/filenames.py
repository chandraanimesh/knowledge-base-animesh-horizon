import re
from typing import Optional

# Uploaded documents in ragdemo1 are prefixed with a 32-hex character UUID and underscore:
# e.g. "e8c1715305764a119183b8820bd8f58d_ov2_implementation_plan.md"
_UPLOAD_SAFE_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}_(.+)$", re.IGNORECASE)


def derive_original_filename(file_name: Optional[str], file_path: Optional[str] = None) -> Optional[str]:
    """
    Derives the genuine original filename from document metadata.

    The application's `upload_document` endpoint stores uploaded files as
    `uploaded_docs/{uuid4().hex}_{original_name}` and assigns `path.name` to `document.file_name`.
    If the document originated from storage (`uploaded_docs`) and matches the strict 32-hex UUID prefix,
    the original filename given by the user is extracted reliably.

    If the filename does not match or was not stored in `uploaded_docs`, the input `file_name`
    is returned unaltered without silent transformations.
    """
    if not file_name:
        return file_name

    # If file_path is available, verify that this is an uploaded document storage path
    if file_path:
        norm_path = file_path.replace("\\", "/")
        if "uploaded_docs/" not in norm_path and not norm_path.startswith("uploaded_docs"):
            return file_name

    match = _UPLOAD_SAFE_NAME_PATTERN.match(file_name)
    if match:
        return match.group(1)

    return file_name
