import sys, uuid
from pathlib import Path
sys.path.insert(0, 'src')
from rag_app.ingestion.loader import load_document
from rag_app.ingestion.chunker import chunk_elements
from rag_app.ingestion.hierarchy import HierarchyBuilder

path = Path('uploaded_docs')
docx_files = list(path.rglob('*.docx'))
print(f'DOCX files: {len(docx_files)}')
md_files = list(Path('sample_docs').glob('*.md'))
print(f'Markdown files: {len(md_files)}')

# Use the git_code_review_guidelines.md as the test document
p = md_files[0]
print('Processing:', p.name)
elements = load_document(p)
chunks = chunk_elements(elements)
print(f'Elements: {len(elements)}, Chunks: {len(chunks)}')
builder = HierarchyBuilder(document_id=uuid.uuid4(), file_name=p.stem)
result = builder.build(chunks)
print(f'Total nodes: {len(result.all_nodes)}')
for n in result.all_nodes:
    indent = '  ' * n.level
    print(f'{indent}[{n.ordinal_path}] {n.node_type} title={n.title[:40]!r} ({len(n.chunk_indices)} chunks)')
    if n.l0_abstract:
        print(f'{indent}   L0: {n.l0_abstract[:90]!r}')
