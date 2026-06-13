from langchain_docling.loader import DoclingLoader

FILE_PATH = "insurance.pdf"

loader = DoclingLoader(file_path=FILE_PATH)

# Load all documents
documents = loader.load()
print("Length of documents : ",len(documents))
# # For large datasets, lazily load documents
# for document in loader.lazy_load():
#     print(document)

for d in documents[:3]:
    print(f"- {d.page_content=}")