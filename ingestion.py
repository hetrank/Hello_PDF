from ocr_utils import pdf_to_doc

pdf_path = r"C:\MY STUFF\DAIICT\GenAI\Research-Assistant\sample_pdfs\intro_comm_systems_madhow_jan2014b_0.pdf"


docs = pdf_to_doc(pdf_path)

print(len(docs))
print(type(docs[0]))
print(docs[0].page_content)