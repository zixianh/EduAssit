FILES:

data_collection.py: Collect text data from all PDFs under the folder "PDFS", store into "mydatabse.db".

image_process.py: Collect image data from PDFS/ML.pdf, store into "images.db" with "id, page_number, index_number, image_path, description_text". The extracted images are stored in the folder "Images", referring to the "image_path". (Note: I use ML.pdf instead of ISLR.pdf since the difficulty to extract SVG vector graphics in ISLR)

model.py: Build the language model, return the text and image answers. Compute the similarity score between the user question and description text of images in the databse. If similarity score > 0.8, then return path of the most relevant image. Else, show no image.


**Some Limitations**: 
1. Space: Currently storing images locally, to be improved later. 
2. Algorithms could be improved: If calculating the similarity between bot_response and the description text, the results are terrible. Now only the similarity between user_question and image text is calculated. In this case, the accuracy of the algorithm is poor for some complex problems. But when presenting, we can ask conceptual questions such as "what is classification" and "what is a neural network" , the code performs well.



How to run the code:
1. python3 data_collection.py
2. python3 image_process.py
3. python3 model.py
If you encounter the error: "ImportError: cannot import name 'triu' from 'scipy.linalg'", please install scipy==1.10.1

Libraries:
Package                  Version
------------------------ -----------
aiofiles                 23.2.1
aiohttp                  3.9.3
aiosignal                1.3.1
annotated-types          0.6.0
anyio                    4.3.0
attrs                    23.2.0
certifi                  2024.2.2
cffi                     1.16.0
chardet                  5.2.0
charset-normalizer       3.3.2
ci-info                  0.3.0
click                    8.1.7
configobj                5.0.8
configparser             6.0.1
contourpy                1.2.1
cryptography             42.0.5
cssselect                1.2.0
cycler                   0.12.1
dataclasses-json         0.6.4
distro                   1.9.0
etelemetry               0.3.1
faiss-cpu                1.8.0
filelock                 3.13.3
fitz                     0.0.1.dev2
fonttools                4.51.0
frontend                 0.0.3
frozenlist               1.4.1
fsspec                   2024.3.1
gensim                   4.3.2
h11                      0.14.0
httpcore                 1.0.5
httplib2                 0.22.0
httpx                    0.27.0
huggingface-hub          0.22.2
idna                     3.6
isodate                  0.6.1
itsdangerous             2.1.2
Jinja2                   3.1.3
joblib                   1.3.2
jsonpatch                1.33
jsonpointer              2.4
kiwisolver               1.4.5
langchain                0.1.14
langchain-community      0.0.31
langchain-core           0.1.40
langchain-openai         0.1.1
langchain-text-splitters 0.0.1
langsmith                0.1.40
looseversion             1.3.0
lxml                     5.2.1
MarkupSafe               2.1.5
marshmallow              3.21.1
matplotlib               3.8.4
mpmath                   1.3.0
multidict                6.0.5
mypy-extensions          1.0.0
networkx                 3.2.1
nibabel                  5.2.1
nipype                   1.8.6
nltk                     3.8.1
numpy                    1.26.4
openai                   1.16.2
opencv-python            4.9.0.80
orjson                   3.10.0
packaging                23.2
pandas                   2.2.1
pathlib                  1.0.1
pdf2image                1.17.0
pdfminer.six             20231228
pdfquery                 0.4.3
pillow                   10.3.0
pip                      24.0
prov                     2.0.0
pycparser                2.22
pydantic                 2.6.4
pydantic_core            2.16.3
pydot                    2.0.0
PyMuPDF                  1.24.1
PyMuPDFb                 1.24.1
pyparsing                3.1.2
pypdf                    4.1.0
PyPDF2                   3.0.1
pyquery                  2.0.0
pytesseract              0.3.10
python-dateutil          2.9.0.post0
pytz                     2024.1
pyxnat                   1.6.2
PyYAML                   6.0.1
rdflib                   7.0.0
regex                    2023.12.25
requests                 2.31.0
roman                    4.1
safetensors              0.4.2
scikit-learn             1.4.1.post1
scipy                    1.10.1
sentence-transformers    2.6.1
setuptools               69.1.1
simplejson               3.19.2
six                      1.16.0
smart-open               7.0.4
sniffio                  1.3.1
SQLAlchemy               2.0.29
starlette                0.37.2
sympy                    1.12
tenacity                 8.2.3
threadpoolctl            3.4.0
tiktoken                 0.6.0
tokenizers               0.15.2
torch                    2.2.2
torchvision              0.17.2
tqdm                     4.66.2
traits                   6.3.2
transformers             4.39.3
typing_extensions        4.10.0
typing-inspect           0.9.0
tzdata                   2024.1
urllib3                  2.2.1
uvicorn                  0.29.0
Wand                     0.6.13
wheel                    0.42.0
wrapt                    1.16.0
yarl                     1.9.4