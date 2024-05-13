import fitz
import re
import os
import math
import sqlite3
import torch
import torchvision.transforms as transforms
from PIL import Image
from sentence_transformers import SentenceTransformer


def create_database(database_path):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Image_Text
                 (id INTEGER PRIMARY KEY,
                 page_number INTEGER,
                 index_number INTEGER,
                 image_path TEXT,
                 description_text TEXT)''')
    conn.commit()
    conn.close()

def insert_into_database(database_path, page_number, index_number, image_path, description_text):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute('''INSERT INTO Image_Text (page_number, index_number, image_path, description_text)
                 VALUES (?, ?, ?, ?)''', (page_number, index_number, image_path, description_text))
    conn.commit()
    conn.close()

def extract_images_and_descriptions(pdf_path, output_folder, database_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create database if it doesn't exist
    create_database(database_path)

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    images_with_descriptions = []

    # Iterate through each page
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        
        # Extract images from the page
        images = page.get_images(full=True)
        
        # Iterate through each image on the page
        for img_index, img_info in enumerate(images):
            # img_info: (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer)
            xref = img_info[0]
            # print("img_info", img_info)
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Write the image bytes to a file
            image_file_path = f"{output_folder}/image_page_{page_number+1}_index_{img_index}.{image_ext}"
            with open(image_file_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            # Get the bottom edge of the bounding box
            bbox = page.get_image_bbox(img_info)
            info = page.get_text("blocks")
    
            # extracts the text along with its layout information
            # Extract text blocks
            text_blocks = page.get_text("blocks")
            text = ""

            # Iterate through text blocks
            for b in text_blocks:
                # Concatenate the text of this block to the page text
                text += b[4] + "\n"


            # Search for the description text starting with "FIGURE" after the image
            text_pattern = re.compile(r'FIGURE\s*[\d\.]+(?:\s+.*?)*?\s+(.*?)\n\n', re.DOTALL)
            matches = text_pattern.findall(text)
            if matches:
                for index, match in enumerate(matches):
                    description = match.strip().replace('\n', ' ')
                    
                    # description_file_path = f"{output_folder}/description_page_{page_number+1}_index_{index}.txt"
                    if index == img_index:
                        # Store embeddings in the database
                        # print("image_file_path", image_file_path)
                        # print("text_embedding", text_embedding)
                        insert_into_database(database_path, page_number + 1, img_index, image_file_path, description)
                        
                    # with open(description_file_path, "w") as description_file:
                    #     description_file.write(description)
                        

    # Close the PDF
    pdf_document.close()


# Example usage
pdf_file_path = "PDFS/ML.pdf"
output_directory = "Images"
database_path = "images.db"
extract_images_and_descriptions(pdf_file_path, output_directory, database_path)
