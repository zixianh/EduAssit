from pypdf import PdfReader
import os
import sqlite3

def create_connection(db_file):
    try:
        connection = sqlite3.connect(db_file)
        return connection
    except sqlite3.Error as e:
        print(e)
        return None
    
def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS information (
                name TEXT,
                content TEXT
            );
        ''')
        connection.commit()
        # print("Table created or already exists.")
    except sqlite3.Error as e:
        print(e)
        
def insert_data(connection, name, content):
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO information (name, content) VALUES (?, ?);", (name, content))
        connection.commit()
        # print("Data inserted.")
    except sqlite3.Error as e:
        print(e)
        
def query_data(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT name, content FROM information;")
        rows = cursor.fetchall()
        dicttemp = {}
        for row in rows:
            name, content = row
            dicttemp[name] = content
        return dicttemp
    except sqlite3.Error as e:
        print(e)
 
 
def get_pdf_file_paths(folder_path):
    pdf_file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_file_paths.append(os.path.join(root, file))
    return pdf_file_paths

def exact_content(filename):
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

 
def main():
    db_file = "mydatabase.db"
    connection = create_connection(db_file)
    if connection:
        create_table(connection)
    folder_path = r'./PDFS'
    pdf_paths = get_pdf_file_paths(folder_path)
    for item in pdf_paths:
        content = exact_content(item)
        insert_data(connection,item,content)
        

if __name__ == "__main__":
    main()