import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

names = ["shruti","scarlet","bryan"]
usernames = ["shruti","scarlet","bryan"]
passwords = ["abc","abc","abc"]

hashed_passwords = Hasher(passwords).generate()
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
     pickle.dump (hashed_passwords, file)
