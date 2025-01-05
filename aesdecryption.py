import joblib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import json

# AES Decryption setup
def aes_decrypt(encrypted_data, iv, key):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()

    # Decrypt the data
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove padding to retrieve the original data
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()

    return data.decode()

# Paths to the files
hashed_data_path = "hashed_normal_data.pkl"
key_path = "aes_key.key"

# Load the AES key
with open(key_path, "rb") as key_file:
    aes_key = key_file.read()

# Load the encrypted dataset
hashed_data = joblib.load(hashed_data_path)

decrypted_rows = []

# Decrypt each row
for encrypted_row, iv in hashed_data:
    decrypted_row_json = aes_decrypt(encrypted_row, iv, aes_key)  # Decrypt JSON string
    decrypted_row = json.loads(decrypted_row_json)  # Convert JSON string back to dictionary
    decrypted_rows.append(decrypted_row)

# Convert the decrypted rows back to a DataFrame
import pandas as pd
decrypted_df = pd.DataFrame(decrypted_rows)

# Save or inspect the decrypted DataFrame
print("Decrypted DataFrame:")
print(decrypted_df.head())

decrypted_data_path = "decrypted_normal_data.csv"
decrypted_df.to_csv(decrypted_data_path, index=False)
print(f"Decrypted data saved to {decrypted_data_path}")
