import gdown
import os

MODEL_FILE_ID = "1hi8tyZkUwbeGac4hLvKUOANYVKWcmUZj" 
LABEL_MAP_FILE_ID = "1ChVjG3cd8uCkdkacMl5gWeT7pidRU14x"

def download_files():
    if not os.path.exists('transaction_classifier.pth'):
        print("Downloading model file from Google Drive")
        try:
            gdown.download(
                f'https://drive.google.com/uc?id={MODEL_FILE_ID}', 
                'transaction_classifier.pth', 
                quiet=False
            )
            print("Model file downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print("Model file already exists locally")
   
    if not os.path.exists('label_map.pkl'):
        print("Downloading label map from Google Drive")
        try:
            gdown.download(
                f'https://drive.google.com/uc?id={LABEL_MAP_FILE_ID}', 
                'label_map.pkl', 
                quiet=False
            )
            print("Label map downloaded successfully!")
        except Exception as e:
            print(f"Error downloading label map: {e}")
            raise
    else:
        print("Label map already exists locally")
    
    print("\nAll model files ready\n")

if __name__ == "__main__":
    download_files()