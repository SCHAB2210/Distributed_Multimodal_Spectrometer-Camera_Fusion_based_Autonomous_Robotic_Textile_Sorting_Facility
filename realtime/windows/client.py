import socket
import time
import glob
import os
from tensorflow.keras.models import load_model
from label_data import predict_label
#from label_data_features import predict_label

# === CONFIG ===
WATCH_FOLDER = r'C:\Users\amirs\Desktop\MAS500\MAS500\NIR\rt\monitor'
os.chdir(WATCH_FOLDER)

host = '192.168.56.101'  # Ubuntu server IP
port = 12345             # Must match the port used by the server
# path = C:\Users\amirs\Desktop\MAS500\MAS500\NIR\results\multi_output_model_combined_v5\multi_output_model_v5_20.h5
model_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\NIR\results\multi_output_model_combined_v5\multi_output_model_v5_20.h5'
model = load_model(model_path, compile=False)
running = True

# === HELPERS ===
def get_latest_csv():
    csv_files = glob.glob("*.csv")
    return csv_files[0] if csv_files else None

def connect_to_server():
    while True:
        try:
            print(f"🔌 Trying to connect to server at {host}:{port}...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            print("✅ Connected to server.")
            return s
        except socket.error as e:
            print(f"⛔ Connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

# === MAIN LOOP ===
def send_prediction():
    while running:
        s = connect_to_server()
        try:
            while True:
                file = get_latest_csv()
                if file:
                    try:
                        label = predict_label(file, model)
                        if label != '':
                            s.sendall(label.encode('utf-8'))
                            print(f"📤 Sent label: {label}")
                            time.sleep(0.2)
                            os.remove(file)
                            print(f"🗑️ Removed: {file}")
                    except PermissionError:
                        print(f"⚠️ File {file} is in use. Retrying...")
                    except Exception as e:
                        print(f"⚠️ Unexpected error processing {file}: {e}")
                time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Socket error during communication: {e}")
            s.close()
            print("🔁 Reconnecting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    try:
        send_prediction()
    except KeyboardInterrupt:
        print("🛑 Interrupted by user. Exiting...")