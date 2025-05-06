
# Realtime Textile Sorting System

This folder contains code to run the real-time classification and pick-and-place sorting system across two machines.

---

##  System Overview

### Windows (NIR Sensor + Classifier)
- **label_data.py**: Predicts textile class from a `.csv` NIR scan file using a pretrained neural network.
- **client.py**: Sends the predicted class label (0–5) to the Ubuntu machine via TCP socket.

### Ubuntu (UR5 Arm + Camera)
- **main.py**: Receives the label from Windows, detects objects using a Realsense camera + YOLO, and commands the UR5 arm to sort the textile based on class.

---

##  Data Flow

1. Windows PC monitors a folder for a new NIR `.csv` scan.
2. Once a file is found, it is classified by `label_data.py`.
3. The class label is sent via `client.py` to the Ubuntu system.
4. Ubuntu’s `main.py` receives the label and waits for a YOLO detection.
5. If an object is centered in the frame, the UR5 performs a pick-and-place to the corresponding bin.

---

##  Folder Structure

```text
realtime/
├── windows/
│   ├── label_data.py  # Preprocess + classify NIR scan
│   └── client.py      # Send class label over socket
├── UR5_ubuntu/
│   └── main.py        # Realsense + YOLO + UR5 sorting logic
```

---

##  How to Use

### Windows Machine
1. Ensure your `.csv` NIR scan files appear in the watched folder defined in `client.py`:
   ```python
   WATCH_FOLDER = r'C:\Users\amirs\Desktop\MAS500\MAS500\NIR\rt\monitor'
   ```
2. Start the client:
   ```bash
   python client.py
   ```

### Ubuntu Machine
1. Make sure Realsense SDK, and UR5 network control are set up.
2. Launch the script:
   ```bash
   python3 main.py
   ```

---

##  Notes

- Labels are expected as integers 0–5:
  | Label | Class                 |
  |-------|------------------------|
  | 0     | Cotton-White          |
  | 1     | Cotton-Black          |
  | 2     | Cotton-Other          |
  | 3     | Polyester-White       |
  | 4     | Polyester-Black       |
  | 5     | Polyester-Other       |

- Be sure to adjust IPs in both scripts to match your network setup.
- Both machines must be on the **same subnet** for socket communication to work.

---

##  Related
- Model used in `label_data.py` is found in: `SO1/SO1.3/model/multi_output_model_v5_20.h5`

