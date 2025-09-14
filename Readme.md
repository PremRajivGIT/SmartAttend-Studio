

# Yo! How's your Day Going! 😎

## Anyways, refer to these instructions to run this website/server/studio :)



### 1. Install WSL (Important!)

Follow these steps:

```bash
wsl --install




### 2. Clone this repository (hint: use git clone)

```bash
git clone https://github.com/<your-username>/SmartAttend-Studio.git
cd SmartAttend-Studio
```



### 3. Install requirements.txt

(TIP: Create a venv or rewrite source packages 💀 xd)
***Note: Use python3.11 for better compatibility***

```bash
# install python 3.11
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv -y

# create venv
python3.11 -m venv venv
source venv/bin/activate

# install requirements
pip install --upgrade pip
pip install -r requirements.txt
```



### 4. You need to Download these files and move them to the project directory (tbc)

* Face detection model: `yolov8m_200e.pt` → put inside `models/`

---

### 5. Run app.py

```bash
flask run
```

Visit 👉 [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.



# 📚 Project Info

## 🚀 Features

* 📷 **YOLOv8-based face detection** (`yolov8m_200e.pt`)
* 🧠 **Face embeddings** with TensorFlow/Keras FaceNet
* 🔄 **Real-time attendance marking**
* 🏫 **Multi-classroom support**
* 🗄 **Database integration** with SQLAlchemy
* 📊 **Model training & export** (ONNX / TFLite)
* ⚡ Optimized for **GPU acceleration**



## 🖥️ Usage

* Select a class → scan student faces → attendance auto-marked ✅
* Missed students? → re-scan → system merges results.




## 📁 Project Structure

```
SmartAttend-Studio/
│── app.py                # Flask entrypoint
│── requirements.txt       # Python dependencies
│── models/
│   └── yolov8m_200e.pt    # YOLOv8 face detection model
│── static/                # Frontend assets
│── templates/             # HTML templates
│── utils/                 # Helper scripts (training, inference)
│── database/              # SQLAlchemy models & migrations
```



## 🔮 Future Improvements

* [ ] Web UI for training status
* [ ] Real-time webcam attendance
* [ ] Mobile app integration
* [ ] Cloud deployment



## 📝 License

MIT License © 2025 \[Your Name]

```

---

```
