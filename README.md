# DeepFace UI

DeepFace UI is a web application for facial recognition and analysis built with DeepFace. It offers an intuitive interface to upload images, automatically extract faces, and perform state-of-the-art facial verification and attribute analysis.

![alt text](https://imgur.com/xx2y6Wi.png)

---

## Requirements 📦
- **Python 3.x**  
- **Dependencies:**
  - Flask
  - OpenCV (`opencv-python`)
  - NumPy
  - Requests
  - DeepFace
  - Werkzeug

> **Tip:** Use a virtual environment to manage dependencies efficiently.

---

## Installation 🛠️

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/deepface-ui.git
   cd deepface-ui
   ```

2. **Create and Activate a Virtual Environment:**
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the Dependencies:**
   ```bash
   pip install flask opencv-python numpy requests deepface werkzeug
   ```

---

## Usage 🚀

Start the application with:

```bash
python app.py
```

Then, open your browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to begin using DeepFace UI.

---

## Configuration ⚙️

DeepFace UI is preconfigured with industry-standard settings, but you can easily adjust parameters such as the DeepFace model, distance metric, and detector backend to tailor the analysis to your needs.

---

## Contributing 🤝

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Commit with clear, concise messages.
4. Open a pull request for review.
