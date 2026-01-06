\# Acoustic Tesseract



\*\*Acoustic Tesseract\*\* is a research-oriented, accessibility-focused data exploration system that enables users—especially those with visual impairments—to explore high-dimensional datasets using \*\*spatial audio\*\* and optional visual feedback. The project converts N-dimensional data into a navigable 3D space and maps data proximity and direction to sound.



---



\## 📌 Project Motivation



Traditional data visualization techniques rely heavily on visual perception, making them inaccessible to visually impaired users. Screen readers linearize data and fail to convey spatial relationships such as clusters, density, or outliers.



\*\*Acoustic Tesseract\*\* addresses this gap by:



\* Preserving the \*topological structure\* of data

\* Allowing users to \*navigate data using sound\*

\* Providing a parallel visualizer for validation and demonstrations



---



\## 🧠 Core Concept



1\. \*\*High‑Dimensional Input\*\* (e.g., Iris dataset – 4D)

2\. \*\*Dimensionality Reduction\*\* using \*\*t‑SNE\*\* → 3D space

3\. \*\*Spatial Indexing\*\* using \*\*KD‑Tree\*\* for fast nearest‑neighbor queries

4\. \*\*Real‑Time Navigation\*\* via keyboard input

5\. \*\*Spatial Audio Feedback\*\* based on distance and direction



The closer a data point is, the louder it sounds. Left/right positioning is conveyed using stereo panning.



---



\## 🗂️ Project Structure



```

DS Project 1/

│

├── acoustic\_tesseract\_poc.py   # Main interactive application

├── mapper.py                  # Standalone t‑SNE mapper \& visual proof

├── sound.wav                  # Base sound for spatial audio

├── Project Report 1.docx      # Detailed academic report

├── venv/                      # Python virtual environment

└── README.md                  # Project documentation

```



---



\## 🚀 Features



\### 🎧 Spatial Audio Navigation



\* Stereo audio output (left/right ear separation)

\* Distance‑based volume attenuation

\* Direction‑aware panning

\* Continuous proximity feedback



\### 🧭 Interactive Exploration



\* Keyboard navigation (W/A/S/D)

\* Direction indicator and trail

\* Speed boost and reset controls



\### 📊 Visual Support (Optional)



\* 2D top‑down visualization of 3D space

\* Cluster‑colored data points

\* Mini‑radar for spatial awareness



\### ⚡ Performance Optimizations



\* KD‑Tree for O(log n) nearest‑neighbor queries

\* Cached coordinate mapping

\* Particle and trail limits to control memory usage



---



\## 🛠️ Technologies Used



\* \*\*Python 3\*\* – Core language

\* \*\*NumPy\*\* – Vector and matrix operations

\* \*\*scikit‑learn\*\* – t‑SNE, datasets

\* \*\*SciPy\*\* – KDTree spatial indexing

\* \*\*Pygame\*\* – Real‑time loop, audio, and rendering

\* \*\*Matplotlib\*\* – 3D visualization (mapper.py)



---



\## ⚙️ Installation \& Setup



\### 1️⃣ Create Virtual Environment (Recommended)



```bash

python -m venv venv

venv\\Scripts\\activate   # Windows

```



\### 2️⃣ Install Dependencies



```bash

pip install numpy scipy scikit-learn pygame matplotlib

```



\### 3️⃣ Run the Application



```bash

python acoustic\_tesseract\_poc.py

```



You will be prompted to choose a dataset:



\* \*\*I\*\* – Iris dataset

\* \*\*R\*\* – Random synthetic dataset

\* \*\*D\*\* – Digits dataset



---



\## 🎮 Controls



| Key            | Action                  |

| -------------- | ----------------------- |

| W / S          | Move forward / backward |

| A / D or ← / → | Rotate left / right     |

| ↑ Arrow        | Speed boost             |

| SPACE          | Reset position          |

| H              | Toggle help overlay     |

| Q / ESC        | Quit application        |



---



\## 🧪 mapper.py (Validation Script)



`mapper.py` is a standalone script that:



\* Loads the Iris dataset

\* Applies 3D t‑SNE

\* Displays a 3D scatter plot using Matplotlib



This script serves as \*\*visual proof\*\* that clustering is preserved before integrating audio navigation.



Run it using:



```bash

python mapper.py

```



---



\## 📈 Results



\* Users can locate and follow clusters using sound alone

\* Volume and stereo separation provide intuitive spatial cues

\* Visualizer confirms direct correlation between sound and position



This validates spatial audio as a viable alternative to visual data exploration.



---



\## 🔮 Future Enhancements



\* 🎧 \*\*True 3D Audio (HRTF)\*\* for vertical sound perception

\* 🔔 \*\*Cluster‑specific sounds\*\* for better differentiation

\* 📂 \*\*CSV upload support\*\* for custom datasets

\* 👥 \*\*Usability testing\*\* with visually impaired users

\* 🌐 \*\*Web‑based version\*\* using WebAudio + WebGL



---



\## 👨‍💻 Authors



\* \*\*Swayam\*\*

\* \*\*Deep\*\*

\* \*\*Suyog\*\*

\* \*\*Yug\*\*



\*Academic Project – November 2025\*



---



\## 📄 License



This project is intended for \*\*academic and research purposes\*\*. Please cite appropriately if used in publications or derivative works.



---



\## ⭐ Acknowledgements



\* scikit‑learn team for dimensionality reduction tools

\* Pygame community for audio and real‑time rendering support

\* Accessibility research initiatives inspiring inclusive design



---



\*\*Acoustic Tesseract\*\* — \*Exploring data through sound.\*

....



