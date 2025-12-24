# Audio CNN

<img width="1920" height="1001" alt="Screenshot 2025-07-14 094831" src="https://github.com/user-attachments/assets/d8ac076b-6cc7-4cc8-88b0-48fb5b1ed0c0" />

## Overview

In this project, I trained and deployed an audio classification Convolutional Neural Network (CNN) from scratch using PyTorch. The model is capable of classifying environmental sounds, such as dog barks and bird chirps, from audio files. I implemented advanced techniques including Residual Networks (ResNet), data augmentation through audio mixing, and Mel Spectrogram transformations to create a robust training pipeline. To make the model accessible and interactive, I built a full-stack web dashboard using Next.js, React, and Tailwind CSS (following the T3 Stack architecture). The dashboard allows users to upload audio files and visualize how the model processes and interprets them through its internal layers. All tools and services used in this project are open-source and freely available.

## Features:

- 🧠 Deep Audio CNN for sound classification
- 🧱 ResNet-style architecture with residual blocks
- 🎼 Mel Spectrogram audio-to-image conversion
- 🎛️ Data augmentation with Mixup & Time/Frequency Masking
- ⚡ Serverless GPU inference with Modal
- 📊 Interactive Next.js & React dashboard
- 👁️ Visualization of internal CNN feature maps
- 📈 Real-time audio classification with confidence scores
- 🌊 Waveform and Spectrogram visualization
- 🚀 FastAPI inference endpoint
- ⚙️ Optimized training with AdamW & OneCycleLR scheduler
- 📈 TensorBoard integration for training analysis
- 🛡️ Batch Normalization for stable & fast training
- 🎨 Modern UI with Tailwind CSS & Shadcn UI
- ✅ Pydantic data validation for robust API requests

## Prerequisites

- Python 3.12+
- Node.js 18+
- npm package manager
- Modal account (for serverless GPU inference)

## Setup

Follow these steps to install and set up the project.

### 1. Clone the Repository

```bash
git clone https://github.com/hasratmd697/Audio-Classifier--CNN.git
cd Audio-Classifier--CNN
```

### 2. Create Python Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Running Locally

### Backend (Modal Inference Server)

#### Option 1: Deploy to Modal (Recommended)

```bash
modal token new    # Authenticate with Modal (first time only)
modal deploy main.py
```

The endpoint will be available at:
`https://<your-username>--audio-cnn-inference-audioclassifier-inference.modal.run`

#### Option 2: Run Locally with Modal

```bash
modal serve main.py
```

### Frontend (Next.js Dashboard)

Navigate to the frontend directory:

```bash
cd audio-cnn-visualisation
```

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm run start
```

## Training the Model

To train the CNN model on the ESC-50 dataset:

```bash
modal run train.py
```

## Project Structure

```
Audio-Classifier--CNN/
├── main.py                    # Modal inference endpoint
├── train.py                   # Model training script
├── model.py                   # CNN architecture definition
├── requirements.txt           # Python dependencies
├── audio-cnn-visualisation/   # Next.js frontend
│   ├── src/
│   │   ├── app/               # Next.js app router
│   │   ├── components/        # React components
│   │   └── styles/            # Global CSS
│   ├── package.json
│   └── render.yaml            # Render deployment config
└── tensorboard_logs/          # Training logs
```

## Deployment

### Frontend on Render (Free Tier)

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Create a new Web Service
4. Set **Root Directory** to: `audio-cnn-visualisation`
5. Configure:
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm run start`
   - **Plan:** Free

### Backend on Modal

```bash
modal deploy main.py
```

## Technologies Used

- **PyTorch** - Deep learning framework
- **Modal** - Serverless GPU compute
- **Next.js** - React framework
- **Tailwind CSS** - Styling
- **Radix UI** - UI components
- **FastAPI** - API framework
- **Librosa** - Audio processing
