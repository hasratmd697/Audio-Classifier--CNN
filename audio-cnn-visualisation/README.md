# CNN Audio Visualizer

A web application for visualizing audio classification predictions using a Convolutional Neural Network (CNN). Upload a WAV file to see the model's predictions, spectrograms, waveforms, and feature maps.

## Tech Stack

- [Next.js](https://nextjs.org) - React framework
- [Tailwind CSS](https://tailwindcss.com) - Styling
- [Radix UI](https://radix-ui.com) - UI components
- [Modal](https://modal.com) - Serverless backend for ML inference

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Modal account (for backend inference - optional for frontend-only testing)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hasratmd697/Audio-Classifier--CNN.git
cd Audio-Classifier--CNN/audio-cnn-visualisation
```

### 2. Install dependencies

```bash
npm install
```

### 3. Run the development server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the app.

### 4. Build for production

```bash
npm run build
npm run start
```

## Backend Setup (Modal)

The app uses a Modal serverless endpoint for audio classification. To deploy your own backend:

### 1. Install Modal CLI (from project root)

```bash
cd ..
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token new
```

### 3. Deploy the inference endpoint

```bash
modal deploy main.py
```

The endpoint will be available at `https://<your-username>--audio-cnn-inference-audioclassifier-inference.modal.run`

## Deployment

### Render (Free Tier)

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Create a new Web Service
4. Set **Root Directory** to: `audio-cnn-visualisation`
5. Configure:
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm run start`
   - **Plan:** Free

### Vercel

```bash
npm run build
vercel deploy
```

## Project Structure

```
audio-cnn-visualisation/
├── src/
│   ├── app/           # Next.js app router pages
│   ├── components/    # React components
│   ├── lib/           # Utility functions
│   └── styles/        # Global CSS
├── public/            # Static assets
├── render.yaml        # Render deployment config
└── next.config.js     # Next.js configuration
```

## Features

- 🎵 Upload WAV audio files for classification
- 📊 View top predictions with confidence scores
- 📈 Visualize input spectrograms and audio waveforms
- 🧠 Explore CNN layer feature maps
- 🎨 Beautiful, responsive UI with Tailwind CSS

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Modal Documentation](https://modal.com/docs)
