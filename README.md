# OpenAI ChatGPT Flask and React Application

This is a full-stack web application built with **Flask** as the backend and **React** as the frontend. The application integrates with OpenAI's GPT-3/ChatGPT to provide conversational AI functionality.

## Features

- **React Frontend**: A modern frontend with a responsive user interface.
- **Flask Backend**: A simple API built using Flask to handle communication between the frontend and OpenAI's API.
- **ChatGPT Integration**: The backend is connected to OpenAI's GPT-3 API to process and respond to user input.

## Installation

### Prerequisites

- Python 3.x
- Node.js (for React)
- npm (for React dependencies)

### Backend Setup (Flask)

1. Clone this repository:
   
   git clone https://github.com/yourusername/openai_chatgpt.git
   cd openai_chatgpt

2. Create and activate a virtual environment:

  python -m venv openai_env
  openai_env\Scripts\activate

3.Set up your .env file for Flask (for example, API keys and configurations):
  API_KEY=your-openai-api-key

4.Run the Flask backend:
  python app.py

The Flask API will be running at http://localhost:5000.


### Frontend Setup (React)
1.Navigate to the frontend/ folder:
  cd frontend

2.Install React dependencies:
  npm install

3.Run the React development server:
  npm start

The React app will be running at http://localhost:3000.


## Usage
Once both the backend and frontend are running, you can open the React app in your browser.
The application will allow you to chat with the AI model powered by OpenAI's GPT-3.











