

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
    const [message, setMessage] = useState('');
    const [file, setFile] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [emotion, setEmotion] = useState(''); // To store detected emotion
  const [running, setRunning] = useState(false); // To track if detection is running
    const handleSubmit = async () => {
        if (!message.trim() && !file) {
            alert("Please provide a message or upload an image.");
            return;
        }

        try {
            if (file && !message.trim()) {
                await handleImageSubmit();
            } else if (!file && message.trim()) {
                await handleTextSubmit();
            } else if (file && message.trim()) {
                await handleImageAndTextSubmit();
            }
        } catch (error) {
            console.error("Error during submission:", error);
        }
    };

    const handleTextSubmit = async () => {
        const newMessage = { text: message, sender: 'user' };
        setChatHistory([...chatHistory, newMessage]);
        setMessage('');

        try {
            const res = await axios.post('http://127.0.0.1:5000/predict', { statement: message });
            const botResponse = { text: res.data.status || 'No response from server.', sender: 'bot' };
            setChatHistory((prevChat) => [...prevChat, botResponse]);
        } catch (err) {
            console.error("Text submission error:", err);
            const errorResponse = { text: 'Error occurred while communicating with the server for text input.', sender: 'bot' };
            setChatHistory((prevChat) => [...prevChat, errorResponse]);
        }
    };

    const handleImageSubmit = async () => {
        if (!file.type.startsWith('image/')) {
            alert("Please upload a valid image file.");
            setFile(null);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        const newMessage = { text: 'Image Uploaded', sender: 'user' };
        setChatHistory([...chatHistory, newMessage]);
        setFile(null);

        try {
            const res = await axios.post('http://127.0.0.1:5000/predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            const extractedTextResponse = { text: res.data.extracted_text || 'No text extracted from the image.', sender: 'bot' };
            const botResponse = { text: res.data.status || 'No additional insights provided by the server.', sender: 'bot' };

            setChatHistory((prevChat) => [...prevChat, extractedTextResponse, botResponse]);
        } catch (err) {
            console.error("Image submission error:", err);
            const errorResponse = { text: 'Error occurred while communicating with the server for image input.', sender: 'bot' };
            setChatHistory((prevChat) => [...prevChat, errorResponse]);
        }
    };

    const handleImageAndTextSubmit = async () => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('statement', message);

        const newMessage = { text: `Message: "${message}" and Image Uploaded`, sender: 'user' };
        setChatHistory([...chatHistory, newMessage]);
        setMessage('');
        setFile(null);

        try {
            const res = await axios.post('http://127.0.0.1:5000/predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            const extractedTextResponse = { text: res.data.extracted_text || 'No text extracted from the image.', sender: 'bot' };
            const botResponse = { text: res.data.status || 'No additional insights provided by the server.', sender: 'bot' };

            setChatHistory((prevChat) => [...prevChat, extractedTextResponse, botResponse]);
        } catch (err) {
            console.error("Combined input submission error:", err);
            const errorResponse = { text: 'Error occurred while communicating with the server for combined input.', sender: 'bot' };
            setChatHistory((prevChat) => [...prevChat, errorResponse]);
        }
    };

    const startDetection = async () => {
        setRunning(true);
    const eventSource = new EventSource('http://127.0.0.1:5000/detect_emotion');

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.emotion) {
        setEmotion(data.emotion);
      } else if (data.message) {
        setEmotion(data.message);
        eventSource.close(); // Stop listening when the server indicates the user exited
        setRunning(false);
      } else if (data.error) {
        setEmotion(data.error);
        eventSource.close();
        setRunning(false);
      }
    };
    eventSource.onerror = () => {
        setEmotion('An error occurred.');
        eventSource.close();
        setRunning(false);
      };
    };


    return (
        <div className="chat-container">
            <div className="chat-box">
                <div className="messages">
                    {chatHistory.map((msg, index) => (
                        <div key={index} className={`message ${msg.sender}`}>
                            <p>{msg.text}</p>
                        </div>
                    ))}
                </div>
                <div className="input-container">
                    <input
                        type="text"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder="Type a message"
                    />
                </div>
                <div className="image-upload">
                    <input
                        type="file"
                        onChange={(e) => setFile(e.target.files[0])}
                    />
                </div>
                <button onClick={handleSubmit}>Send</button>
            </div>
            <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Emotion Detection</h1>
      <p>{emotion}</p>
      {!running && (
        <button onClick={startDetection} style={{ padding: '10px 20px', fontSize: '16px' }}>
          Start Emotion Detection
        </button>
      )}
    </div>
        </div>
    );
};

export default App;
