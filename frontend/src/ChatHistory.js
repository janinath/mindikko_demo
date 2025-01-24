import React, { useState, useEffect } from "react";
import axios from "axios";

const ChatHistory = () => {
    const [history, setHistory] = useState({});
    const [selectedChat, setSelectedChat] = useState(null);

    useEffect(() => {
        axios.get("http://127.0.0.1:5000/messages")
            .then(response => {
                setHistory(response.data);
            })
            .catch(error => {
                console.error("Error fetching chat history:", error);
            });
    }, []);

    const handleChatClick = (chat) => {
        setSelectedChat(chat.full_chat);
    };

    return (
        <div className="chat-history-container">
            <div className="history-sidebar">
                {Object.keys(history).map((category) => (
                    <div key={category} className="history-group">
                        <h3>{category}</h3>
                        {history[category].map((chat, index) => (
                            <div
                                key={index}
                                className="chat-preview"
                                onClick={() => handleChatClick(chat)}
                            >
                                {chat.preview}
                            </div>
                        ))}
                    </div>
                ))}
            </div>
            <div className="chat-details">
                {selectedChat ? (
                    <div>
                        <h3>Full Conversation</h3>
                        <p><strong>User:</strong> {selectedChat.user_message}</p>
                        <p><strong>Bot:</strong> {selectedChat.bot_response}</p>
                        <p><strong>Date:</strong> {selectedChat.date}</p>
                    </div>
                ) : (
                    <p>Select a chat to view details</p>
                )}
            </div>
        </div>
    );
};

export default ChatHistory;
