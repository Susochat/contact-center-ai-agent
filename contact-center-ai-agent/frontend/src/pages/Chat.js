import React, { useState } from 'react';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  
  const sendMessage = async (message) => {
    // Update chat history with user message
    const newMessages = [...messages, { text: message, sender: 'user' }];
    setMessages(newMessages);
    setInputText('');
  
    try {
      const response = await fetch('http://127.0.0.1:5004/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input_text: message }),  // Changed to match the new format
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }
  
      const data = await response.json();
      const botMessage = data.response;  // Assuming your backend sends response in a 'response' field
      
      // Update chat history with bot's response
      setMessages([...newMessages, { text: botMessage, sender: 'bot' }]);
    } catch (error) {
      console.error('Error while fetching response:', error);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim()) {
      sendMessage(inputText);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-history">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            <p>{message.text}</p>
          </div>
        ))}
      </div>
      
      <form onSubmit={handleSubmit} className="chat-input">
        <input 
          type="text" 
          value={inputText} 
          onChange={(e) => setInputText(e.target.value)} 
          placeholder="Type your message..."
        />
        <button type="submit" className="send-button">Send</button>
      </form>
    </div>
  );
};

export default Chat;
