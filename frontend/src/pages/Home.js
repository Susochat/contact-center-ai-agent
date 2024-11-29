// src/pages/Home.js
import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, Container, Box, Paper } from '@mui/material';

const Home = () => {
  const [response, setResponse] = useState('');
  const [inputText, setInputText] = useState('');
  const [conversation, setConversation] = useState([]); // Store conversation history

  const handleSubmit = async () => {
    if (!inputText) return;

    const userMessage = { sender: 'user', text: inputText };
    setConversation([...conversation, userMessage]); // Add user message to conversation

    try {
      const res = await axios.post('http://127.0.0.1:5004/generate', { input_text: inputText });
      const aiMessage = { sender: 'ai', text: res.data.response };
      setConversation((prevConv) => [...prevConv, aiMessage]); // Add AI response to conversation
      setResponse(res.data.response);  // Display AI's response
    } catch (error) {
      console.error('Error fetching data', error);
    }

    setInputText('');  // Clear input field after sending the message
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: '50px' }}>
      <Box display="flex" flexDirection="column" alignItems="center">
        <Typography variant="h4" gutterBottom>
          Contact Center AI
        </Typography>
        
        {/* Chat history */}
        <Box
          component={Paper}
          style={{ width: '100%', height: '300px', overflowY: 'auto', padding: '10px', marginBottom: '20px' }}
        >
          {conversation.map((message, index) => (
            <Box
              key={index}
              style={{
                textAlign: message.sender === 'user' ? 'right' : 'left',
                marginBottom: '10px',
              }}
            >
              <Typography variant="body1" style={{ backgroundColor: message.sender === 'user' ? '#e1f5fe' : '#f1f1f1', padding: '8px', borderRadius: '10px', display: 'inline-block' }}>
                {message.text}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Single-line Input and Send button */}
        <Box display="flex" width="100%" alignItems="center">
          <TextField
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            label="Enter your query"
            variant="outlined"
            fullWidth
            style={{ marginBottom: '20px', marginRight: '10px' }}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            style={{ height: '100%' }}
          >
            Send
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default Home;
