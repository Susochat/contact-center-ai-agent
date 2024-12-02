import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, Container, Box, Paper, Grid, Card, CardContent } from '@mui/material';

const Home = () => {
  const [inputText, setInputText] = useState('');
  const [conversation, setConversation] = useState([]); // Store conversation history
  const [responseData, setResponseData] = useState(null); // To store full response data

  // Function to handle the form submission and differentiate between normal input and JSON
  const handleSubmit = async () => {
    if (!inputText) return;

    try {
      // Check if inputText is a valid JSON string
      const parsedInput = JSON.parse(inputText);

      // If it's valid JSON, set the conversation directly
      if (parsedInput && parsedInput.transcript) {
        setConversation(parsedInput.transcript);
      }
    } catch (error) {
      // Otherwise, treat it as a normal user input message
      const userMessage = { sender: 'user', text: inputText };
      setConversation([...conversation, userMessage]); // Add user message to conversation

      // Send the user message to your backend to get AI response
      try {
        const res = await axios.post('http://127.0.0.1:5004/analyze', { input_text: inputText });
        const aiMessage = { sender: 'ai', text: res.data.response };
        setConversation((prevConv) => [...prevConv, aiMessage]); // Add AI response to conversation
        setResponseData(res.data); // Set the response data to display it below
      } catch (error) {
        console.error('Error fetching data', error);
      }
    }

    setInputText('');  // Clear input field after sending the message
  };

  return (
    <Container maxWidth="md" style={{ marginTop: '50px' }}>
      <Box display="flex" flexDirection="column" alignItems="center">
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', color: '#333' }}>
          Contact Center AI Agent Assistant
        </Typography>
        
        {/* Chat history */}
        <Box
          component={Paper}
          style={{ width: '100%', height: '300px', overflowY: 'auto', padding: '15px', marginBottom: '20px', backgroundColor: '#f9f9f9' }}
          elevation={3}
        >
          {conversation.map((message, index) => (
            <Box
              key={index}
              style={{
                textAlign: message.sender === 'user' ? 'right' : 'left',
                marginBottom: '10px',
              }}
            >
              <Typography variant="body1" style={{
                backgroundColor: message.sender === 'user' ? '#007BFF' : '#f1f1f1',
                color: message.sender === 'user' ? '#fff' : '#333',
                padding: '10px',
                borderRadius: '20px',
                display: 'inline-block',
                maxWidth: '70%',
                wordBreak: 'break-word'
              }}>
                {message.text}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Input and Send button */}
        <Box display="flex" width="100%" alignItems="center" marginBottom={2}>
          <TextField
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            label="Type your message..."
            variant="outlined"
            fullWidth
            sx={{
              marginRight: 2,
              borderRadius: '20px',
              backgroundColor: '#fff',
              '& .MuiOutlinedInput-root': { 
                borderRadius: '20px',
              },
              '& .MuiInputLabel-root': {
                color: '#333',
              }
            }}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            sx={{
              padding: '10px 20px',
              height: '56px',
              borderRadius: '20px',
              fontWeight: 'bold',
              textTransform: 'none',
            }}
          >
            Send
          </Button>
        </Box>

        {/* Displaying the AI Response Structure */}
        {responseData && (
          <Box marginTop={4} width="100%">
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#444' }}>
              AI Response Summary
            </Typography>
            <Paper style={{ padding: '20px', marginBottom: '20px', backgroundColor: '#f5f5f5' }} elevation={3}>
              <Grid container spacing={2}>
                {/* Sentiment */}
                <Grid item xs={12} sm={6}>
                  <Card sx={{ height: '200px', display: 'flex', flexDirection: 'column', borderRadius: '15px', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
                    <CardContent sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#007BFF' }}>Sentiment:</Typography>
                      <Typography variant="body2">Label: {responseData.sentiment.label}</Typography>
                      <Typography variant="body2">Score: {responseData.sentiment.score.toFixed(3)}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                {/* Intent */}
                <Grid item xs={12} sm={6}>
                  <Card sx={{ height: '200px', display: 'flex', flexDirection: 'column', borderRadius: '15px', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
                    <CardContent sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#007BFF' }}>Intent:</Typography>
                      <Typography variant="body2">{responseData.intent}</Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* AI Response */}
                <Grid item xs={12}>
                  <Card sx={{ borderRadius: '15px', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#007BFF' }}>AI Response:</Typography>
                      <Typography variant="body2">{responseData.response}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        )}
      </Box>
    </Container>
  );
};

export default Home;
