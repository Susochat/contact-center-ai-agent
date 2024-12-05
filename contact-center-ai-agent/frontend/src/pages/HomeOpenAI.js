import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, Container, Box, Paper, Grid, Card, CardContent, LinearProgress } from '@mui/material';

// Function to map sentiment score to color
const getSentimentColor = (sentiment) => {
  if (sentiment === 'Positive') return '#4caf50'; // Green
  if (sentiment === 'Neutral') return '#ffc107'; // Amber
  if (sentiment === 'Negative') return '#f44336'; // Red
  return '#9e9e9e'; // Default grey
};

const HomeOpenAI = () => {
  const [inputText, setInputText] = useState('');
  const [conversation, setConversation] = useState([]); // Store conversation history
  const [responseData, setResponseData] = useState(null); // Store AI response data

  // Handle user input submission
  const handleSubmit = async () => {
    if (!inputText) return;
  
    const userMessage = { sender: 'user', text: inputText };
    const updatedConversation = [...conversation, userMessage];
  
    setConversation(updatedConversation);
  
    try {
      // Send the full conversation history as `history`
      const res = await axios.post('http://127.0.0.1:8000/analyze', {
        history: updatedConversation.map((msg) => ({ sender: msg.sender, text: msg.text }))
      });

      const aiMessage = { sender: 'ai', text: res.data.suggestions };
      setConversation([...updatedConversation, aiMessage]);
      setResponseData(res.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  
    setInputText('');
  };

  return (
    <Container maxWidth="md" style={{ marginTop: '50px' }}>
      <Box display="flex" flexDirection="column" alignItems="center">
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', color: '#333' }}>
          Contact Center AI Agent Assistant
        </Typography>

        {/* Chat History */}
        <Box
          component={Paper}
          style={{
            width: '100%',
            height: '300px',
            overflowY: 'auto',
            padding: '15px',
            marginBottom: '20px',
            backgroundColor: '#f9f9f9'
          }}
          elevation={3}
        >
          {conversation.map((message, index) => (
            <Box
              key={index}
              style={{
                textAlign: message.sender === 'user' ? 'right' : 'left',
                marginBottom: '10px'
              }}
            >
              <Typography
                variant="body1"
                style={{
                  backgroundColor: message.sender === 'user' ? '#007BFF' : '#f1f1f1',
                  color: message.sender === 'user' ? '#fff' : '#333',
                  padding: '10px',
                  borderRadius: '20px',
                  display: 'inline-block',
                  maxWidth: '70%',
                  wordBreak: 'break-word',
                  whiteSpace: 'pre-wrap' // Ensures multi-line text is properly displayed
                }}
              >
                {message.text}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Input and Send Button */}
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
                borderRadius: '20px'
              },
              '& .MuiInputLabel-root': {
                color: '#333'
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
              textTransform: 'none'
            }}
          >
            Send
          </Button>
        </Box>

        {/* AI Response Details */}
        {responseData && (
          <Box marginTop={4} width="100%">
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#444' }}>
              AI Response Summary
            </Typography>
            <Paper sx={{ padding: 3, backgroundColor: '#f5f5f5' }}>
              <Grid container spacing={2}>
                {/* Sentiment with Indicator */}
                <Grid item xs={12} sm={6}>
                  <Card
                    sx={{
                      height: '200px',
                      display: 'flex',
                      flexDirection: 'column',
                      borderRadius: '15px',
                    }}
                  >
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#007BFF' }}>
                        Sentiment:
                      </Typography>
                      <Typography
                        variant="h6"
                        sx={{
                          fontWeight: 'bold',
                          color: getSentimentColor(responseData.sentiment),
                          marginBottom: 1,
                        }}
                      >
                        {responseData.sentiment}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={
                          responseData.sentiment === 'Positive'
                            ? 100
                            : responseData.sentiment === 'Neutral'
                            ? 50
                            : 25
                        }
                        sx={{
                          height: 10,
                          borderRadius: 5,
                          backgroundColor: '#e0e0e0',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getSentimentColor(responseData.sentiment),
                          },
                        }}
                      />
                    </CardContent>
                  </Card>
                </Grid>

                {/* Intent */}
                <Grid item xs={12} sm={6}>
                  <Card
                    sx={{
                      height: '200px',
                      display: 'flex',
                      flexDirection: 'column',
                      borderRadius: '15px',
                      boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)'
                    }}
                  >
                    <CardContent sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#007BFF' }}>
                        Intent:
                      </Typography>
                      <Typography variant="body2">{responseData.intent}</Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* AI Response Summary */}
                <Grid item xs={12}>
                  <Card sx={{ borderRadius: '15px', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#007BFF' }}>
                        Summary:
                      </Typography>
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {responseData.summary}
                      </Typography>
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

export default HomeOpenAI;

