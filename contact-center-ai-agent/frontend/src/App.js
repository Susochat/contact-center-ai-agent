// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import HomeOpenAI from './pages/HomeOpenAI';

const App = () => {
  return (
    <Router>
      <Routes>
        {/* <Route path="/" element={<Home />} />  */}
        <Route path="/" element={<HomeOpenAI />} />
      </Routes>
    </Router>
  );
};

export default App;
