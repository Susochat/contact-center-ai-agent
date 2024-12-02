// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import Home_dgpt2 from './pages/Home_dgpt2';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} /> 
        {/* <Route path="/" element={<Home_dgpt2 />} /> */}
      </Routes>
    </Router>
  );
};

export default App;
