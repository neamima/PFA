import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000', // L'adresse de ton backend FastAPI
});

export default api;