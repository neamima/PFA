import { useState } from 'react';
import api from '../api';

function Login({ setUser }) {
  const [isRegistering, setIsRegistering] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    // Pour FastAPI, il faut envoyer les formulaires en format "FormData"
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    try {
      if (isRegistering) {
        // --- INSCRIPTION ---
        await api.post('/register', formData);
        setSuccess("Compte créé avec succès ! Vous pouvez maintenant vous connecter.");
        setIsRegistering(false); // On repasse sur le mode connexion
        setPassword('');
      } else {
        // --- CONNEXION ---
        const response = await api.post('/login', formData);
        const userData = response.data;
        
        // On sauvegarde l'utilisateur dans le stockage du navigateur
        localStorage.setItem('user', JSON.stringify(userData));
        setUser(userData); // Met à jour le routeur dans App.jsx
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Une erreur est survenue.");
    }
  };

  return (
    <div className="max-w-md mx-auto mt-16 bg-white p-8 rounded-xl shadow-lg border border-gray-100">
      <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">
        {isRegistering ? "Créer un compte" : "Connexion"}
      </h2>

      {error && <div className="bg-red-50 text-red-600 p-3 rounded-lg mb-4 text-sm text-center">{error}</div>}
      {success && <div className="bg-green-50 text-green-600 p-3 rounded-lg mb-4 text-sm text-center">{success}</div>}

      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Nom d'utilisateur</label>
          <input 
            type="text" 
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Mot de passe</label>
          <input 
            type="password" 
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        <button 
          type="submit" 
          className="w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition duration-200 mt-2"
        >
          {isRegistering ? "S'inscrire" : "Se connecter"}
        </button>
      </form>

      <p className="text-center text-gray-500 text-sm mt-6">
        {isRegistering ? "Déjà un compte ? " : "Pas encore de compte ? "}
        <button 
          onClick={() => { setIsRegistering(!isRegistering); setError(''); setSuccess(''); }} 
          className="text-blue-600 font-semibold hover:underline"
        >
          {isRegistering ? "Connectez-vous" : "Créez-en un"}
        </button>
      </p>
    </div>
  );
}

export default Login;