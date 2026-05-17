import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Login from './pages/Login';
import Diagnostic from './pages/Diagnostic'; 
import History from './pages/History';
import AdminDashboard from './pages/AdminDashboard';

function App() {
  const [user, setUser] = useState(null);

  // Vérifie si un utilisateur est déjà connecté (sauvegardé dans le navigateur)
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('user');
    setUser(null);
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 font-sans text-gray-900">
        
        {/* BARRE DE NAVIGATION */}
        <nav className="bg-blue-600 text-white shadow-md p-4">
          <div className="max-w-6xl mx-auto flex justify-between items-center">
            <h1 className="text-xl font-bold tracking-wider">🔬 PeauIA</h1>
            
            {user && (
              <div className="flex gap-6 items-center">
                <Link to="/" className="hover:text-blue-200 transition">Diagnostic</Link>
                <Link to="/history" className="hover:text-blue-200 transition">Mon Historique</Link>
                {user.role === 'admin' && (
                  <Link to="/admin" className="text-yellow-300 font-bold hover:text-yellow-100 transition">⚡ Admin</Link>
                )}
                <div className="border-l border-blue-400 pl-4 flex items-center gap-4">
                  <span className="text-sm bg-blue-800 px-3 py-1 rounded-full">👤 {user.username}</span>
                  <button onClick={handleLogout} className="text-sm text-red-200 hover:text-white">Déconnexion</button>
                </div>
              </div>
            )}
          </div>
        </nav>

        {/* ZONE PRINCIPALE (Là où les pages s'affichent) */}
        <main className="max-w-6xl mx-auto p-6 mt-6">
          <Routes>
            {/* Si non connecté, on force vers Login. Sinon, vers Diagnostic */}
            <Route path="/login" element={!user ? <Login setUser={setUser} /> : <Navigate to="/" />} />
            
            {/* Routes protégées (on décommentera quand on aura créé les pages) */}
            <Route path="/" element={user ? <Diagnostic user={user} /> : <Navigate to="/login" />} />
            <Route path="/history" element={user ? <History user={user} /> : <Navigate to="/login" />} />
            <Route 
              path="/admin" 
              element={user?.role === 'admin' ? <AdminDashboard user={user} /> : <Navigate to="/" />} 
            />
          </Routes>
        </main>

      </div>
    </Router>
  );
}

export default App;