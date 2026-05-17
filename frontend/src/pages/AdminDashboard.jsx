import { useState, useEffect } from 'react';
import api from '../api';

function AdminDashboard({ user }) {
  const [activeTab, setActiveTab] = useState('users'); // 'users' ou 'logs'
  const [usersList, setUsersList] = useState([]);
  const [logs, setLogs] = useState([]);
  
  // État pour la modale d'inspection de l'historique
  const [inspectingUser, setInspectingUser] = useState(null);
  const [userHistory, setUserHistory] = useState([]);

  // Chargement des données au montage
  useEffect(() => {
    fetchUsers();
    fetchLogs();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await api.get('/admin/users');
      setUsersList(response.data);
    } catch (err) {
      console.error("Erreur chargement utilisateurs");
    }
  };

  const fetchLogs = async () => {
    try {
      const response = await api.get('/admin/logs');
      setLogs(response.data);
    } catch (err) {
      console.error("Erreur chargement logs");
    }
  };

  // Bannir / Supprimer un compte
  const handleDeleteUser = async (userId, username) => {
    if (window.confirm(`⚠️ ATTENTION : Voulez-vous vraiment bannir et supprimer le compte de "${username}" ? Toutes ses données médicales seront perdues.`)) {
      try {
        await api.delete(`/admin/users/${userId}`);
        fetchUsers(); // On rafraîchit la liste
      } catch (err) {
        alert("Erreur lors de la suppression.");
      }
    }
  };

  // Ouvrir la modale pour voir le dossier patient
  const handleInspectHistory = async (targetUser) => {
    setInspectingUser(targetUser);
    try {
      const response = await api.get(`/history/${targetUser.id}`);
      setUserHistory(response.data);
    } catch (err) {
      console.error("Erreur chargement historique");
    }
  };

  // Formatage de la date
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('fr-FR');
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="bg-gray-800 rounded-xl shadow-lg p-6 mb-8 text-white flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold flex items-center gap-2">🛡️ Centre de Commandement</h2>
          <p className="text-gray-400 mt-1">Espace réservé aux administrateurs</p>
        </div>
      </div>

      {/* SYSTÈME D'ONGLETS */}
      <div className="flex gap-4 mb-6">
        <button 
          onClick={() => setActiveTab('users')}
          className={`px-6 py-2 rounded-lg font-semibold transition ${activeTab === 'users' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-100'}`}
        >
          👥 Utilisateurs & Patients
        </button>
        <button 
          onClick={() => setActiveTab('logs')}
          className={`px-6 py-2 rounded-lg font-semibold transition ${activeTab === 'logs' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-100'}`}
        >
          📜 Logs & Sécurité
        </button>
      </div>

      {/* CONTENU : ONGLET UTILISATEURS */}
      {activeTab === 'users' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-50 border-b text-gray-600">
                <th className="p-4 font-semibold">ID</th>
                <th className="p-4 font-semibold">Utilisateur</th>
                <th className="p-4 font-semibold">Rôle</th>
                <th className="p-4 font-semibold">Date de création</th>
                <th className="p-4 font-semibold text-right">Actions de modération</th>
              </tr>
            </thead>
            <tbody>
              {usersList.map((u) => (
                <tr key={u.id} className="border-b hover:bg-gray-50">
                  <td className="p-4">{u.id}</td>
                  <td className="p-4 font-semibold">{u.username}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${u.role === 'admin' ? 'bg-purple-100 text-purple-700' : 'bg-green-100 text-green-700'}`}>
                      {u.role.toUpperCase()}
                    </span>
                  </td>
                  <td className="p-4 text-sm text-gray-500">{formatDate(u.created_at)}</td>
                  <td className="p-4 flex gap-2 justify-end">
                    <button 
                      onClick={() => handleInspectHistory(u)}
                      className="bg-blue-50 text-blue-600 px-3 py-1 rounded border border-blue-200 hover:bg-blue-100 text-sm font-semibold transition"
                    >
                      Dossier Patient
                    </button>
                    {/* On empêche l'admin de se supprimer lui-même */}
                    {u.id !== user.id && (
                      <button 
                        onClick={() => handleDeleteUser(u.id, u.username)}
                        className="bg-red-50 text-red-600 px-3 py-1 rounded border border-red-200 hover:bg-red-100 text-sm font-semibold transition"
                      >
                        Bannir
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CONTENU : ONGLET LOGS */}
      {activeTab === 'logs' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-50 border-b text-gray-600">
                <th className="p-4 font-semibold">Date & Heure</th>
                <th className="p-4 font-semibold">Utilisateur</th>
                <th className="p-4 font-semibold">Action effectuée</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.id} className="border-b hover:bg-gray-50 text-sm">
                  <td className="p-4 text-gray-500">{formatDate(log.timestamp)}</td>
                  <td className="p-4 font-semibold">{log.username} <span className="text-xs text-gray-400">({log.role})</span></td>
                  <td className="p-4">{log.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* MODALE D'INSPECTION DU DOSSIER PATIENT */}
      {inspectingUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col">
            <div className="p-6 border-b flex justify-between items-center bg-gray-50 rounded-t-xl">
              <h3 className="text-xl font-bold">Dossier médical de : <span className="text-blue-600">{inspectingUser.username}</span></h3>
              <button onClick={() => setInspectingUser(null)} className="text-gray-500 hover:text-red-500 font-bold text-xl">&times;</button>
            </div>
            
            <div className="p-6 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-4">
              {userHistory.length === 0 ? (
                <p className="col-span-2 text-center text-gray-500 py-10">Aucun diagnostic effectué par cet utilisateur.</p>
              ) : (
                userHistory.map(diag => (
                  <div key={diag.id} className="border rounded-lg p-4 flex gap-4 bg-gray-50">
                    <img src={`http://localhost:8000/${diag.image_path}`} alt="Lésion" className="w-24 h-24 object-cover rounded shadow" />
                    <div>
                      <p className="text-xs text-gray-400 mb-1">{formatDate(diag.created_at)}</p>
                      <p className="text-sm font-semibold">Zone: {diag.localization}</p>
                      <p className={`text-sm font-bold mt-2 ${diag.top_prediction.includes("Mélanome") ? 'text-red-600' : 'text-green-600'}`}>
                        {diag.top_prediction} ({diag.top_probability}%)
                      </p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default AdminDashboard;