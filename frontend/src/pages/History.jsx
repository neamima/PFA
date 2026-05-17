import { useState, useEffect } from 'react';
import api from '../api';

function History({ user }) {
  const [diagnoses, setDiagnoses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await api.get(`/history/${user.id}`);
        setDiagnoses(response.data);
      } catch (err) {
        setError("Impossible de charger l'historique.");
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [user.id]);

  if (loading) {
    return <div className="text-center mt-20 text-xl animate-pulse">⏳ Chargement de votre dossier patient...</div>;
  }

  if (error) {
    return <div className="text-center mt-20 text-red-500 font-bold">{error}</div>;
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="bg-white rounded-xl shadow-sm p-6 mb-8 border border-gray-100 flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">📁 Mon Historique Clinique</h2>
          <p className="text-gray-500 mt-1">Dossier patient de <span className="font-semibold">{user.username}</span></p>
        </div>
        <div className="bg-blue-50 text-blue-800 px-4 py-2 rounded-lg font-semibold">
          {diagnoses.length} analyse(s) effectuée(s)
        </div>
      </div>

      {diagnoses.length === 0 ? (
        <div className="text-center bg-white p-10 rounded-xl border border-dashed border-gray-300 text-gray-500">
          <span className="text-6xl mb-4 block">📭</span>
          Vous n'avez pas encore effectué de diagnostic.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {diagnoses.map((diag) => {
            const isMelanoma = diag.top_prediction.includes("Mélanome");
            // Formatage de la date SQL vers un format lisible
            const diagDate = new Date(diag.created_at).toLocaleDateString('fr-FR', {
              day: '2-digit', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit'
            });

            return (
              <div key={diag.id} className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition">
                {/* Image : on demande l'image au backend FastAPI */}
                <div className="h-48 bg-gray-200 overflow-hidden relative">
                  <img 
                    src={`http://localhost:8000/${diag.image_path}`} 
                    alt="Lésion" 
                    className="w-full h-full object-cover"
                  />
                  {isMelanoma && (
                    <div className="absolute top-2 right-2 bg-red-600 text-white text-xs font-bold px-2 py-1 rounded shadow">
                      Alerte IA
                    </div>
                  )}
                </div>
                
                <div className="p-5">
                  <div className="text-sm text-gray-400 mb-3">{diagDate}</div>
                  
                  <div className="grid grid-cols-2 gap-2 text-sm mb-4">
                    <div className="bg-gray-50 p-2 rounded">
                      <span className="block text-gray-500 text-xs">Patient</span>
                      <span className="font-semibold text-gray-700">{diag.age} ans, {diag.sex}</span>
                    </div>
                    <div className="bg-gray-50 p-2 rounded">
                      <span className="block text-gray-500 text-xs">Zone</span>
                      <span className="font-semibold text-gray-700">{diag.localization}</span>
                    </div>
                  </div>

                  <div className={`p-3 rounded-lg border ${isMelanoma ? 'bg-red-50 border-red-100' : 'bg-green-50 border-green-100'}`}>
                    <span className="block text-xs font-semibold mb-1 opacity-75">Résultat de l'analyse :</span>
                    <div className={`font-bold ${isMelanoma ? 'text-red-700' : 'text-green-700'}`}>
                      {diag.top_prediction}
                    </div>
                    <div className="mt-2 text-sm flex justify-between items-center">
                      <span>Confiance :</span>
                      <span className="font-bold">{diag.top_probability}%</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default History;