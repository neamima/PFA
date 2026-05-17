import { useState, useRef } from 'react';
import api from '../api';

function Diagnostic({ user }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [age, setAge] = useState(30);
  const [sex, setSex] = useState('Non spécifié');
  const [localization, setLocalization] = useState('Non spécifiée');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  // Gère la sélection de l'image et crée une prévisualisation
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null); // On efface l'ancien résultat si on change d'image
    }
  };

  // Envoi des données au backend FastAPI
  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Veuillez sélectionner une image.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('user_id', user.id);
    formData.append('age', age);
    formData.append('sex', sex);
    formData.append('localization', localization);
    formData.append('file', file);

    try {
      const response = await api.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (err) {
      setError("Erreur lors de l'analyse : " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      
      {/* HEADER DE LA PAGE */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-6 border border-gray-100 text-center">
        <h2 className="text-2xl font-bold text-gray-800">🔬 Aide au Diagnostic Dermatologique</h2>
        <p className="text-gray-500 mt-2">Uploadez une image dermoscopique et renseignez le dossier patient.</p>
        <div className="mt-4 inline-block bg-yellow-50 text-yellow-700 px-4 py-2 rounded-lg text-sm border border-yellow-200">
          ⚠️ <strong>Avertissement :</strong> Prototype PFA. Ne remplace pas l'avis d'un médecin.
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* COLONNE GAUCHE : FORMULAIRE */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-700 border-b pb-2 mb-4">📋 Dossier Patient</h3>
          
          <form onSubmit={handleAnalyze} className="flex flex-col gap-4">
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-600 mb-1">Âge</label>
                <input type="number" min="0" max="120" value={age} onChange={(e) => setAge(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
              </div>
              <div>
                <label className="block text-sm text-gray-600 mb-1">Sexe</label>
                <select value={sex} onChange={(e) => setSex(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none">
                  <option>Non spécifié</option><option>Homme</option><option>Femme</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm text-gray-600 mb-1">Localisation de la lésion</label>
              <select value={localization} onChange={(e) => setLocalization(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none">
                <option>Non spécifiée</option><option>Dos</option><option>Visage</option><option>Tronc</option>
                <option>Bras/Jambe</option><option>Cuir chevelu</option><option>Main/Pied</option><option>Autre</option>
              </select>
            </div>

            <div className="mt-2">
              <label className="block text-sm text-gray-600 mb-2">📸 Image dermoscopique</label>
              
              {/* ZONE DRAG & DROP DESIGN */}
              <div 
                onClick={() => fileInputRef.current.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:bg-gray-50 transition"
              >
                {preview ? (
                  <img src={preview} alt="Aperçu" className="max-h-48 mx-auto rounded-lg shadow-sm" />
                ) : (
                  <div className="text-gray-400">
                    <span className="text-3xl block mb-2">📂</span>
                    <span className="text-sm">Cliquez pour sélectionner une image</span>
                  </div>
                )}
              </div>
              <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden" />
            </div>

            {error && <div className="text-red-500 text-sm mt-2">{error}</div>}

            <button 
              type="submit" 
              disabled={loading || !file}
              className={`mt-4 w-full py-3 rounded-lg text-white font-semibold flex justify-center items-center gap-2 transition
                ${loading || !file ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
            >
              {loading ? (
                <><span className="animate-spin text-xl">⏳</span> Analyse en cours...</>
              ) : (
                <><span className="text-xl">✨</span> Lancer l'analyse IA</>
              )}
            </button>
          </form>
        </div>

        {/* COLONNE DROITE : RÉSULTAT */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 flex flex-col justify-center min-h-[400px]">
          {result ? (
            <div className="text-center animate-fade-in">
              <h3 className="text-xl font-bold text-gray-800 mb-6">Résultat de l'IA</h3>
              
              <div className={`text-6xl mb-4 ${result.prediction.includes("Mélanome") ? 'text-red-500' : 'text-green-500'}`}>
                {result.prediction.includes("Mélanome") ? '🚨' : '✅'}
              </div>
              
              <div className="text-lg text-gray-600 mb-2">Prédiction principale :</div>
              <div className={`text-2xl font-bold px-4 py-2 rounded-lg inline-block
                ${result.prediction.includes("Mélanome") ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                {result.prediction}
              </div>
              
              <div className="mt-6 text-gray-500">
                Indice de confiance :
                <div className="w-full bg-gray-200 rounded-full h-4 mt-2 max-w-xs mx-auto">
                  <div 
                    className={`h-4 rounded-full ${result.prediction.includes("Mélanome") ? 'bg-red-500' : 'bg-green-500'}`} 
                    style={{ width: `${result.probability}%` }}
                  ></div>
                </div>
                <div className="font-bold text-gray-700 mt-1">{result.probability}%</div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-400">
              <span className="text-6xl block mb-4 opacity-50">🤖</span>
              <p>L'intelligence artificielle est en attente d'une image pour formuler une prédiction.</p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default Diagnostic;