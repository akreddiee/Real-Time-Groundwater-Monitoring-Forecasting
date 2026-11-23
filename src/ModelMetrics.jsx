import React, { useState, useEffect } from "react";
import { FaBars, FaTimes, FaUserCircle, FaTachometerAlt, FaMapMarkerAlt, FaBell, FaChartBar, FaBrain } from "react-icons/fa";
import { Link } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function ModelMetrics({ user }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "https://realtime-ground-water-level-monitoring-1.onrender.com";

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/metrics`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load metrics");
        return res.json();
      })
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Unable to load model metrics.");
        setLoading(false);
      });
  }, []);

  return (
    <div className="flex flex-col min-h-screen">
      {/* Navbar */}
      <nav className="flex items-center justify-between bg-blue-600 p-4 text-white z-50">
        <div className="flex items-center space-x-3">
          <button onClick={() => setMenuOpen(!menuOpen)} className="text-2xl">
            {menuOpen ? <FaTimes /> : <FaBars />}
          </button>
          <span className="font-bold text-xl">Aquapulse — ML Metrics</span>
        </div>

        <FaUserCircle className="cursor-pointer text-2xl" onClick={() => setProfileOpen(!profileOpen)} />
      </nav>

      {/* Sidebar */}
      <div className={`fixed top-0 left-0 h-full w-64 bg-blue-700 text-white z-[9999] shadow-2xl transform transition-transform duration-300 ${menuOpen ? "translate-x-0" : "-translate-x-full"}`}>
        <div className="p-6 space-y-4">
          <Link to="/dashboard" className="flex items-center space-x-2"><FaTachometerAlt /> <span>Dashboard</span></Link>
          <Link to="/stations" className="flex items-center space-x-2"><FaMapMarkerAlt /> <span>Stations</span></Link>
          <Link to="/alerts" className="flex items-center space-x-2"><FaBell /> <span>Alerts</span></Link>
          <Link to="/charts" className="flex items-center space-x-2"><FaChartBar /> <span>Charts</span></Link>
          <Link to="/metrics" className="flex items-center space-x-2"><FaBrain /> <span>ML Performance</span></Link>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-grow p-6 bg-gray-100">
        <div className="max-w-6xl mx-auto space-y-6">
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <FaBrain className="text-blue-600" /> Machine Learning Model Performance & Metrics
          </h2>

          {loading && <p>Loading model evaluation stats...</p>}
          {error && <p className="text-red-600">{error}</p>}

          {metrics && (
            <>
              {/* Scorecards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Groundwater Model */}
                <div className="bg-white p-6 rounded-2xl shadow border-t-4 border-blue-600">
                  <h3 className="text-lg font-semibold text-gray-700 mb-4">Groundwater Model (Gradient Boosting)</h3>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="bg-blue-50 p-3 rounded-xl">
                      <p className="text-xs text-gray-500 font-bold uppercase">R² Score</p>
                      <p className="text-2xl font-black text-blue-600">{metrics.groundwater_model?.r2_score}</p>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-xl">
                      <p className="text-xs text-gray-500 font-bold uppercase">RMSE</p>
                      <p className="text-2xl font-black text-blue-600">{metrics.groundwater_model?.rmse}m</p>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-xl">
                      <p className="text-xs text-gray-500 font-bold uppercase">MAE</p>
                      <p className="text-2xl font-black text-blue-600">{metrics.groundwater_model?.mae}m</p>
                    </div>
                  </div>
                </div>

                {/* Rainfall Model */}
                <div className="bg-white p-6 rounded-2xl shadow border-t-4 border-emerald-500">
                  <h3 className="text-lg font-semibold text-gray-700 mb-4">Rainfall Model (Gradient Boosting)</h3>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="bg-emerald-50 p-3 rounded-xl">
                      <p className="text-xs text-gray-500 font-bold uppercase">R² Score</p>
                      <p className="text-2xl font-black text-emerald-600">{metrics.rainfall_model?.r2_score}</p>
                    </div>
                    <div className="bg-emerald-50 p-3 rounded-xl">
                      <p className="text-xs text-gray-500 font-bold uppercase">RMSE</p>
                      <p className="text-2xl font-black text-emerald-600">{metrics.rainfall_model?.rmse}mm</p>
                    </div>
                    <div className="bg-emerald-50 p-3 rounded-xl">
                      <p className="text-xs text-gray-500 font-bold uppercase">MAE</p>
                      <p className="text-2xl font-black text-emerald-600">{metrics.rainfall_model?.mae}mm</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Feature Importances Chart */}
              <div className="bg-white p-6 rounded-2xl shadow">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Feature Importance Rankings</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={metrics.feature_importances?.slice(0, 10)} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" width={140} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
