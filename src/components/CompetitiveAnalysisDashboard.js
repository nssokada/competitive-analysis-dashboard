// src/components/CompetitiveAnalysisDashboard.js
import { useState, useEffect } from 'react';
import Papa from 'papaparse';

const CompetitiveAnalysisDashboard = () => {
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [features, setFeatures] = useState([]);
  const [xAxis, setXAxis] = useState('');
  const [yAxis, setYAxis] = useState('');
  const [hoveredCompany, setHoveredCompany] = useState(null);

  // Load CSV data on component mount
  useEffect(() => {
    const loadCSVData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch the CSV file from the public folder
        const response = await fetch(`${process.env.PUBLIC_URL}/data/companies.csv`);
        
        if (!response.ok) {
          throw new Error(`Failed to load CSV: ${response.status} ${response.statusText}`);
        }

        const csvText = await response.text();

        // Parse CSV with Papa Parse
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            if (results.errors.length > 0) {
              console.warn('CSV parsing warnings:', results.errors);
            }

            // Clean and validate data
            const cleanedData = results.data
              .filter(row => row.name && row.name.trim()) // Remove rows without names
              .map(row => {
                // Clean up the row data
                const cleanedRow = {};
                Object.keys(row).forEach(key => {
                  const cleanKey = key.trim().toLowerCase();
                  if (cleanKey === 'name') {
                    cleanedRow[cleanKey] = row[key].toString().trim();
                  } else {
                    // Convert to number, default to 0 if invalid
                    cleanedRow[cleanKey] = isNaN(row[key]) ? 0 : Number(row[key]);
                  }
                });
                return cleanedRow;
              });

            if (cleanedData.length === 0) {
              throw new Error('No valid data found in CSV file');
            }

            // Extract feature columns (excluding 'name')
            const sampleRow = cleanedData[0];
            const featureKeys = Object.keys(sampleRow).filter(key => key !== 'name');
            
            if (featureKeys.length < 2) {
              throw new Error('CSV must contain at least 2 numeric columns besides name');
            }

            // Create features array with proper labels
            const featureLabels = {
              target: { label: 'Target Market Score', unit: '%' },
              indicator: { label: 'Key Indicator Performance', unit: '%' },
              delivery: { label: 'Delivery Excellence', unit: '%' },
              stage: { label: 'Development Stage', unit: '/10' },
              funding: { label: 'Funding Score', unit: '%' }
            };

            const detectedFeatures = featureKeys.map(key => ({
              key,
              label: featureLabels[key]?.label || key.charAt(0).toUpperCase() + key.slice(1),
              unit: featureLabels[key]?.unit || ''
            }));

            setCompanies(cleanedData);
            setFeatures(detectedFeatures);
            
            // Set default axes to first two features
            setXAxis(featureKeys[0]);
            setYAxis(featureKeys[1]);
            
            setLoading(false);
          },
          error: (error) => {
            throw new Error(`CSV parsing failed: ${error.message}`);
          }
        });

      } catch (err) {
        console.error('Error loading CSV data:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    loadCSVData();
  }, []);

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-slate-600">Loading competitive analysis data...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto p-6">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-slate-800 mb-2">Error Loading Data</h2>
          <p className="text-slate-600 mb-4">{error}</p>
          <p className="text-sm text-slate-500">
            Make sure your CSV file is located at <code className="bg-slate-100 px-2 py-1 rounded">public/data/companies.csv</code>
          </p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // No data state
  if (companies.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <p className="text-slate-600">No company data available</p>
        </div>
      </div>
    );
  }

  const chartWidth = 600;
  const chartHeight = 400;
  const padding = 60;

  const getFeatureRange = (featureKey) => {
    const values = companies.map(c => c[featureKey]).filter(val => !isNaN(val));
    if (values.length === 0) return { min: 0, max: 100 };
    
    return {
      min: Math.min(...values),
      max: Math.max(...values)
    };
  };

  const xRange = getFeatureRange(xAxis);
  const yRange = getFeatureRange(yAxis);

  const scaleX = (value) => {
    if (xRange.max === xRange.min) return chartWidth / 2;
    return padding + ((value - xRange.min) / (xRange.max - xRange.min)) * (chartWidth - 2 * padding);
  };

  const scaleY = (value) => {
    if (yRange.max === yRange.min) return chartHeight / 2;
    return chartHeight - padding - ((value - yRange.min) / (yRange.max - yRange.min)) * (chartHeight - 2 * padding);
  };

  const getQuadrantLabel = (x, y) => {
    const midX = (xRange.min + xRange.max) / 2;
    const midY = (yRange.min + yRange.max) / 2;
    
    if (x >= midX && y >= midY) return "Leaders";
    if (x < midX && y >= midY) return "Challengers";
    if (x < midX && y < midY) return "Niche Players";
    return "Visionaries";
  };

  const getCompanyColor = (company) => {
    if (company.name.toLowerCase().includes("your") || company.name.toLowerCase().includes("our")) {
      return "#3b82f6";
    }
    return "#64748b";
  };

  const selectedXFeature = features.find(f => f.key === xAxis);
  const selectedYFeature = features.find(f => f.key === yAxis);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 mb-2">Competitive Analysis Dashboard</h1>
          <p className="text-slate-600">Dynamic quad chart analysis across multiple dimensions</p>
          <p className="text-sm text-slate-500 mt-1">
            Data loaded from CSV • {companies.length} companies • {features.length} features
          </p>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                X-Axis Feature
              </label>
              <select 
                value={xAxis} 
                onChange={(e) => setXAxis(e.target.value)}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              >
                {features.map(feature => (
                  <option key={feature.key} value={feature.key}>
                    {feature.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                Y-Axis Feature
              </label>
              <select 
                value={yAxis} 
                onChange={(e) => setYAxis(e.target.value)}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              >
                {features.map(feature => (
                  <option key={feature.key} value={feature.key}>
                    {feature.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-slate-800 mb-6 text-center">
            {selectedYFeature.label} vs {selectedXFeature.label}
          </h2>
          
          <div className="flex justify-center">
            <div className="relative">
              <svg width={chartWidth} height={chartHeight} className="border border-slate-200 rounded-lg">
                {/* Grid lines */}
                <defs>
                  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f1f5f9" strokeWidth="1"/>
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />
                
                {/* Quadrant lines */}
                <line 
                  x1={scaleX((xRange.min + xRange.max) / 2)} 
                  y1={padding} 
                  x2={scaleX((xRange.min + xRange.max) / 2)} 
                  y2={chartHeight - padding}
                  stroke="#cbd5e1" 
                  strokeWidth="2" 
                  strokeDasharray="5,5"
                />
                <line 
                  x1={padding} 
                  y1={scaleY((yRange.min + yRange.max) / 2)} 
                  x2={chartWidth - padding} 
                  y2={scaleY((yRange.min + yRange.max) / 2)}
                  stroke="#cbd5e1" 
                  strokeWidth="2" 
                  strokeDasharray="5,5"
                />

                {/* Quadrant labels */}
                <text x={chartWidth - padding/2} y={padding/2} textAnchor="middle" className="text-xs font-semibold fill-slate-600">Leaders</text>
                <text x={padding/2} y={padding/2} textAnchor="middle" className="text-xs font-semibold fill-slate-600">Challengers</text>
                <text x={padding/2} y={chartHeight - padding/4} textAnchor="middle" className="text-xs font-semibold fill-slate-600">Niche Players</text>
                <text x={chartWidth - padding/2} y={chartHeight - padding/4} textAnchor="middle" className="text-xs font-semibold fill-slate-600">Visionaries</text>

                {/* Axes */}
                <line x1={padding} y1={chartHeight - padding} x2={chartWidth - padding} y2={chartHeight - padding} stroke="#1e293b" strokeWidth="2"/>
                <line x1={padding} y1={padding} x2={padding} y2={chartHeight - padding} stroke="#1e293b" strokeWidth="2"/>

                {/* Data points */}
                {companies.map((company, index) => {
                  const xValue = company[xAxis];
                  const yValue = company[yAxis];
                  
                  // Skip if values are invalid
                  if (isNaN(xValue) || isNaN(yValue)) return null;

                  return (
                    <g key={`${company.name}-${index}`}>
                      <circle
                        cx={scaleX(xValue)}
                        cy={scaleY(yValue)}
                        r={getCompanyColor(company) === "#3b82f6" ? 8 : 6}
                        fill={getCompanyColor(company)}
                        stroke="white"
                        strokeWidth="2"
                        className="cursor-pointer transition-all duration-200"
                        onMouseEnter={() => setHoveredCompany(company)}
                        onMouseLeave={() => setHoveredCompany(null)}
                        style={{
                          filter: hoveredCompany === company ? 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))' : 'none',
                          transform: hoveredCompany === company ? 'scale(1.2)' : 'scale(1)',
                          transformOrigin: `${scaleX(xValue)}px ${scaleY(yValue)}px`
                        }}
                      />
                      {getCompanyColor(company) === "#3b82f6" && (
                        <text
                          x={scaleX(xValue)}
                          y={scaleY(yValue) - 15}
                          textAnchor="middle"
                          className="text-sm font-bold fill-blue-600"
                        >
                          You
                        </text>
                      )}
                    </g>
                  );
                })}

                {/* Axis labels */}
                <text x={chartWidth/2} y={chartHeight - 10} textAnchor="middle" className="text-sm font-semibold fill-slate-700">
                  {selectedXFeature.label} {selectedXFeature.unit && `(${selectedXFeature.unit})`}
                </text>
                <text x={20} y={chartHeight/2} textAnchor="middle" className="text-sm font-semibold fill-slate-700" transform={`rotate(-90 20 ${chartHeight/2})`}>
                  {selectedYFeature.label} {selectedYFeature.unit && `(${selectedYFeature.unit})`}
                </text>

                {/* Axis ticks and values */}
                {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
                  const xVal = xRange.min + (xRange.max - xRange.min) * ratio;
                  const yVal = yRange.min + (yRange.max - yRange.min) * ratio;
                  return (
                    <g key={ratio}>
                      {/* X axis ticks */}
                      <line x1={scaleX(xVal)} y1={chartHeight - padding} x2={scaleX(xVal)} y2={chartHeight - padding + 5} stroke="#1e293b"/>
                      <text x={scaleX(xVal)} y={chartHeight - padding + 20} textAnchor="middle" className="text-xs fill-slate-600">
                        {Math.round(xVal * 10) / 10}
                      </text>
                      
                      {/* Y axis ticks */}
                      <line x1={padding} y1={scaleY(yVal)} x2={padding - 5} y2={scaleY(yVal)} stroke="#1e293b"/>
                      <text x={padding - 10} y={scaleY(yVal) + 4} textAnchor="end" className="text-xs fill-slate-600">
                        {Math.round(yVal * 10) / 10}
                      </text>
                    </g>
                  );
                })}
              </svg>

              {/* Tooltip */}
              {hoveredCompany && (
                <div className="absolute bg-slate-800 text-white p-3 rounded-lg shadow-lg pointer-events-none z-10 transform -translate-x-1/2 -translate-y-full"
                     style={{
                       left: scaleX(hoveredCompany[xAxis]),
                       top: scaleY(hoveredCompany[yAxis]) - 10
                     }}>
                  <div className="font-semibold">{hoveredCompany.name}</div>
                  <div className="text-sm">
                    {selectedXFeature.label}: {hoveredCompany[xAxis]}{selectedXFeature.unit}
                  </div>
                  <div className="text-sm">
                    {selectedYFeature.label}: {hoveredCompany[yAxis]}{selectedYFeature.unit}
                  </div>
                  <div className="text-sm text-slate-300 mt-1">
                    Quadrant: {getQuadrantLabel(hoveredCompany[xAxis], hoveredCompany[yAxis])}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="flex justify-center mt-6 space-x-6">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-500 rounded-full border-2 border-white"></div>
              <span className="text-sm font-medium text-slate-700">Your Company</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-slate-400 rounded-full border-2 border-white"></div>
              <span className="text-sm font-medium text-slate-700">Competitors</span>
            </div>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
          {["Leaders", "Challengers", "Visionaries", "Niche Players"].map(quadrant => {
            const companiesInQuadrant = companies.filter(company => {
              const xVal = company[xAxis];
              const yVal = company[yAxis];
              if (isNaN(xVal) || isNaN(yVal)) return false;
              
              return getQuadrantLabel(xVal, yVal) === quadrant;
            });

            const yourCompanyInQuadrant = companiesInQuadrant.some(c => 
              c.name.toLowerCase().includes("your") || c.name.toLowerCase().includes("our")
            );

            return (
              <div key={quadrant} className="bg-white rounded-lg shadow-md p-4">
                <h3 className="font-semibold text-slate-800 mb-2">{quadrant}</h3>
                <div className="text-2xl font-bold text-blue-600 mb-1">{companiesInQuadrant.length}</div>
                <div className="text-sm text-slate-600">
                  {companiesInQuadrant.length === 1 ? 'company' : 'companies'}
                </div>
                {yourCompanyInQuadrant && (
                  <div className="mt-2 text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                    You are here
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default CompetitiveAnalysisDashboard;