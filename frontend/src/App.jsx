import React, { useState, useEffect, useRef } from 'react';
import AIWorkspace from './AIWorkspace';
import DataPlayground from './DataPlayground';
import Workspace from './Workspace';

// Steps mirrored from Workspace.jsx for context pills in the toggle bar
const STEPS = [
  "Upload Dataset", "Preview", "Analysis", "Target & Problem",
  "Remove Unnecessary Columns", "Preprocessing", "Model Selection", "Parameters", "Train",
  "Evaluation", "Predict & Download"
];

// Interactive 3D Particle/Neural Mesh Canvas
function Interactive3DCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    let width = (canvas.width = canvas.parentElement.offsetWidth);
    let height = (canvas.height = canvas.parentElement.offsetHeight);

    const handleResize = () => {
      if (!canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.offsetWidth;
      height = canvas.height = canvas.parentElement.offsetHeight;
    };
    window.addEventListener('resize', handleResize);

    // 3D Nodes
    const NODE_COUNT = 45;
    const nodes = [];
    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push({
        x: (Math.random() - 0.5) * 600,
        y: (Math.random() - 0.5) * 450,
        z: (Math.random() - 0.5) * 500,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        vz: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 2.5 + 1.5,
      });
    }

    let rotX = 0;
    let rotY = 0;
    let targetRotX = 0;
    let targetRotY = 0;

    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - width / 2;
      const mouseY = e.clientY - rect.top - height / 2;
      targetRotY = (mouseX / width) * 0.8;
      targetRotX = -(mouseY / height) * 0.8;
    };
    window.addEventListener('mousemove', handleMouseMove);

    const fov = 400;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      rotX += (targetRotX - rotX) * 0.05 + 0.001;
      rotY += (targetRotY - rotY) * 0.05 + 0.002;

      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);
      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);

      // Project points
      const projected = nodes.map((node) => {
        // Move
        node.x += node.vx;
        node.y += node.vy;
        node.z += node.vz;

        if (Math.abs(node.x) > 300) node.vx *= -1;
        if (Math.abs(node.y) > 225) node.vy *= -1;
        if (Math.abs(node.z) > 250) node.vz *= -1;

        // 3D rotation
        const x1 = node.x * cosY + node.z * sinY;
        const z1 = -node.x * sinY + node.z * cosY;

        const y2 = node.y * cosX - z1 * sinX;
        const z2 = node.y * sinX + z1 * cosX + 450;

        const scale = fov / (fov + z2);
        const px = x1 * scale + width / 2;
        const py = y2 * scale + height / 2;

        return { px, py, scale, z2, radius: node.radius * scale };
      });

      // Draw connections
      for (let i = 0; i < projected.length; i++) {
        for (let j = i + 1; j < projected.length; j++) {
          const p1 = projected[i];
          const p2 = projected[j];
          const dx = p1.px - p2.px;
          const dy = p1.py - p2.py;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 110) {
            const alpha = (1 - dist / 110) * 0.28;
            ctx.beginPath();
            ctx.moveTo(p1.px, p1.py);
            ctx.lineTo(p2.px, p2.py);
            ctx.strokeStyle = `rgba(96, 165, 250, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      projected.forEach((p) => {
        if (p.scale <= 0) return;
        ctx.beginPath();
        ctx.arc(p.px, p.py, Math.max(1, p.radius), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(147, 197, 253, ${Math.min(1, p.scale * 1.2)})`;
        ctx.shadowColor = '#3b82f6';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none opacity-60" />;
}

function App() {
  const [view, setViewState] = useState(() => window.localStorage.getItem('active_workspace_id') ? 'workspace' : 'landing');
  const [handoff, setHandoff] = useState(null);
  // Cross-mode state: whether the Data Playground overlay is visible while in workspace
  const [showPlayground, setShowPlayground] = useState(false);
  // Context shared from Workspace to DataPlayground and the toggle bar
  const [mlContext, setMlContext] = useState(null);
  // External dataset to push into Workspace when user clicks "Use this version in ML"
  const [externalDataset, setExternalDataset] = useState(null);

  const setView = (next) => {
    setViewState(next);
    if (next === 'workspace') window.localStorage.setItem('active_workspace_view', 'workspace');
    if (next === 'data-playground') window.localStorage.setItem('active_workspace_view', 'data-playground');
  };

  /** Called when user clicks "Use this version in ML" from Data Playground while inside workspace */
  const handleWorkspaceHandoff = (result) => {
    // result has { ml_dataset: { id, ... }, ml_workspace: { id, ... } }
    if (result?.ml_dataset) {
      setExternalDataset(result.ml_dataset);
    }
    setShowPlayground(false);
  };

  /** Open playground from within workspace flow; guard if Workspace loading state needs check */
  const openPlayground = () => {
    setShowPlayground(true);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white selection:bg-blue-600 selection:text-white">
      {view === 'data-playground' ? <DataPlayground onBack={() => setView('landing')} onContinueML={(result) => { setHandoff(result); setView('ml-handoff'); }} /> : view === 'ml-handoff' ? <Workspace initialDatasetId={handoff?.ml_dataset?.id} initialWorkspaceId={handoff?.ml_workspace?.id} onBack={() => setView('data-playground')} /> : view === 'landing' ? (
        <div className="relative overflow-hidden">
          {/* Top Navbar */}
          <nav className="relative z-20 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 py-4">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-bold text-lg shadow-md shadow-blue-500/20">
                  ML
                </div>
                <span className="font-extrabold text-xl tracking-tight text-slate-100">
                  No-Code ML
                </span>
              </div>
              <div className="flex items-center gap-4">
                <button onClick={() => setView('data-playground')} className="text-sm font-semibold text-cyan-300 border border-cyan-800 bg-cyan-950/50 px-3 py-2 rounded-lg">Data Playground</button>
                <a href="#features" className="text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors">Features</a>
                <a href="#models" className="text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors">Algorithms</a>
                <a href="#pipeline" className="text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors">Pipeline</a>
                <button
                  onClick={() => setView('workspace')}
                  className="bg-blue-600 hover:bg-blue-500 text-white font-medium py-2 px-5 rounded-lg shadow-md shadow-blue-500/25 transition-all text-sm"
                >
                  Open Workspace &rarr;
                </button>
              </div>
            </div>
          </nav>

          {/* Hero Section with Interactive 3D Canvas */}
          <section className="relative min-h-[85vh] flex items-center justify-center px-4 py-20">
            <Interactive3DCanvas />
            <div className="relative z-10 max-w-4xl mx-auto text-center">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-900/40 border border-blue-500/30 text-blue-300 text-xs font-semibold mb-8 backdrop-blur-sm shadow-sm">
                <span className="w-2 h-2 rounded-full bg-blue-400"></span>
                A considered workspace for applied machine learning
              </div>

              <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 leading-tight">
                Train, Evaluate & Deploy{' '}
                <span className="text-blue-400">
                  Scikit-Learn Models
                </span>{' '}
                Without Code
              </h1>

              <p className="text-lg md:text-xl text-slate-300 max-w-2xl mx-auto mb-10 leading-relaxed">
                Explore your data, shape a reproducible pipeline, train real models, and compare results — all in a clear visual workflow.
              </p>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <button
                  onClick={() => setView('workspace')}
                  className="w-full sm:w-auto bg-blue-600 hover:bg-blue-500 text-white font-bold py-4 px-9 rounded-xl shadow-xl shadow-blue-600/30 hover:scale-[1.02] active:scale-[0.98] transition-all text-lg flex items-center justify-center gap-3"
                >
                  <span>Launch ML Workspace</span>
                  <span>&rarr;</span>
                </button>
                <a
                  href="#pipeline"
                  className="w-full sm:w-auto bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-300 font-semibold py-4 px-8 rounded-xl transition-all text-lg backdrop-blur-sm"
                >
                  Explore Workflow
                </a>
              </div>

              {/* Trust & Metric Badges */}
              <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4 text-left">
                <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/70 p-4 rounded-xl">
                  <div className="text-2xl font-black text-blue-400">6</div>
                  <div className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Core Estimators</div>
                  <div className="text-xs text-slate-500 mt-1">Classification & Regression</div>
                </div>
                <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/70 p-4 rounded-xl">
                  <div className="text-2xl font-black text-emerald-400">0%</div>
                  <div className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Data Leakage</div>
                  <div className="text-xs text-slate-500 mt-1">Strict Train/Test Isolation</div>
                </div>
                <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/70 p-4 rounded-xl">
                  <div className="text-2xl font-black text-indigo-400">Auto</div>
                  <div className="text-xs text-slate-400 uppercase tracking-wider font-semibold">SMOTE & VIF</div>
                  <div className="text-xs text-slate-500 mt-1">Imbalance & Collinearity</div>
                </div>
                <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/70 p-4 rounded-xl">
                  <div className="text-2xl font-black text-purple-400">100%</div>
                  <div className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Reproducible</div>
                  <div className="text-xs text-slate-500 mt-1">Joblib & JSON Artifacts</div>
                </div>
              </div>
            </div>
          </section>

          {/* Workflow Steps Preview */}
          <section id="pipeline" className="py-20 border-t border-slate-800 bg-slate-900/60">
            <div className="max-w-7xl mx-auto px-6">
              <div className="text-center max-w-2xl mx-auto mb-16">
                <h2 className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-2">Architected Pipeline</h2>
                <h3 className="text-3xl md:text-4xl font-extrabold text-slate-100">
                  A Complete 10-Step Machine Learning Lifecycle
                </h3>
                <p className="text-slate-400 mt-4">
                  From raw CSV upload to inference predictions and model artifact downloads with full mathematical integrity.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
                {[
                  { step: '01', title: 'Upload Dataset', desc: 'CSV format validation, row & column inspection.' },
                  { step: '02', title: 'Preview & Shape', desc: 'Inspect first 20 rows, types, and column schema.' },
                  { step: '03', title: 'Deep EDA & VIF', desc: 'Summary statistics, correlations, VIF & YData Profiling.' },
                  { step: '04', title: 'Target Selection', desc: 'Designate target column and auto-detect problem type.' },
                  { step: '05', title: 'Preprocessing', desc: 'Imputation, scaling, outlier clipping & SMOTE.' },
                  { step: '06', title: 'Model Selection', desc: 'Choose from 6 verified scikit-learn estimators.' },
                  { step: '07', title: 'Hyperparameters', desc: 'Type-safe parameter tuning with min/max validation.' },
                  { step: '08', title: 'Model Training', desc: 'Train/test split first, fit strictly on training fold.' },
                  { step: '09', title: 'Evaluation & Runs', desc: 'Confusion matrix, precision, recall, F1, MAE, RMSE & run history.' },
                  { step: '10', title: 'Predict & Export', desc: 'Batch inference with saved pipelines & artifact bundles.' },
                ].map((item) => (
                  <div
                    key={item.step}
                    className="bg-slate-800/80 border border-slate-700/80 rounded-xl p-5 hover:border-blue-500/50 hover:bg-slate-800 transition-all group"
                  >
                    <div className="text-xs font-mono font-bold text-blue-400 mb-2 group-hover:text-blue-300">
                      STEP {item.step}
                    </div>
                    <h4 className="font-bold text-slate-200 text-base mb-1">{item.title}</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Core Feature Deep-Dive */}
          <section id="features" className="py-20 border-t border-slate-800">
            <div className="max-w-7xl mx-auto px-6">
              <div className="text-center max-w-2xl mx-auto mb-16">
                <h2 className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-2">Core Capabilities</h2>
                <h3 className="text-3xl md:text-4xl font-extrabold text-slate-100">
                  Engineered for Reliability and ML Safety
                </h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="bg-slate-800 p-8 rounded-2xl border border-slate-700 relative overflow-hidden">
                  <div className="text-3xl mb-4">🛡️</div>
                  <h4 className="text-xl font-bold mb-3 text-slate-100">Zero Data Leakage</h4>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Splitting occurs strictly before any transformers are fitted. Test splits and inference batches are transformed via saved preprocessor objects without refitting.
                  </p>
                  <span className="text-xs font-mono text-blue-400 bg-blue-900/30 px-2.5 py-1 rounded border border-blue-800/50">
                    sklearn ColumnTransformer
                  </span>
                </div>

                <div className="bg-slate-800 p-8 rounded-2xl border border-slate-700 relative overflow-hidden">
                  <div className="text-3xl mb-4">📊</div>
                  <h4 className="text-xl font-bold mb-3 text-slate-100">Multicollinearity & VIF</h4>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Full Variance Inflation Factor (VIF) calculation with statsmodels. Alerts for moderate (VIF &gt; 5) and severe (VIF &gt; 10) collinearity before model training.
                  </p>
                  <span className="text-xs font-mono text-emerald-400 bg-emerald-900/30 px-2.5 py-1 rounded border border-emerald-800/50">
                    statsmodels OLS VIF
                  </span>
                </div>

                <div className="bg-slate-800 p-8 rounded-2xl border border-slate-700 relative overflow-hidden">
                  <div className="text-3xl mb-4">⚖️</div>
                  <h4 className="text-xl font-bold mb-3 text-slate-100">SMOTE Class Balancing</h4>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Synthetic Minority Oversampling Technique (SMOTE) automatically balances classification classes strictly on the training fold, preserving authentic test distribution.
                  </p>
                  <span className="text-xs font-mono text-purple-400 bg-purple-900/30 px-2.5 py-1 rounded border border-purple-800/50">
                    imblearn SMOTE
                  </span>
                </div>

                <div className="bg-slate-800 p-8 rounded-2xl border border-slate-700 relative overflow-hidden">
                  <div className="text-3xl mb-4">📑</div>
                  <h4 className="text-xl font-bold mb-3 text-slate-100">YData Profiling Engine</h4>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Generate exhaustive interactive HTML profiling reports directly from your dataset. View correlations, missing matrices, and duplicate alerts in one click.
                  </p>
                  <span className="text-xs font-mono text-sky-400 bg-sky-900/30 px-2.5 py-1 rounded border border-sky-800/50">
                    ydata-profiling HTML
                  </span>
                </div>

                <div className="bg-slate-800 p-8 rounded-2xl border border-slate-700 relative overflow-hidden">
                  <div className="text-3xl mb-4">🕒</div>
                  <h4 className="text-xl font-bold mb-3 text-slate-100">Multi-Run Training History</h4>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Compare multiple model runs side-by-side. Track accuracy, F1, RMSE, and hyperparameters across iterations without losing previous benchmarks.
                  </p>
                  <span className="text-xs font-mono text-indigo-400 bg-indigo-900/30 px-2.5 py-1 rounded border border-indigo-800/50">
                    Persistent Run Ledgers
                  </span>
                </div>

                <div className="bg-slate-800 p-8 rounded-2xl border border-slate-700 relative overflow-hidden">
                  <div className="text-3xl mb-4">📦</div>
                  <h4 className="text-xl font-bold mb-3 text-slate-100">Traceable Artifact Store</h4>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Download serialized estimators (<code className="text-slate-300">model.joblib</code>), fitted transformers, config records (<code className="text-slate-300">config.json</code>), and predictions CSV.
                  </p>
                  <span className="text-xs font-mono text-amber-400 bg-amber-900/30 px-2.5 py-1 rounded border border-amber-800/50">
                    joblib & CSV Serialization
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* Model Registry Matrix */}
          <section id="models" className="py-20 border-t border-slate-800 bg-slate-900/60">
            <div className="max-w-7xl mx-auto px-6">
              <div className="text-center max-w-2xl mx-auto mb-16">
                <h2 className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-2">Model Registry</h2>
                <h3 className="text-3xl md:text-4xl font-extrabold text-slate-100">
                  Supported Machine Learning Algorithms
                </h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Classification */}
                <div className="bg-slate-800/90 rounded-2xl p-6 border border-slate-700">
                  <div className="flex items-center gap-3 mb-6">
                    <span className="px-3 py-1 rounded-md bg-blue-900/60 text-blue-400 text-xs font-bold uppercase tracking-wider">
                      Classification
                    </span>
                    <h4 className="text-xl font-bold text-white">Categorical & Label Targets</h4>
                  </div>
                  <div className="space-y-4">
                    <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-700/60">
                      <div className="font-bold text-blue-300">Logistic Regression</div>
                      <p className="text-xs text-slate-400 mt-1">Linear classifier with L2 regularization. Configurable C and max_iter parameters.</p>
                    </div>
                    <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-700/60">
                      <div className="font-bold text-blue-300">Decision Tree Classifier</div>
                      <p className="text-xs text-slate-400 mt-1">Non-parametric tree with max_depth, min_samples_split, and min_samples_leaf controls.</p>
                    </div>
                    <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-700/60">
                      <div className="font-bold text-blue-300">Random Forest Classifier</div>
                      <p className="text-xs text-slate-400 mt-1">Ensemble bagging of decision trees with n_estimators and pruning parameters.</p>
                    </div>
                  </div>
                </div>

                {/* Regression */}
                <div className="bg-slate-800/90 rounded-2xl p-6 border border-slate-700">
                  <div className="flex items-center gap-3 mb-6">
                    <span className="px-3 py-1 rounded-md bg-emerald-900/60 text-emerald-400 text-xs font-bold uppercase tracking-wider">
                      Regression
                    </span>
                    <h4 className="text-xl font-bold text-white">Continuous Numerical Targets</h4>
                  </div>
                  <div className="space-y-4">
                    <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-700/60">
                      <div className="font-bold text-emerald-300">Linear Regression</div>
                      <p className="text-xs text-slate-400 mt-1">Ordinary least squares linear model with fit_intercept toggle.</p>
                    </div>
                    <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-700/60">
                      <div className="font-bold text-emerald-300">Decision Tree Regressor</div>
                      <p className="text-xs text-slate-400 mt-1">Tree regressor predicting continuous values with configurable depth and splitting rules.</p>
                    </div>
                    <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-700/60">
                      <div className="font-bold text-emerald-300">Random Forest Regressor</div>
                      <p className="text-xs text-slate-400 mt-1">Averaged ensemble regression over randomized trees for nonlinear continuous estimation.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* CTA Footer Section */}
          <section className="py-20 border-t border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 text-center px-6">
            <div className="max-w-3xl mx-auto">
              <h3 className="text-3xl md:text-4xl font-extrabold mb-6">Ready to Train Your First Model?</h3>
              <p className="text-slate-400 mb-8">
                No setup, no environment debugging, and no ML syntax required. Simply upload your CSV file to get started.
              </p>
              <button
                onClick={() => setView('workspace')}
                className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-4 px-10 rounded-xl shadow-xl shadow-blue-500/25 transition-all text-lg"
              >
                Launch Workspace
              </button>
            </div>
          </section>

          {/* Footer */}
          <footer className="border-t border-slate-800 py-8 px-6 text-center text-sm text-slate-500">
            <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
              <div>No-Code Machine Learning Platform &bull; Built with FastAPI, scikit-learn &amp; React</div>
              <div className="flex gap-4">
                <a href="/docs" target="_blank" rel="noreferrer" className="text-slate-400 hover:text-white transition-colors">
                  API Docs (Swagger)
                </a>
                <span className="text-slate-700">|</span>
                <span className="text-slate-400">Production Ready</span>
              </div>
            </div>
          </footer>
        </div>
      ) : (
        /* Workspace mode: render both AIWorkspace and DataPlayground simultaneously
           using CSS display toggling so Workspace never unmounts (training polling continues). */
        <div className="ml-dp-wrapper">
          {/* Persistent mode toggle bar — always visible in workspace mode */}
          <div className="ml-dp-toggle-bar">
            <button
              className={`ml-dp-mode-btn ${!showPlayground ? 'active' : 'inactive'}`}
              onClick={() => setShowPlayground(false)}
            >
              ⚙ ML Workflow
            </button>
            <button
              className={`ml-dp-mode-btn ${showPlayground ? 'active' : 'inactive'}`}
              onClick={openPlayground}
            >
              ▤ Data Playground
            </button>
            {mlContext?.trainingActive && (
              <span className="ml-dp-training-badge">
                <span className="dot" />
                {mlContext.modelName || 'Model'} training…
              </span>
            )}
            {mlContext?.datasetName && (
              <span className="ml-dp-dataset-pill">▤ {mlContext.datasetName} · v{mlContext.version || 1}</span>
            )}
            {mlContext?.step >= 0 && (
              <span className="ml-dp-step-pill">Step {(mlContext.step || 0) + 1}: {STEPS[mlContext.step || 0]}</span>
            )}
          </div>

          {/* AI Workspace — always mounted; hidden with CSS when playground is active */}
          <div className="ml-dp-pane" style={{ display: showPlayground ? 'none' : 'block' }}>
            <AIWorkspace
              onOpenPlayground={openPlayground}
              onContextUpdate={setMlContext}
              onDatasetHandoff={handleWorkspaceHandoff}
              externalDataset={externalDataset}
            />
          </div>

          {/* Data Playground — always mounted; hidden with CSS when ML is active */}
          <div className="ml-dp-pane" style={{ display: showPlayground ? 'block' : 'none' }}>
            <DataPlayground
              onBack={() => setShowPlayground(false)}
              onBackToML={() => setShowPlayground(false)}
              mlContext={mlContext}
              onContinueML={handleWorkspaceHandoff}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
