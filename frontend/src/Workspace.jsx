import React, { useState, useEffect, useRef } from 'react';
import * as api from './api';
import AICompanion from './AICompanion';

const STEPS = [
  "Upload Dataset", "Preview", "Analysis", "Target & Problem",
  "Remove Unnecessary Columns", "Preprocessing", "Model Selection", "Parameters", "Train",
  "Evaluation", "Predict & Download"
];

export default function Workspace({ initialGoal = '', initialFile = null, initialWorkspaceId = null, initialDatasetId = null, onOpenPlayground, onContextUpdate, onDatasetHandoff, externalDataset }) {
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Data states
  const [datasetId, setDatasetId] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [columnStats, setColumnStats] = useState([]);
  const [retainedColumns, setRetainedColumns] = useState([]);
  const [columnSearch, setColumnSearch] = useState('');
  const [columnBusy, setColumnBusy] = useState(false);
  const [preview, setPreview] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [correlation, setCorrelation] = useState(null);
  const [vifData, setVifData] = useState(null);
  const [profilingStatus, setProfilingStatus] = useState(null);
  const [workspaceId, setWorkspaceId] = useState(null);
  const [target, setTarget] = useState('');
  const [problemType, setProblemType] = useState('classification');
  
  const [preprocessing, setPreprocessing] = useState({
    missing_values: 'mean',
    scaling: 'none',
    outlier_handling: false
  });
  
  const [availableModels, setAvailableModels] = useState([]);
  const [modelId, setModelId] = useState('');
  const [modelParams, setModelParams] = useState({});
  const [selectedModelObj, setSelectedModelObj] = useState(null);
  
  const [trainStatus, setTrainStatus] = useState(null);
  const [trainingElapsed, setTrainingElapsed] = useState(0);
  const [trainMode, setTrainMode] = useState('manual'); // 'manual' | 'optuna'
  const [optunaConfig, setOptunaConfig] = useState({
    strategy: 'optuna',
    candidate_models: [],
    metric: 'accuracy',
    n_trials: 10,
    cv_folds: 3,
    timeout: 60,
  });
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [runs, setRuns] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [predictionUrl, setPredictionUrl] = useState(null);
  const [predictionWarnings, setPredictionWarnings] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [aiPopulated, setAiPopulated] = useState(false);
  const initialUploadStarted = useRef(false);
  const trainingActive = trainStatus && !['COMPLETED', 'FAILED', 'CANCELLED'].includes(trainStatus.status);

  // Notify parent of key state changes (for cross-mode context pills)
  useEffect(() => {
    if (!onContextUpdate) return;
    onContextUpdate({
      datasetId,
      datasetName: datasetInfo?.filename || datasetInfo?.name || null,
      version: 1,
      trainingActive,
      modelName: selectedModelObj?.name || modelId || null,
      step,
    });
  }, [datasetId, datasetInfo, trainingActive, selectedModelObj, modelId, step]);

  // Toast state for external dataset handoff notification
  const [externalDatasetToast, setExternalDatasetToast] = useState('');

  // When a new external dataset is handed off from Data Playground, update our state
  useEffect(() => {
    if (!externalDataset?.id || externalDataset.id === datasetId) return;
    let active = true;
    (async () => {
      try {
        const ds = await api.getDataset(externalDataset.id);
        const ws = await api.createWorkspace(externalDataset.id);
        if (!active) return;
        setDatasetId(externalDataset.id);
        setDatasetInfo(ds);
        setWorkspaceId(ws.id);
        window.localStorage.setItem('active_workspace_id', ws.id);
        // Reset ML-specific state so the user starts fresh with the new dataset
        setTarget('');
        setPreprocessing({ missing_values: 'mean', scaling: 'none', outlier_handling: false });
        setModelId('');
        setModelParams({});
        setSelectedModelObj(null);
        setEvaluation(null);
        setPredictions(null);
        setStep(1);
        const versionNum = externalDataset.version || externalDataset.current_version || 1;
        setExternalDatasetToast(`Dataset updated to v${versionNum} from Data Playground`);
        window.setTimeout(() => setExternalDatasetToast(''), 4000);
      } catch (e) {
        setError(`Could not load dataset from Data Playground: ${e.message}`);
      }
    })();
    return () => { active = false; };
  }, [externalDataset?.id]);

  useEffect(() => {
    if (!workspaceId) return undefined;
    const savedJobId = window.localStorage.getItem(`training_job_${workspaceId}`);
    if (!savedJobId) return undefined;
    let active = true;
    const poll = async () => {
      try {
        const workspace = await api.getWorkspace(workspaceId);
        const dataset = await api.getDataset(workspace.dataset_id);
        const modelResponse = await api.getModels(workspace.problem_type || 'classification');
        if (!active) return;
        setDatasetId(workspace.dataset_id);
        setDatasetInfo(dataset);
        setTarget(workspace.target || '');
        setProblemType(workspace.problem_type || 'classification');
        setPreprocessing(current => ({ ...current, ...(workspace.preprocessing || {}), scaling: workspace.preprocessing?.scaling || 'none' }));
        setModelId(workspace.model_id || '');
        setModelParams(workspace.model_params || {});
        const models = modelResponse.models || [];
        setAvailableModels(models);
        setSelectedModelObj(models.find(model => model.model_id === workspace.model_id) || null);
        setStep(8);
        while (active) {
          const response = await api.getTrainingJob(savedJobId);
          const job = response.job;
          if (!active) return;
          setTrainStatus(job);
          if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(job.status)) {
            window.localStorage.removeItem(`training_job_${workspaceId}`);
            if (job.status === 'COMPLETED') {
              const [evaluationResult, runsResult] = await Promise.all([
                api.getEvaluation(workspaceId),
                api.getRuns(workspaceId).catch(() => ({ runs: [] })),
              ]);
              if (!active) return;
              setEvaluation(evaluationResult);
              setRuns(runsResult.runs || []);
              setTrainStatus(current => ({ ...current, result: job.result }));
              setStep(current => current === 8 ? 9 : current);
            }
            return;
          }
          await new Promise(resolve => window.setTimeout(resolve, 900));
        }
      } catch (pollError) {
        if (active) setError(`Could not check training status: ${pollError.message}`);
      }
    };
    void poll();
    return () => { active = false; };
  }, [workspaceId, trainStatus?.job_id]);

  useEffect(() => {
    if (!trainStatus?.started_at || ['COMPLETED', 'FAILED', 'CANCELLED'].includes(trainStatus.status)) return undefined;
    const updateElapsed = () => setTrainingElapsed(Math.max(0, Math.floor((Date.now() - Date.parse(trainStatus.started_at)) / 1000)));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [trainStatus?.started_at, trainStatus?.status]);

  // Helpers
  const goNext = () => setStep(s => Math.min(s + 1, STEPS.length - 1));
  const goPrev = () => setStep(s => Math.max(s - 1, 0));
  const gotoStep = (s) => setStep(s);

  const handleError = (e) => {
    setError(e.message || String(e));
    setLoading(false);
  };

  useEffect(() => {
    if (!initialWorkspaceId || workspaceId) return;
    let active = true;
    (async () => {
      const ws = await api.getWorkspace(initialWorkspaceId);
      const ds = await api.getDataset(ws.dataset_id);
      if (!active) return;
      setWorkspaceId(ws.id);
      setDatasetId(ws.dataset_id);
      setDatasetInfo(ds);
      setStep(1);
    })().catch(handleError);
    return () => { active = false; };
  }, [initialWorkspaceId, workspaceId]);

  useEffect(() => {
    if (!initialDatasetId || datasetId || initialWorkspaceId) return;
    let active = true;
    (async () => {
      const ds = await api.getDataset(initialDatasetId);
      const ws = await api.createWorkspace(initialDatasetId);
      if (!active) return;
      setDatasetId(initialDatasetId);
      setDatasetInfo(ds);
      setWorkspaceId(ws.id);
      setStep(1);
    })().catch(handleError);
    return () => { active = false; };
  }, [initialDatasetId, datasetId]);

  const uploadDatasetFile = async (file) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.uploadDataset(file);
      setDatasetId(res.id);
      setDatasetInfo(res);
      
      // Auto-create workspace
      const wsRes = await api.createWorkspace(res.id);
      setWorkspaceId(wsRes.id);
      window.localStorage.setItem('active_workspace_id', wsRes.id);
      
      setLoading(false);
      goNext();
    } catch (e) {
      handleError(e);
    }
  };

  const handleUploadDataset = (e) => uploadDatasetFile(e.target.files?.[0]);

  useEffect(() => {
    if (!initialFile || initialUploadStarted.current) return;
    initialUploadStarted.current = true;
    uploadDatasetFile(initialFile);
  }, [initialFile]);

  const loadPreview = async () => {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getPreview(datasetId);
      setPreview(res);
      setLoading(false);
    } catch (e) {
      handleError(e);
    }
  };

  const loadAnalysis = async () => {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getAnalysis(datasetId);
      setAnalysis(res);
      setCorrelation(res.correlation);
      setLoading(false);
    } catch (e) {
      handleError(e);
    }
  };

  useEffect(() => {
    if (step !== 4 || !datasetId) return;
    let active = true;
    api.getColumnStatistics(datasetId, target).then(result => {
      if (!active) return;
      setColumnStats(result.columns || []);
      setRetainedColumns(current => current.length ? current.filter(c => result.columns.some(col => col.name === c)) : (result.columns || []).map(col => col.name));
    }).catch(handleError);
    return () => { active = false; };
  }, [step, datasetId, target]);

  const loadModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getModels(problemType);
      setAvailableModels(res.models);
      setLoading(false);
      goNext();
    } catch(e) {
      handleError(e);
    }
  };

  const handleSelectModel = (model) => {
    setModelId(model.model_id);
    setSelectedModelObj(model);
    // Initialize default params
    const initialParams = {};
    model.parameters?.forEach(p => {
      initialParams[p.name] = p.default;
    });
    setModelParams(initialParams);
  };

  const fillAIConfiguration = async (proposal) => {
    const configuration = proposal.configuration || {};
    if (proposal.stage_id === 'model_configuration' || proposal.stage_id === 'training' || proposal.stage_id === 'optimization') {
      const config = configuration.configuration || configuration;
      if (config.target) setTarget(config.target);
      if (config.problem_type) setProblemType(config.problem_type);
      if (config.preprocessing) setPreprocessing(current => ({ ...current, ...config.preprocessing, scaling: config.preprocessing.scaling || 'none' }));
      if (config.model_id) {
        setModelId(config.model_id);
        let models = availableModels;
        if (!models.some(model => model.model_id === config.model_id)) {
          try { const result = await api.getModels(config.problem_type || problemType); models = result.models || []; setAvailableModels(models); } catch (_) { /* form remains editable when model metadata is unavailable */ }
        }
        const selected = models.find(model => model.model_id === config.model_id);
        if (selected) setSelectedModelObj(selected);
      }
      if (config.model_params) setModelParams(config.model_params);
      setStep(config.model_params ? 7 : config.model_id ? 6 : 5);
    }
    if (proposal.stage_id === 'column_selection' && configuration.retained_columns) {
      setRetainedColumns(configuration.retained_columns); setStep(4);
    }
    setAiPopulated(true);
  };

  const handleTrain = async () => {
    if (!workspaceId) return;
    setLoading(true);
    setError(null);
    try {
      // First configure
      const prepConfig = {
        ...preprocessing,
        scaling: preprocessing.scaling === 'none' ? null : preprocessing.scaling
      };
      await api.configureWorkspace(workspaceId, {
        target,
        problem_type: problemType,
        preprocessing: prepConfig,
        model_id: modelId,
        model_params: modelParams
      });

      // Then train
      const { job } = await api.startTrainingJob(workspaceId);
      window.localStorage.setItem(`training_job_${workspaceId}`, job.job_id);
      setTrainStatus(job);
      setLoading(false);
    } catch(e) {
      handleError(e);
    }
  };

  const handleCancelTraining = async () => {
    if (!trainStatus?.job_id || trainStatus.status === 'CANCELLING') return;
    if (!window.confirm('Cancel this training job? The in-progress run will be discarded; previously completed models will be kept.')) return;
    try {
      const { job } = await api.cancelTrainingJob(trainStatus.job_id);
      setTrainStatus(job);
    } catch (cancelError) {
      setError(`Could not cancel training: ${cancelError.message}`);
    }
  };

  const handleOptimize = async () => {
    if (!workspaceId) return;
    setLoading(true);
    setError(null);
    try {
      const prepConfig = {
        ...preprocessing,
        scaling: preprocessing.scaling === 'none' ? null : preprocessing.scaling
      };
      await api.configureWorkspace(workspaceId, {
        target,
        problem_type: problemType,
        preprocessing: prepConfig,
        model_id: modelId || availableModels[0]?.model_id,
        model_params: modelParams || {}
      });

      const candidateModels = optunaConfig.candidate_models.length > 0
        ? optunaConfig.candidate_models
        : (modelId ? [modelId] : availableModels.map(m => m.model_id));

      const res = await api.optimizeWorkspace(workspaceId, {
        strategy: optunaConfig.strategy,
        candidate_models: candidateModels,
        metric: optunaConfig.metric,
        n_trials: optunaConfig.n_trials,
        cv_folds: optunaConfig.cv_folds,
        timeout: optunaConfig.timeout,
        n_iter: optunaConfig.n_iter || 10,
      });

      setOptimizationResult(res.optimization);
      setTrainStatus({ run_id: res.run_id, status: 'trained' });

      const evalRes = await api.getEvaluation(workspaceId);
      setEvaluation(evalRes);

      try {
        const runsRes = await api.getRuns(workspaceId);
        setRuns(runsRes.runs || []);
      } catch (err) {
        console.error('Failed to load runs', err);
      }

      setLoading(false);
      goNext();
    } catch(e) {
      handleError(e);
    }
  };

  const handlePredictUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file || !workspaceId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictWorkspace(workspaceId, file);
      setPredictions(res.predictions);
      setPredictionUrl(res.download_url);
      setPredictionWarnings(res.warnings || []);
      
      const artRes = await api.getArtifacts(workspaceId);
      setArtifacts(artRes.artifacts || []);
      
      setLoading(false);
    } catch(e) {
      handleError(e);
    }
  };

  // Step renderers
  const renderStep0 = () => (
    <div className="flex flex-col items-center p-8 bg-slate-800 rounded-lg border border-slate-700">
      <h2 className="text-2xl font-bold mb-4">Upload Dataset</h2>
      <p className="text-slate-400 mb-6">Select a CSV file to get started.</p>
      <input type="file" accept=".csv,text/csv" onChange={handleUploadDataset} className="sr-only peer" id="upload-csv" />
      <label htmlFor="upload-csv" className="cursor-pointer bg-blue-600 hover:bg-blue-500 px-6 py-3 rounded-lg font-medium text-white shadow-lg transition-colors peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-4 peer-focus-visible:outline-amber-700">
        {loading ? 'Uploading…' : 'Choose CSV file'}
      </label>
      <p className="text-xs text-slate-500 mt-3">CSV files only · Your original dataset is kept unchanged</p>
    </div>
  );

  const renderStep1 = () => {
    if (!datasetInfo) return <div>No dataset loaded</div>;
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Rows</div>
            <div className="text-2xl font-bold">{datasetInfo.rows}</div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Columns</div>
            <div className="text-2xl font-bold">{datasetInfo.num_columns}</div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Missing Values</div>
            <div className="text-2xl font-bold">{datasetInfo.missing_values ? Object.values(datasetInfo.missing_values).reduce((a,b) => a+b, 0) : 0}</div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Duplicate Rows</div>
            <div className="text-2xl font-bold">{datasetInfo.duplicate_rows}</div>
          </div>
        </div>
        
        <div className="flex justify-between items-center">
          <h3 className="text-xl font-bold">Data Preview</h3>
          <button onClick={() => { loadPreview(); }} className="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg text-sm">
            Load Preview
          </button>
        </div>

        {preview && (
          <div className="overflow-x-auto bg-slate-800 rounded-lg border border-slate-700">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-slate-700 text-slate-300">
                <tr>
                  {preview.columns.map(c => <th key={c} className="p-3 font-medium border-b border-slate-600">{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {preview.data.map((row, i) => (
                  <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    {preview.columns.map(c => <td key={c} className="p-3">{row[c]?.toString()}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="flex justify-end">
          <button onClick={() => { loadAnalysis(); goNext(); }} className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-medium">Continue</button>
        </div>
      </div>
    );
  };

  const renderStep2 = () => {
    if (!analysis) return <div className="text-center py-8"><button onClick={loadAnalysis} className="bg-blue-600 px-4 py-2 rounded-lg">Load Analysis</button></div>;
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
            <div className="text-sm text-slate-400">Numerical Columns</div>
            <div className="text-2xl font-bold text-blue-400">{analysis.numerical_columns.length}</div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
            <div className="text-sm text-slate-400">Categorical Columns</div>
            <div className="text-2xl font-bold text-indigo-400">{analysis.categorical_columns.length}</div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
            <div className="text-sm text-slate-400">Total Missing</div>
            <div className="text-2xl font-bold text-red-400">{analysis.missing_total}</div>
          </div>
        </div>

        <h3 className="text-xl font-bold">Column Statistics</h3>
        <div className="overflow-x-auto bg-slate-800 rounded-lg border border-slate-700">
          <table className="w-full text-left text-sm whitespace-nowrap">
            <thead className="bg-slate-700 text-slate-300">
              <tr>
                <th className="p-3">Name</th>
                <th className="p-3">Type</th>
                <th className="p-3">Missing</th>
                <th className="p-3">Missing %</th>
                <th className="p-3">Unique</th>
                <th className="p-3">Mean</th>
                <th className="p-3">Min/Max</th>
              </tr>
            </thead>
            <tbody>
              {analysis.column_stats.map(col => (
                <tr key={col.name} className="border-b border-slate-700/50">
                  <td className="p-3 font-medium">
                    {col.name}
                    {col.warning && <span className="ml-2 px-2 py-0.5 text-xs bg-yellow-900/50 text-yellow-500 rounded-full">{col.warning}</span>}
                  </td>
                  <td className="p-3 text-slate-400">{col.dtype}</td>
                  <td className="p-3">{col.missing}</td>
                  <td className="p-3">{col.missing_pct}%</td>
                  <td className="p-3">{col.unique}</td>
                  <td className="p-3">{col.mean != null ? Number(col.mean).toFixed(4) : '-'}</td>
                  <td className="p-3">{col.min != null ? `${Number(col.min).toFixed(2)} / ${Number(col.max).toFixed(2)}` : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {correlation && correlation.columns.length >= 2 ? (
          <div className="mt-8">
            <h3 className="text-xl font-bold mb-4">Correlation Matrix</h3>
            <div className="overflow-x-auto bg-slate-800 rounded-lg border border-slate-700">
              <table className="w-full text-center text-sm border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 border border-slate-700 bg-slate-900"></th>
                    {correlation.columns.map(c => <th key={c} className="p-2 border border-slate-700 bg-slate-700">{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {correlation.matrix.map((row, i) => (
                    <tr key={i}>
                      <th className="p-2 border border-slate-700 bg-slate-700 text-left">{correlation.columns[i]}</th>
                      {row.map((val, j) => {
                        let bgColor = '';
                        if (val > 0.7) bgColor = 'bg-blue-600/50';
                        else if (val < -0.7) bgColor = 'bg-red-600/50';
                        else if (val > 0.3 || val < -0.3) bgColor = 'bg-slate-600/50';
                        else bgColor = 'bg-transparent';
                        return (
                          <td key={j} className={`p-2 border border-slate-700 ${bgColor}`}>
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="mt-8 text-slate-400">Not enough numerical columns for correlation analysis</div>
        )}

        {/* VIF Analysis */}
        <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 mt-8">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold">Feature Multicollinearity (VIF)</h3>
            <button
              onClick={async () => {
                try {
                  const res = await api.getVif(datasetId);
                  setVifData(res);
                } catch (e) { handleError(e); }
              }}
              className="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg text-sm"
            >
              Compute VIF
            </button>
          </div>
          {vifData?.message && <p className="text-slate-400 text-sm">{vifData.message}</p>}
          {vifData?.features?.length > 0 && (
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-700 text-slate-300">
                <tr>
                  <th className="p-3">Feature</th>
                  <th className="p-3">VIF</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {vifData.features.map(f => (
                  <tr key={f.feature} className="border-b border-slate-700/50">
                    <td className="p-3 font-medium">{f.feature}</td>
                    <td className="p-3">{f.vif != null ? f.vif.toFixed(2) : 'N/A'}</td>
                    <td className="p-3">
                      {f.warning ? (
                        <span className={`text-xs px-2 py-1 rounded ${f.warning.includes('High') ? 'bg-red-900/50 text-red-300' : 'bg-yellow-900/50 text-yellow-300'}`}>
                          {f.warning}
                        </span>
                      ) : (
                        <span className="text-xs px-2 py-1 rounded bg-green-900/50 text-green-300">OK</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* YData Profiling */}
        <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 mt-8">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold">Automated EDA Report</h3>
            <button
              onClick={async () => {
                setProfilingStatus('generating');
                setError(null);
                try {
                  const res = await api.generateProfiling(datasetId);
                  setProfilingStatus(res.report_available ? 'ready' : 'error');
                  if (!res.report_available) setError(res.error || 'Profiling failed');
                } catch (e) {
                  setProfilingStatus('error');
                  handleError(e);
                }
              }}
              disabled={profilingStatus === 'generating'}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium"
            >
              {profilingStatus === 'generating' ? (
                <span className="animate-pulse">Generating Report...</span>
              ) : profilingStatus === 'ready' ? (
                'Regenerate Report'
              ) : (
                'Generate EDA Report'
              )}
            </button>
          </div>
          <p className="text-slate-400 text-sm mb-4">Generate an automated exploratory data analysis report using YData Profiling.</p>
          {profilingStatus === 'ready' && (
            <a
              href={api.getProfilingReportUrl(datasetId)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block bg-green-600 hover:bg-green-500 px-4 py-2 rounded-lg text-sm font-medium"
            >
              View Full Report ↗
            </a>
          )}
        </div>

        <div className="flex justify-end mt-6">
          <button onClick={goNext} className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-medium">Continue</button>
        </div>
      </div>
    );
  };

  const renderStep3 = () => {
    return (
      <div className="space-y-6 max-w-xl mx-auto bg-slate-800 p-6 rounded-lg border border-slate-700">
        <h3 className="text-xl font-bold">Target & Problem Type</h3>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Target Column</label>
          <select 
            className={`w-full bg-slate-900 border border-slate-600 rounded-lg p-2 text-white focus:border-blue-500 outline-none ${aiPopulated ? 'ai-populated-field' : ''}`}
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">Select a target column...</option>
            {datasetInfo?.columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Problem Type</label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="problem" value="classification" checked={problemType === 'classification'} onChange={(e)=>setProblemType(e.target.value)} className="w-4 h-4 text-blue-600" />
              <span>Classification</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="problem" value="regression" checked={problemType === 'regression'} onChange={(e)=>setProblemType(e.target.value)} className="w-4 h-4 text-blue-600" />
              <span>Regression</span>
            </label>
          </div>
        </div>

        {target && problemType === 'regression' && analysis?.categorical_columns.includes(target) && (
          <div className="bg-yellow-900/50 border border-yellow-700 text-yellow-200 p-3 rounded-lg text-sm">
            Warning: You selected Regression but the target column appears to be categorical.
          </div>
        )}

        <div className="flex justify-end pt-4">
          <button 
            disabled={!target} 
            onClick={loadModels}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-2 rounded-lg font-medium"
          >
            Find Models
          </button>
        </div>
      </div>
    );
  };

  const renderStep4 = () => {
    const visible = columnStats.filter(column => column.name.toLowerCase().includes(columnSearch.toLowerCase()));
    const toggleColumn = (name, keep) => setRetainedColumns(current => keep
      ? (current.includes(name) ? current : [...current, name])
      : current.filter(column => column !== name));
    const applySelection = async () => {
      if (!workspaceId || !retainedColumns.length) return;
      setColumnBusy(true); setError(null);
      try {
        const result = await api.applyColumnSelection(workspaceId, retainedColumns, target);
        setDatasetId(result.dataset_id);
        setDatasetInfo(current => ({ ...current, id: result.dataset_id, columns: result.retained_columns,
          parent_dataset_id: result.original_dataset_id, num_columns: result.retained_columns.length, dtypes: result.schema, rows: result.rows }));
        setPreview(null); setAnalysis(null); setColumnStats([]);
        goNext();
      } catch (e) { handleError(e); }
      finally { setColumnBusy(false); }
    };
    const restoreOriginal = async () => {
      if (!workspaceId) return;
      setColumnBusy(true); setError(null);
      try {
        const result = await api.restoreColumnSelection(workspaceId);
        setDatasetId(result.dataset_id);
        setDatasetInfo(current => ({ ...current, id: result.dataset_id, parent_dataset_id: null, columns: result.columns, num_columns: result.columns.length }));
        setRetainedColumns(result.columns); setPreview(null); setAnalysis(null);
        setColumnStats((await api.getColumnStatistics(result.dataset_id, target)).columns || []);
      } catch (e) { handleError(e); }
      finally { setColumnBusy(false); }
    };
    return (
      <section className="space-y-5 bg-slate-800 p-5 rounded-lg border border-slate-700">
        <div className="flex flex-wrap justify-between items-start gap-3">
          <div><h3 className="text-xl font-bold">Remove Unnecessary Columns</h3>
            <p className="text-sm text-slate-400 mt-1">Recommendations use column metadata only. Review domain meaning before removing any feature.</p></div>
          <input aria-label="Search columns" value={columnSearch} onChange={e => setColumnSearch(e.target.value)} placeholder="Search columns…" className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm" />
        </div>
        <div className="flex flex-wrap gap-2 text-sm">
          <button onClick={() => setRetainedColumns(columnStats.map(c => c.name))} className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-2">Select all</button>
          <button onClick={() => setRetainedColumns(target ? [target] : [])} className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-2">Deselect all</button>
          <span className="self-center text-slate-300">{retainedColumns.length} retained · {columnStats.length - retainedColumns.length} marked for removal</span>
        </div>
        <div className="overflow-x-auto rounded border border-slate-700">
          <table className="w-full text-left text-xs min-w-[1050px]">
            <thead className="bg-slate-900 text-slate-400"><tr>{['Keep','Column','Type','Missing','Unique','Ratio','Examples','Suggested role','Action'].map(label => <th key={label} className="p-3">{label}</th>)}</tr></thead>
            <tbody className="divide-y divide-slate-700">
              {visible.map(column => {
                const keep = retainedColumns.includes(column.name);
                const targetColumn = column.name === target;
                return <tr key={column.name} className={targetColumn ? 'bg-blue-950/30' : ''}>
                  <td className="p-3"><input aria-label={`Keep ${column.name}`} type="checkbox" checked={keep} disabled={targetColumn} onChange={e => toggleColumn(column.name, e.target.checked)} /></td>
                  <td className="p-3 font-medium">{column.name}{targetColumn && <span className="ml-2 text-blue-300">Target</span>}</td>
                  <td className="p-3">{column.dtype}</td><td className="p-3">{column.missing_pct}%</td>
                  <td className="p-3">{column.unique_count.toLocaleString()}</td><td className="p-3">{(column.unique_ratio * 100).toFixed(1)}%</td>
                  <td className="p-3 max-w-48 truncate" title={column.example_values.join(', ')}>{column.example_values.join(', ') || '—'}</td>
                  <td className="p-3"><span className={column.suggested_role === 'review_remove' ? 'text-amber-300' : column.suggested_role === 'retain' ? 'text-emerald-300' : 'text-slate-300'}>{column.suggested_role.replace('_', ' ')}</span><div className="text-slate-500 mt-1">{column.recommendation_reasons.join('; ')}</div></td>
                  <td className="p-3"><button onClick={() => toggleColumn(column.name, !keep)} disabled={targetColumn} className="underline disabled:opacity-40">{keep ? 'Remove Column' : 'Keep Column'}</button></td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
        <div className="bg-slate-900 rounded p-3 text-sm text-slate-300" aria-live="polite">
          Preview: training will use {retainedColumns.length} of {columnStats.length} columns. Removed: {columnStats.filter(c => !retainedColumns.includes(c.name)).map(c => c.name).join(', ') || 'none'}.
        </div>
        <div className="flex flex-wrap justify-between gap-3">
          <button onClick={restoreOriginal} disabled={columnBusy || !datasetInfo?.parent_dataset_id} className="border border-slate-600 hover:bg-slate-700 disabled:opacity-40 rounded px-4 py-2">Restore Original Dataset</button>
          <button onClick={applySelection} disabled={columnBusy || !retainedColumns.length || (target && !retainedColumns.includes(target)) || (target && retainedColumns.length < 2)} className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded px-5 py-2 font-semibold">{columnBusy ? 'Applying…' : 'Apply Selection'}</button>
        </div>
      </section>
    );
  };

  const renderStep5 = () => {
    return (
      <div className="space-y-6 max-w-xl mx-auto bg-slate-800 p-6 rounded-lg border border-slate-700">
        <h3 className="text-xl font-bold">Preprocessing Configuration</h3>
        
        {selectedModelObj && (
          <div className={`p-3 rounded-lg text-sm mb-4 ${selectedModelObj.needs_scaling ? 'bg-blue-900/30 border border-blue-700 text-blue-200' : 'bg-yellow-900/30 border border-yellow-700 text-yellow-200'}`}>
            {selectedModelObj.needs_scaling 
              ? `${selectedModelObj.name} benefits from feature scaling. Consider enabling Standard or Min-Max scaling.`
              : `${selectedModelObj.name} does not require feature scaling. Scaling is optional for tree-based models.`
            }
          </div>
        )}
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Missing Values Strategy</label>
          <select 
            className={`w-full bg-slate-900 border border-slate-600 rounded-lg p-2 text-white ${aiPopulated ? 'ai-populated-field' : ''}`}
            value={preprocessing.missing_values}
            onChange={(e) => setPreprocessing({...preprocessing, missing_values: e.target.value})}
          >
            <option value="mean">Mean (Numerical) / Mode (Categorical)</option>
            <option value="median">Median (Numerical) / Mode (Categorical)</option>
            <option value="mode">Mode (All)</option>
            <option value="drop">Drop Rows with Missing Values</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Scaling</label>
          <select 
            className={`w-full bg-slate-900 border border-slate-600 rounded-lg p-2 text-white ${aiPopulated ? 'ai-populated-field' : ''}`}
            value={preprocessing.scaling}
            onChange={(e) => setPreprocessing({...preprocessing, scaling: e.target.value})}
          >
            <option value="none">None</option>
            <option value="standard">Standard Scaler</option>
            <option value="minmax">Min-Max Scaler</option>
          </select>
        </div>

        <div>
          <label className="flex items-center gap-2 cursor-pointer text-sm font-medium text-slate-300">
            <input 
              type="checkbox" 
              checked={preprocessing.outlier_handling}
              onChange={(e) => setPreprocessing({...preprocessing, outlier_handling: e.target.checked})}
              className="w-4 h-4 rounded text-blue-600" 
            />
            <span>Handle Outliers (Clip to 1st-99th percentile)</span>
          </label>
        </div>

        {problemType === 'classification' && (
          <div>
            <label className="flex items-center gap-2 cursor-pointer text-sm font-medium text-slate-300">
              <input 
                type="checkbox" 
                checked={preprocessing.smote || false}
                onChange={(e) => setPreprocessing({...preprocessing, smote: e.target.checked})}
                className="w-4 h-4 rounded text-blue-600" 
              />
              <span>Apply SMOTE (Synthetic Minority Oversampling)</span>
            </label>
            <p className="text-xs text-slate-500 mt-1 ml-6">Only applies to training data. Helps with imbalanced class distributions.</p>
          </div>
        )}

        <div>
          <label className="flex items-center gap-2 cursor-pointer text-sm font-medium text-slate-300">
            <input 
              type="checkbox" 
              checked={preprocessing.pca || false}
              onChange={(e) => setPreprocessing({...preprocessing, pca: e.target.checked})}
              className="w-4 h-4 rounded text-blue-600" 
            />
            <span>Apply PCA (Principal Component Analysis)</span>
          </label>
          <p className="text-xs text-slate-500 mt-1 ml-6">Reduces dimensionality and addresses high multicollinearity without data leakage.</p>
          {preprocessing.pca && (
            <div className="mt-2 ml-6 flex items-center gap-3">
              <span className="text-xs text-slate-400">PCA Components (leave blank for auto):</span>
              <input 
                type="number" 
                min="1" 
                max="50"
                value={preprocessing.pca_components || ''} 
                onChange={(e) => setPreprocessing({...preprocessing, pca_components: e.target.value ? parseInt(e.target.value) : null})}
                className="w-20 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-white text-sm"
                placeholder="Auto"
              />
            </div>
          )}
        </div>

        <div className="flex justify-end pt-4">
          <button onClick={goNext} className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-medium">Continue</button>
        </div>
      </div>
    );
  };

  const renderStep6 = () => {
    return (
      <div className="space-y-6">
        <h3 className="text-xl font-bold">Select Model ({problemType})</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {availableModels.map(model => (
            <div 
              key={model.model_id}
              onClick={() => handleSelectModel(model)}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${modelId === model.model_id ? 'bg-blue-900/40 border-blue-500' : 'bg-slate-800 border-slate-700 hover:border-slate-500'} ${aiPopulated && modelId === model.model_id ? 'ai-populated-field' : ''}`}
            >
              <h4 className="font-bold text-lg mb-2">{model.name}</h4>
              <p className="text-sm text-slate-400">{model.description}</p>
              {!model.needs_scaling && preprocessing.scaling !== 'none' && (
                 <div className="mt-3 text-xs text-yellow-500">Note: This model doesn't require scaling.</div>
              )}
            </div>
          ))}
        </div>
        <div className="flex justify-end pt-4">
          <button 
            disabled={!modelId} 
            onClick={goNext} 
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-6 py-2 rounded-lg font-medium"
          >
            Configure Parameters
          </button>
        </div>
      </div>
    );
  };

  const renderStep7 = () => {
    if (!selectedModelObj) return <div>No model selected.</div>;
    return (
      <div className="space-y-6 max-w-2xl mx-auto bg-slate-800 p-6 rounded-lg border border-slate-700">
        <h3 className="text-xl font-bold">Parameters for {selectedModelObj.name}</h3>
        
        {selectedModelObj.parameters?.length === 0 ? (
          <p className="text-slate-400">No configurable parameters for this model.</p>
        ) : (
          <div className="space-y-4">
            {selectedModelObj.parameters?.map(p => {
              const val = modelParams[p.name];
              const isNull = p.nullable && val === null;
              
              return (
                <div key={p.name} className="border-b border-slate-700 pb-4">
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-sm font-medium text-slate-200">{p.name}</label>
                    {p.nullable && (
                      <label className="text-xs text-slate-400 flex items-center gap-1 cursor-pointer">
                        <input 
                          type="checkbox" 
                          checked={isNull} 
                          onChange={(e) => {
                            setModelParams({...modelParams, [p.name]: e.target.checked ? null : p.default});
                          }} 
                        />
                        Use Default (None)
                      </label>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mb-2">{p.description}</p>
                  
                  {!isNull && (
                    <>
                      {p.type === 'bool' && (
                        <input type="checkbox" checked={val === true} onChange={(e)=>setModelParams({...modelParams, [p.name]: e.target.checked})} />
                      )}
                      {p.type === 'string' && p.options && (
                        <select className="bg-slate-900 border border-slate-600 rounded p-1 text-sm w-full" value={val || ''} onChange={(e)=>setModelParams({...modelParams, [p.name]: e.target.value})}>
                          {p.options.map(o => <option key={o} value={o}>{o}</option>)}
                        </select>
                      )}
                      {p.type === 'int' && (
                        <input type="number" step="1" min={p.min} max={p.max} className="bg-slate-900 border border-slate-600 rounded p-1 text-sm w-full" value={val !== null ? val : ''} onChange={(e)=>setModelParams({...modelParams, [p.name]: parseInt(e.target.value, 10)})} />
                      )}
                      {p.type === 'float' && (
                        <input type="number" step="0.01" min={p.min} max={p.max} className="bg-slate-900 border border-slate-600 rounded p-1 text-sm w-full" value={val !== null ? val : ''} onChange={(e)=>setModelParams({...modelParams, [p.name]: parseFloat(e.target.value)})} />
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <div className="flex justify-end pt-4">
          <button onClick={goNext} className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-medium">Review & Train</button>
        </div>
      </div>
    );
  };

  const renderStep8 = () => {
    const isClf = problemType === 'classification';
    const optunaMetrics = isClf 
      ? ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'] 
      : ['rmse', 'mae', 'mse', 'r2'];

    return (
      <div className="space-y-6 max-w-2xl mx-auto bg-slate-800 p-6 rounded-lg border border-slate-700">
        {/* Mode Switcher */}
        <div className="flex border-b border-slate-700 pb-4 justify-center gap-4">
          <button
            onClick={() => setTrainMode('manual')}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-colors ${trainMode === 'manual' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
          >
            Manual Model Training
          </button>
          <button
            onClick={() => {
              setTrainMode('optuna');
              if (!optunaConfig.metric) {
                setOptunaConfig(c => ({...c, metric: isClf ? 'accuracy' : 'rmse'}));
              }
            }}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-colors flex items-center gap-2 ${trainMode === 'optuna' ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
          >
            <span>⚡ Optuna AutoML</span>
          </button>
        </div>

        {trainMode === 'manual' ? (
          <>
            <h3 className="text-xl font-bold">Review Configuration</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Target</span><span className="font-bold">{target}</span>
              </div>
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Problem Type</span><span className="font-bold capitalize">{problemType}</span>
              </div>
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Model</span><span className="font-bold">{selectedModelObj?.name}</span>
              </div>
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Missing Values</span><span className="font-bold capitalize">{preprocessing.missing_values}</span>
              </div>
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Scaling</span><span className="font-bold capitalize">{preprocessing.scaling}</span>
              </div>
              {preprocessing.pca && (
                <div className="flex justify-between border-b border-slate-700 pb-2">
                  <span className="text-slate-400">PCA</span><span className="font-bold">Enabled ({preprocessing.pca_components || 'Auto'} comp)</span>
                </div>
              )}
            </div>

            <div className="flex justify-center pt-6">
              <button 
                onClick={handleTrain} 
                disabled={loading || trainingActive}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 py-3 rounded-lg font-bold text-lg flex items-center justify-center gap-2"
              >
                {loading ? <span className="animate-pulse">Training Model...</span> : 'Train Model'}
              </button>
            </div>
          </>
        ) : (
          <div className="space-y-5">
            <div>
              <h3 className="text-xl font-bold flex items-center gap-2">
                <span className="text-purple-400">⚡</span> Hyperparameter Optimization
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Search the selected model with cross-validation strictly on the training fold. The held-out test set is evaluated only after selection.
              </p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-slate-300 mb-1">Search strategy</label>
              <select className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-white" value={optunaConfig.strategy} onChange={(e) => setOptunaConfig(c => ({...c, strategy: e.target.value}))}>
                <option value="bayesian">Bayesian optimization (Optuna TPE)</option>
                <option value="optuna">Optuna TPE search</option>
                <option value="grid">GridSearchCV (exhaustive)</option>
                <option value="random">RandomizedSearchCV</option>
              </select>
            </div>

            {/* Candidate Models */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-semibold text-slate-300">Candidate Models</label>
                <button
                  type="button"
                  onClick={() => {
                    if (optunaConfig.candidate_models.length === availableModels.length) {
                      setOptunaConfig(c => ({...c, candidate_models: []}));
                    } else {
                      setOptunaConfig(c => ({...c, candidate_models: availableModels.map(m => m.model_id)}));
                    }
                  }}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  {optunaConfig.candidate_models.length === availableModels.length ? 'Clear All' : 'Select All'}
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {availableModels.map(m => {
                  const isChecked = optunaConfig.candidate_models.length === 0 || optunaConfig.candidate_models.includes(m.model_id);
                  return (
                    <label key={m.model_id} className="flex items-center gap-2 text-xs p-2 rounded bg-slate-900 border border-slate-700 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={(e) => {
                          let curr = optunaConfig.candidate_models.length === 0 ? availableModels.map(x=>x.model_id) : [...optunaConfig.candidate_models];
                          if (e.target.checked) {
                            if (!curr.includes(m.model_id)) curr.push(m.model_id);
                          } else {
                            curr = curr.filter(id => id !== m.model_id);
                          }
                          setOptunaConfig(c => ({...c, candidate_models: curr}));
                        }}
                        className="rounded text-purple-600"
                      />
                      <span className="truncate">{m.name}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Metric & Budget */}
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">Target Metric</label>
                <select
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-white uppercase"
                  value={optunaConfig.metric}
                  onChange={(e) => setOptunaConfig(c => ({...c, metric: e.target.value}))}
                >
                  {optunaMetrics.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">Trial Budget</label>
                <select
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-white"
                  value={optunaConfig.strategy === 'random' ? (optunaConfig.n_iter || 10) : optunaConfig.n_trials}
                  onChange={(e) => setOptunaConfig(c => c.strategy === 'random' ? ({...c, n_iter: parseInt(e.target.value, 10)}) : ({...c, n_trials: parseInt(e.target.value, 10)}))}
                >
                  <option value={5}>5 trials (Fast)</option>
                  <option value={10}>10 trials</option>
                  <option value={15}>15 trials (Recommended)</option>
                  <option value={20}>20 trials</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">CV Folds</label>
                <select
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-white"
                  value={optunaConfig.cv_folds}
                  onChange={(e) => setOptunaConfig(c => ({...c, cv_folds: parseInt(e.target.value, 10)}))}
                >
                  <option value={2}>2 Folds</option>
                  <option value={3}>3 Folds (Standard)</option>
                  <option value={5}>5 Folds (Rigorous)</option>
                </select>
              </div>
            </div>

            <div className="flex justify-center pt-4">
              <button 
                onClick={handleOptimize} 
                disabled={loading}
                className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 py-3 rounded-lg font-bold text-lg flex items-center justify-center gap-2 shadow-lg shadow-purple-600/30"
              >
                {loading ? <span className="animate-pulse">Running search…</span> : `Run ${optunaConfig.strategy === 'grid' ? 'GridSearchCV' : optunaConfig.strategy === 'random' ? 'RandomizedSearchCV' : optunaConfig.strategy === 'bayesian' ? 'Bayesian' : 'Optuna'} search`}
              </button>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderStep9 = () => {
    if (!evaluation) return <div>No evaluation data. Train a model first.</div>;
    const { metrics } = evaluation;

    return (
      <div className="space-y-6">
        <h3 className="text-2xl font-bold">Model Evaluation</h3>

        {/* Optuna Summary Card if applicable */}
        {optimizationResult && (
          <div className="bg-gradient-to-r from-purple-950/60 to-indigo-950/60 p-6 rounded-lg border border-purple-500/40 space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">Optuna AutoML Winner</span>
                <h4 className="text-xl font-black text-white capitalize">{optimizationResult.best_model_id?.replace(/_/g, ' ')}</h4>
              </div>
              <div className="text-right">
                <div className="text-xs text-slate-400">Best CV {optimizationResult.metric}</div>
                <div className="text-2xl font-bold text-emerald-400">{optimizationResult.best_cv_score}</div>
              </div>
            </div>

            <div className="text-xs bg-slate-900/80 p-3 rounded border border-slate-700">
              <span className="text-slate-400 font-semibold">Tuned Hyperparameters: </span>
              <span className="font-mono text-purple-300">
                {JSON.stringify(optimizationResult.best_params)}
              </span>
            </div>

            {optimizationResult.trials?.length > 0 && (
              <details className="text-xs text-slate-400 cursor-pointer">
                <summary className="font-semibold text-purple-300 hover:text-purple-200">
                  View Optuna Leaderboard ({optimizationResult.trials.length} trials)
                </summary>
                <div className="mt-3 overflow-x-auto max-h-48 overflow-y-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-700 text-slate-400">
                        <th className="p-1">#</th>
                        <th className="p-1">Model</th>
                        <th className="p-1">Score</th>
                        <th className="p-1">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {optimizationResult.trials.map(t => (
                        <tr key={t.trial_number} className="hover:bg-slate-800">
                          <td className="p-1">{t.trial_number}</td>
                          <td className="p-1 capitalize">{t.model_id?.replace(/_/g, ' ')}</td>
                          <td className="p-1 font-mono text-emerald-400">{t.score ?? 'Failed'}</td>
                          <td className="p-1 text-slate-500">{t.state}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </div>
        )}
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {problemType === 'classification' ? (
            <>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">Accuracy</div>
                <div className="text-2xl font-bold text-blue-400">{metrics.accuracy?.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">Precision</div>
                <div className="text-2xl font-bold text-blue-400">{metrics.precision?.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">Recall</div>
                <div className="text-2xl font-bold text-blue-400">{metrics.recall?.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">F1 Score</div>
                <div className="text-2xl font-bold text-blue-400">{metrics.f1?.toFixed(4)}</div>
              </div>
              {metrics.roc_auc !== undefined && (
                <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center col-span-2 md:col-span-4">
                  <div className="text-sm text-slate-400">ROC-AUC Score</div>
                  <div className="text-2xl font-bold text-purple-400">{metrics.roc_auc?.toFixed(4)}</div>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">MAE</div>
                <div className="text-2xl font-bold text-indigo-400">{metrics.mae?.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">MSE</div>
                <div className="text-2xl font-bold text-indigo-400">{metrics.mse?.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">RMSE</div>
                <div className="text-2xl font-bold text-indigo-400">{metrics.rmse?.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 text-center">
                <div className="text-sm text-slate-400">R² Score</div>
                <div className="text-2xl font-bold text-indigo-400">{metrics.r2?.toFixed(4)}</div>
              </div>
            </>
          )}
        </div>

        {problemType === 'classification' && metrics.confusion_matrix && (
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h4 className="font-bold mb-4">Confusion Matrix</h4>
            <div className="overflow-x-auto">
              <table className="text-center text-sm border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 border border-slate-600 bg-slate-900">True \ Pred</th>
                    {metrics.class_labels?.map(l => <th key={l} className="p-2 border border-slate-600 bg-slate-700">{l}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {metrics.confusion_matrix.map((row, i) => (
                    <tr key={i}>
                      <th className="p-2 border border-slate-600 bg-slate-700">{metrics.class_labels?.[i]}</th>
                      {row.map((val, j) => (
                        <td key={j} className="p-2 border border-slate-600" style={{backgroundColor: `rgba(59, 130, 246, ${Math.max(0.1, val/Math.max(...row.map(r=>Math.max(...r, 1))))})`}}>
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {runs.length > 0 && (
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h4 className="font-bold mb-4">Training Run History ({runs.length})</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-700 text-slate-400">
                    <th className="py-2 px-3">Run ID</th>
                    <th className="py-2 px-3">Model</th>
                    <th className="py-2 px-3">Timestamp</th>
                    <th className="py-2 px-3">Primary Metric</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                  {runs.map((r, idx) => (
                    <tr key={r.run_id || idx} className="hover:bg-slate-700/50">
                      <td className="py-2 px-3 font-mono text-xs text-blue-400">{r.run_id}</td>
                      <td className="py-2 px-3 capitalize">{r.model_id?.replace(/_/g, ' ')}</td>
                      <td className="py-2 px-3 text-slate-400 text-xs">{r.timestamp?.slice(0, 19).replace('T', ' ')}</td>
                      <td className="py-2 px-3 font-bold text-green-400">
                        {r.metrics?.accuracy !== undefined ? `Acc: ${(r.metrics.accuracy * 100).toFixed(1)}%` : (r.metrics?.r2 !== undefined ? `R²: ${r.metrics.r2.toFixed(4)}` : 'Done')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div className="flex justify-end pt-4">
          <button onClick={() => { goNext(); api.getArtifacts(workspaceId).then(r=>setArtifacts(r.artifacts||[])); }} className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-medium">Continue to Predictions</button>
        </div>
      </div>
    );
  };

  const renderStep10 = () => {
    return (
      <div className="space-y-8">
        <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
          <h3 className="text-xl font-bold mb-4">Make Predictions</h3>
          <p className="text-slate-400 mb-4 text-sm">Upload a CSV file with the same features as your training data to get predictions.</p>
          <div className="flex items-center gap-4">
            <input type="file" accept=".csv" onChange={handlePredictUpload} className="hidden" id="upload-predict" />
            <label htmlFor="upload-predict" className="cursor-pointer bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg text-sm font-medium transition-colors">
              Upload Data for Prediction
            </label>
            {loading && <span className="text-slate-400 animate-pulse text-sm">Processing...</span>}
          </div>
        </div>

        {predictionWarnings && predictionWarnings.length > 0 && (
          <div className="bg-yellow-900/40 border border-yellow-600/60 p-4 rounded-lg text-yellow-200 text-sm">
            <div className="font-bold flex items-center gap-2 mb-1">
              <span>⚠️ Notice</span>
            </div>
            <ul className="list-disc list-inside space-y-1">
              {predictionWarnings.map((w, idx) => (
                <li key={idx}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {predictions && (
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 space-y-4">
            <div className="flex justify-between items-center">
              <h4 className="font-bold text-lg text-green-400">Prediction Success!</h4>
              {predictionUrl && (
                <a href={api.getArtifactUrl(workspaceId, predictionUrl.split('/').pop())} download className="bg-green-600 hover:bg-green-500 px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-2">
                  Download Full Predictions
                </a>
              )}
            </div>
            
            <p className="text-sm text-slate-400">Showing first 10 of {predictions.length} predictions:</p>
            <div className="flex flex-wrap gap-2">
              {predictions.slice(0,10).map((p,i) => (
                <span key={i} className="bg-slate-900 border border-slate-600 px-3 py-1 rounded text-sm font-mono">
                  {typeof p === 'number' && !Number.isInteger(p) ? p.toFixed(4) : String(p)}
                </span>
              ))}
              {predictions.length > 10 && <span className="px-3 py-1 text-slate-500">...</span>}
            </div>
          </div>
        )}

        {artifacts.length > 0 && (
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h4 className="font-bold mb-4">Workspace Artifacts</h4>
            <div className="space-y-2">
              {artifacts.map(art => (
                <div key={art.filename} className="flex justify-between items-center p-3 bg-slate-900 border border-slate-700 rounded">
                  <div>
                    <div className="font-medium">{art.filename}</div>
                    <div className="text-xs text-slate-500">{art.type}</div>
                  </div>
                  <a href={api.getArtifactUrl(workspaceId, art.filename)} download className="text-blue-400 hover:text-blue-300 text-sm">Download</a>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Toast notification for external dataset handoff */}
      {externalDatasetToast && <div className="ml-dp-handoff-toast" role="status">{externalDatasetToast}</div>}
      {initialGoal && (
        <section className="workspace-objective" aria-label="Project objective">
          <span>PROJECT OBJECTIVE</span>
          <p>{initialGoal}</p>
        </section>
      )}
      {aiPopulated && <div className="workspace-objective border-blue-500" role="status"><span>AI-POPULATED CONFIGURATION</span><p>Review and edit the highlighted workflow values before running a stage.</p></div>}
      {/* Navigation */}
      <nav aria-label="Machine learning workflow" className="bg-slate-800 p-4 rounded-lg border border-slate-700 overflow-x-auto">
        <div className="flex items-center gap-2">
          <ol className="flex gap-2 min-w-max flex-1">
            {STEPS.map((s, i) => (
              <li
                key={i} 
              >
              <button
                type="button"
                aria-current={i === step ? 'step' : undefined}
                aria-label={`Step ${i + 1}: ${s}${i < step ? ', completed' : ''}`}
                disabled={i > step}
                onClick={() => i <= step && gotoStep(i)}
                className={`px-3 py-2 text-sm rounded-lg whitespace-nowrap transition-colors flex items-center gap-2 disabled:cursor-not-allowed ${
                  i === step ? 'bg-blue-600 text-white font-bold' : 
                  i < step ? 'bg-slate-700 text-slate-300 cursor-pointer hover:bg-slate-600' : 'bg-slate-900/50 text-slate-500 cursor-not-allowed'
                }`}
              >
                <span className="w-5 h-5 rounded-full bg-black/10 flex items-center justify-center text-xs">{i + 1}</span>
                {s}
              </button>
              </li>
            ))}
          </ol>
          {onOpenPlayground && (
            <button
              type="button"
              className="ws-playground-btn"
              onClick={() => {
                if (loading) {
                  if (!window.confirm('Analysis or configuration in progress. Switch to Data Playground anyway?')) return;
                }
                onOpenPlayground();
              }}
            >
              ▤ Playground
            </button>
          )}
        </div>
      </nav>

      {error && (
        <div role="alert" className="bg-red-900/50 border border-red-500 text-red-200 p-4 rounded-lg">
          <strong className="block mb-1">Error</strong>
          {error}
        </div>
      )}

      {trainStatus && (
        <section className={`rounded-lg border p-5 ${trainStatus.status === 'FAILED' ? 'border-red-500 bg-red-950/40' : trainStatus.status === 'COMPLETED' ? 'border-emerald-500 bg-emerald-950/30' : 'border-blue-500 bg-slate-800'}`} aria-live="polite" aria-label="Training progress">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-slate-400">Training job · {trainStatus.status}</p>
              <h2 className="mt-1 text-lg font-bold">{selectedModelObj?.name || trainStatus.model_id || modelId || 'Selected model'}</h2>
              <p className="text-sm text-slate-300">{datasetInfo?.filename || trainStatus.dataset_name || 'Selected dataset'}</p>
            </div>
            {!['COMPLETED', 'FAILED', 'CANCELLED'].includes(trainStatus.status) && (
              <button type="button" onClick={handleCancelTraining} disabled={trainStatus.status === 'CANCELLING'} className="rounded-md border border-red-400 px-4 py-2 font-semibold text-red-200 hover:bg-red-950 disabled:cursor-not-allowed disabled:opacity-60">
                {trainStatus.status === 'CANCELLING' ? 'Cancelling…' : 'Cancel Training'}
              </button>
            )}
          </div>
          <p className="mt-3 text-sm text-slate-200">{trainStatus.message || 'Preparing training job.'}</p>
          <p className="mt-1 text-xs text-slate-400">Stage: {trainStatus.stage || trainStatus.status} · Elapsed: {Math.floor(trainingElapsed / 60)}m {trainingElapsed % 60}s</p>
          {['FAILED', 'CANCELLED'].includes(trainStatus.status) && (
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <p className="text-sm text-red-200">{trainStatus.message}</p>
              <button type="button" onClick={() => { setTrainStatus(null); setError(null); setStep(8); }} className="rounded-md bg-blue-600 px-4 py-2 font-semibold hover:bg-blue-500">Retry with this configuration</button>
            </div>
          )}
          {trainStatus.status === 'CANCELLED' && <p className="mt-3 text-sm text-slate-300">The run was cancelled. Your selected configuration and previously completed models are preserved.</p>}
        </section>
      )}

      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {loading ? 'Working. Please wait.' : error ? `Error: ${error}` : ''}
      </div>

      {/* Content */}
      <div className="min-h-[400px]">
        {step === 0 && renderStep0()}
        {step === 1 && renderStep1()}
        {step === 2 && renderStep2()}
        {step === 3 && renderStep3()}
        {step === 4 && renderStep4()}
        {step === 5 && renderStep5()}
        {step === 6 && renderStep6()}
        {step === 7 && renderStep7()}
        {step === 8 && renderStep8()}
        {step === 9 && renderStep9()}
        {step === 10 && renderStep10()}
      </div>
      <AICompanion workspaceId={workspaceId} onFillConfiguration={fillAIConfiguration} />
    </div>
  );
}
