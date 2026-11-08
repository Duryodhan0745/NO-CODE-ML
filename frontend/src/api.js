const handleResponse = async (res) => {
  if (!res.ok) {
    let msg = 'API Request Failed';
    try {
      const err = await res.json();
      msg = err.detail?.message || err.detail || msg;
    } catch (e) {
      msg = `HTTP Error ${res.status}`;
    }
    throw new Error(msg);
  }
  return res.json();
};

const uploadError = (error) => {
  if (error instanceof TypeError) {
    return new Error('Could not reach the ML backend. Start the backend on port 8000, then retry the CSV upload.');
  }
  return error;
};

const aiHeaders = () => {
  const token = window.sessionStorage.getItem('ai_auth_token') || window.localStorage.getItem('ai_auth_token');
  return token ? { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` } : { 'Content-Type': 'application/json' };
};

export const getAIAuthConfig = async () => {
  const res = await fetch('/api/auth/config');
  return handleResponse(res);
};

export const loginTestAdmin = async (email, password) => {
  const res = await fetch('/api/auth/test-admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) });
  return handleResponse(res);
};

export const createAIConversation = async (workspaceId = null) => {
  const res = await fetch('/api/ai/conversations', { method: 'POST', headers: aiHeaders(), body: JSON.stringify({ workspace_id: workspaceId }) });
  return handleResponse(res);
};

export const sendAIMessage = async (conversationId, content, mode = 'manual') => {
  const res = await fetch(`/api/ai/conversations/${conversationId}/messages`, { method: 'POST', headers: aiHeaders(), body: JSON.stringify({ content, mode }) });
  return handleResponse(res);
};

export const streamAIMessage = async (conversationId, content, mode, onEvent) => {
  const response = await fetch(`/api/ai/conversations/${conversationId}/messages/stream`, {
    method: 'POST',
    headers: aiHeaders(),
    body: JSON.stringify({ content, mode }),
  });
  if (!response.ok) return handleResponse(response);
  if (!response.body) throw new Error('The browser does not support streamed assistant responses.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const dispatch = frame => {
    const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
    if (!data) return;
    const event = JSON.parse(data);
    onEvent(event);
    if (event.type === 'error') throw new Error(event.message || 'The assistant could not complete this response.');
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() || '';
      frames.forEach(dispatch);
      if (done) break;
    }
    if (buffer.trim()) dispatch(buffer);
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
};

export const decideWorkflowProposal = async (proposalId, decision, configurationFingerprint) => handleResponse(await fetch(`/api/ai/workflow-proposals/${proposalId}/decision`, {
  method: 'POST', headers: aiHeaders(), body: JSON.stringify({ decision, configuration_fingerprint: configurationFingerprint }),
}));

export const listWorkflowProposals = async (workspaceId) => handleResponse(await fetch(`/api/ai/workflow-proposals?workspace_id=${encodeURIComponent(workspaceId)}&actionable_only=true`, { headers: aiHeaders() }));

export const createWorkflowProposal = async (proposal) => handleResponse(await fetch('/api/ai/workflow-proposals', {
  method: 'POST', headers: aiHeaders(), body: JSON.stringify(proposal),
}));

export const getAIContext = async (conversationId) => {
  const res = await fetch(`/api/ai/conversations/${conversationId}/context`, { headers: aiHeaders() });
  return handleResponse(res);
};

export const getWorkflowContext = async (workspaceId) => {
  const res = await fetch(`/api/workspaces/${workspaceId}/ai-context`, { headers: aiHeaders() });
  return handleResponse(res);
};

export const getWorkflowEvents = async (workspaceId) => {
  const res = await fetch(`/api/workspaces/${workspaceId}/events`, { headers: aiHeaders() });
  return handleResponse(res);
};

export const getAIMemorySettings = async () => handleResponse(await fetch('/api/ai/memory-settings', { headers: aiHeaders() }));
export const setAIMemorySettings = async (enabled) => handleResponse(await fetch('/api/ai/memory-settings', { method: 'PUT', headers: aiHeaders(), body: JSON.stringify({ enabled }) }));
export const listAIMemories = async ({ workspaceId, scope } = {}) => {
  const params = new URLSearchParams();
  if (workspaceId) params.set('workspace_id', workspaceId);
  if (scope) params.set('scope', scope);
  return handleResponse(await fetch(`/api/ai/memories${params.size ? `?${params}` : ''}`, { headers: aiHeaders() }));
};
export const updateAIMemory = async (memoryId, changes) => handleResponse(await fetch(`/api/ai/memories/${memoryId}`, { method: 'PATCH', headers: aiHeaders(), body: JSON.stringify(changes) }));
export const deleteAIMemory = async (memoryId) => handleResponse(await fetch(`/api/ai/memories/${memoryId}`, { method: 'DELETE', headers: aiHeaders() }));
export const clearAIMemories = async (workspaceId) => {
  const params = new URLSearchParams();
  if (workspaceId) params.set('workspace_id', workspaceId);
  return handleResponse(await fetch(`/api/ai/memories${params.size ? `?${params}` : ''}`, { method: 'DELETE', headers: aiHeaders() }));
};
export const clearAIConversationMemory = async (conversationId) => handleResponse(await fetch(`/api/ai/conversations/${conversationId}/memory`, { method: 'DELETE', headers: aiHeaders() }));

export const claimAIWorkspace = async (workspaceId) => {
  const res = await fetch(`/api/ai/workspaces/${workspaceId}/claim`, { method: 'POST', headers: aiHeaders() });
  return handleResponse(res);
};

export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch('/api/datasets/upload', {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  } catch (error) {
    throw uploadError(error);
  }
};

export const getPreview = async (datasetId, rows = 20) => {
  const res = await fetch(`/api/datasets/${datasetId}/preview?rows=${rows}`);
  return handleResponse(res);
};

export const getDataset = async (datasetId) => handleResponse(await fetch(`/api/datasets/${datasetId}`));

export const getAnalysis = async (datasetId) => {
  const res = await fetch(`/api/datasets/${datasetId}/analysis`);
  return handleResponse(res);
};

export const getColumnStatistics = async (datasetId, target) => {
  const query = target ? `?target=${encodeURIComponent(target)}` : '';
  return handleResponse(await fetch(`/api/datasets/${datasetId}/columns${query}`));
};

export const applyColumnSelection = async (workspaceId, retainedColumns, target) => handleResponse(await fetch(`/api/workspaces/${workspaceId}/column-selection`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ retained_columns: retainedColumns, target: target || null }),
}));

export const restoreColumnSelection = async (workspaceId) => handleResponse(await fetch(`/api/workspaces/${workspaceId}/column-selection/restore`, { method: 'POST' }));

export const getCorrelation = async (datasetId) => {
  const res = await fetch(`/api/datasets/${datasetId}/correlation`);
  return handleResponse(res);
};

export const getVif = async (datasetId) => {
  const res = await fetch(`/api/datasets/${datasetId}/vif`);
  return handleResponse(res);
};

export const generateProfiling = async (datasetId) => {
  const res = await fetch(`/api/datasets/${datasetId}/profiling`, { method: 'POST' });
  return handleResponse(res);
};

export const getProfilingReportUrl = (datasetId) => {
  return `/api/datasets/${datasetId}/profiling/report`;
};

export const getModels = async (problemType) => {
  const res = await fetch(`/api/models?problem_type=${problemType}`);
  return handleResponse(res);
};

export const createWorkspace = async (datasetId) => {
  const res = await fetch('/api/workspaces', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId }),
  });
  const workspace = await handleResponse(res);
  if (workspace?.id) window.localStorage.setItem('active_workspace_id', workspace.id);
  return workspace;
};

export const getWorkspace = async (workspaceId) => handleResponse(await fetch(`/api/workspaces/${workspaceId}`));

export const configureWorkspace = async (wsId, config) => {
  const res = await fetch(`/api/workspaces/${wsId}/configure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return handleResponse(res);
};

export const trainWorkspace = async (wsId) => {
  const res = await fetch(`/api/workspaces/${wsId}/train`, {
    method: 'POST',
  });
  return handleResponse(res);
};

export const startTrainingJob = async (workspaceId) => handleResponse(await fetch(
  `/api/workspaces/${workspaceId}/training-jobs`,
  { method: 'POST', headers: aiHeaders() },
));

export const getTrainingJob = async (jobId) => handleResponse(await fetch(
  `/api/training-jobs/${jobId}`,
  { headers: aiHeaders() },
));

export const cancelTrainingJob = async (jobId) => handleResponse(await fetch(
  `/api/training-jobs/${jobId}/cancel`,
  { method: 'POST', headers: aiHeaders() },
));

export const optimizeWorkspace = async (wsId, config = {}) => {
  const res = await fetch(`/api/workspaces/${wsId}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return handleResponse(res);
};

export const getOptimization = async (wsId) => {
  const res = await fetch(`/api/workspaces/${wsId}/optimization`);
  return handleResponse(res);
};

export const getEvaluation = async (wsId) => {
  const res = await fetch(`/api/workspaces/${wsId}/evaluation`);
  return handleResponse(res);
};

export const getRuns = async (wsId) => {
  const res = await fetch(`/api/workspaces/${wsId}/runs`);
  return handleResponse(res);
};

export const predictWorkspace = async (wsId, file) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`/api/workspaces/${wsId}/predict`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse(res);
};

export const getArtifacts = async (wsId) => {
  const res = await fetch(`/api/workspaces/${wsId}/artifacts`);
  return handleResponse(res);
};

export const getArtifactUrl = (wsId, filename) => {
  return `/api/workspaces/${wsId}/artifacts/${filename}`;
};
