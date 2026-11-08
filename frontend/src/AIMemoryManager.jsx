import React, { useEffect, useState } from 'react';
import { clearAIConversationMemory, clearAIMemories, deleteAIMemory, getAIMemorySettings, listAIMemories, setAIMemorySettings, updateAIMemory } from './api';

export default function AIMemoryManager({ workspaceId, conversationId, onConversationCleared, onClose }) {
  const [scope, setScope] = useState('user');
  const [settings, setSettings] = useState({ enabled: true });
  const [memories, setMemories] = useState([]);
  const [drafts, setDrafts] = useState({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const refresh = async (nextScope = scope) => {
    setBusy(true); setError('');
    try {
      const [config, data] = await Promise.all([
        getAIMemorySettings(),
        listAIMemories({ workspaceId: nextScope === 'project' ? workspaceId : undefined, scope: nextScope }),
      ]);
      setSettings(config);
      setMemories(data.memories || []);
      setDrafts(Object.fromEntries((data.memories || []).map(memory => [memory.id, memory.content])));
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  useEffect(() => { refresh(scope); }, [scope, workspaceId]);

  const toggleMemory = async () => {
    try { const next = await setAIMemorySettings(!settings.enabled); setSettings(next); }
    catch (e) { setError(e.message); }
  };
  const save = async id => {
    try { await updateAIMemory(id, { content: drafts[id] }); await refresh(); }
    catch (e) { setError(e.message); }
  };
  const remove = async id => {
    try { await deleteAIMemory(id); await refresh(); }
    catch (e) { setError(e.message); }
  };
  const clear = async () => {
    if (!window.confirm(`Clear all ${scope} memories?`)) return;
    try { await clearAIMemories(scope === 'project' ? workspaceId : undefined); await refresh(); }
    catch (e) { setError(e.message); }
  };
  const clearConversation = async () => {
    if (!window.confirm('Clear this conversation’s messages, summary, and conversation-scoped memories?')) return;
    try {
      await clearAIConversationMemory(conversationId);
      onConversationCleared?.();
    } catch (e) { setError(e.message); }
  };

  return <section className="ai-memory-manager" aria-label="AI memory management">
    <header className="ai-memory-manager-head">
      <div><span className="ai-eyebrow">PERSONALIZATION</span><h2>Assistant memory</h2></div>
      {onClose && <button className="ai-auth-signout" onClick={onClose}>Back to chat</button>}
    </header>
    <p className="ai-memory-intro">Explicit preferences and project decisions help the assistant maintain continuity. A memory is used only when it matches your current question; workflow results always come from current backend records.</p>
    <label className="ai-memory-toggle"><input type="checkbox" checked={settings.enabled} onChange={toggleMemory}/> Enable long-term memory</label>
    <div className="ai-memory-tabs" role="tablist" aria-label="Memory scope">
      <button role="tab" aria-selected={scope === 'user'} onClick={() => setScope('user')}>Your preferences</button>
      {workspaceId && <button role="tab" aria-selected={scope === 'project'} onClick={() => setScope('project')}>This project</button>}
    </div>
    {busy && <p className="ai-memory-note">Loading memories…</p>}
    {!busy && memories.length === 0 && <p className="ai-memory-note">No saved {scope} memories yet. Tell the assistant a lasting preference, such as “I prefer Random Forest for classification.”</p>}
    <div className="ai-memory-list">{memories.map(memory => <article className="ai-memory-card" key={memory.id}>
      <div className="ai-memory-meta"><strong>{memory.category.replaceAll('_', ' ')}</strong><span>Confidence {Math.round(memory.confidence * 100)}% · importance {memory.importance}/5</span></div>
      <textarea aria-label={`Edit ${memory.category} memory`} value={drafts[memory.id] ?? memory.content} onChange={e => setDrafts(values => ({ ...values, [memory.id]: e.target.value }))} rows="2"/>
      <p className="ai-memory-source">Saved {new Date(memory.created_at).toLocaleDateString()} · updated {new Date(memory.updated_at).toLocaleDateString()}</p>
      <div className="ai-memory-actions"><button onClick={() => save(memory.id)} disabled={drafts[memory.id] === memory.content}>Save correction</button><button onClick={() => remove(memory.id)}>Delete</button></div>
    </article>)}</div>
    {!!memories.length && <button className="ai-memory-clear" onClick={clear}>Clear {scope} memories</button>}
    {conversationId && <button className="ai-memory-clear" onClick={clearConversation}>Clear this conversation’s memory</button>}
    {error && <p className="ai-companion-error" role="alert">{error}</p>}
  </section>;
}
