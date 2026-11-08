import React, { useEffect, useRef, useState } from 'react';
import { claimAIWorkspace, createAIConversation, decideWorkflowProposal, getAIContext, getWorkflowEvents, listWorkflowProposals, streamAIMessage } from './api';
import AIMemoryManager from './AIMemoryManager';
import MarkdownMessage from './MarkdownMessage';

function RobotAvatar({ state = 'idle' }) {
  return <span className={`ai-robot ai-robot-${state}`} aria-hidden="true">
    <img src="/assets/ml-assistant-robot.png" alt="" draggable="false" />
  </span>;
}

const DEFAULT_POSITION = { x: 24, y: 24 };

function readPosition() {
  try {
    const saved = JSON.parse(localStorage.getItem('ai_companion_position') || 'null');
    if (Number.isFinite(saved?.x) && Number.isFinite(saved?.y)) return saved;
  } catch {
    localStorage.removeItem('ai_companion_position');
  }
  return DEFAULT_POSITION;
}

export default function AICompanion({ workspaceId, onFillConfiguration }) {
  const [mode, setMode] = useState('companion');
  const [conversationId, setConversationId] = useState(() => localStorage.getItem(`ai_conversation_${workspaceId || 'new'}`));
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [workflowMode, setWorkflowMode] = useState('manual');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [attention, setAttention] = useState(false);
  const [workflowContext, setWorkflowContext] = useState(null);
  const [showMemoryManager, setShowMemoryManager] = useState(false);
  const [position, setPosition] = useState(readPosition);
  const [repositioning, setRepositioning] = useState(false);
  const [proposal, setProposal] = useState(null);
  const drag = useRef(null);
  const companionRef = useRef(null);
  const messagesRef = useRef(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    if (!conversationId) return;
    getAIContext(conversationId).then(data => setMessages(data.messages || [])).catch(e => setError(e.message));
  }, [conversationId]);

  useEffect(() => {
    if (stickToBottom.current && messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    localStorage.setItem('ai_companion_position', JSON.stringify(position));
  }, [position]);

  useEffect(() => {
    const clampPosition = () => setPosition(current => {
      const rect = companionRef.current?.getBoundingClientRect();
      const width = rect?.width || 118;
      const height = rect?.height || 118;
      return {
        x: Math.max(8, Math.min(window.innerWidth - width - 8, current.x)),
        y: Math.max(8, Math.min(window.innerHeight - height - 8, current.y)),
      };
    });
    window.addEventListener('resize', clampPosition);
    return () => window.removeEventListener('resize', clampPosition);
  }, []);

  useEffect(() => {
    if (!workspaceId) return undefined;
    let active = true;
    const refresh = () => {
      getWorkflowEvents(workspaceId).then(data => {
        if (active) setWorkflowContext(data);
      }).catch(() => {});
      listWorkflowProposals(workspaceId).then(data => {
        if (active && data.proposals?.[0]) setProposal(data.proposals[0]);
      }).catch(() => {});
    };
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, [workspaceId]);

  const start = async () => {
    setBusy(true); setError('');
    try {
      if (workspaceId) await claimAIWorkspace(workspaceId);
      const conversation = await createAIConversation(workspaceId || null);
      setConversationId(conversation.id);
      localStorage.setItem(`ai_conversation_${workspaceId || 'new'}`, conversation.id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  const submit = async e => {
    e.preventDefault(); if (!draft.trim() || !conversationId || busy) return;
    const content = draft.trim(); setDraft(''); setBusy(true); setError('');
    const userId = `local-user-${Date.now()}`;
    const assistantId = `local-assistant-${Date.now()}`;
    setMessages(current => [...current, { id: userId, role: 'user', content }, { id: assistantId, role: 'assistant', content: '', streaming: true, toolEvents: [] }]);
    try {
      await streamAIMessage(conversationId, content, workflowMode, event => {
        if (event.type === 'delta') {
          setMessages(current => current.map(message => message.id === assistantId
            ? { ...message, content: message.content + event.text }
            : message));
        } else if (event.type === 'tool') {
          setMessages(current => current.map(message => message.id === assistantId
            ? { ...message, toolEvents: [...message.toolEvents, event] }
            : message));
        } else if (event.type === 'complete') {
          setMessages(current => current.map(message => message.id === assistantId
            ? { ...event.assistant, toolEvents: message.toolEvents, streaming: false }
            : message));
          if (event.proposal) setProposal(event.proposal);
        } else if (event.type === 'error') {
          setMessages(current => current.map(message => message.id === assistantId
            ? { ...message, content: message.content || 'I couldn’t complete that response. Please retry.', streaming: false }
            : message));
        }
      });
    } catch (err) {
      setError(err.message);
      setMessages(current => current.map(message => message.id === assistantId
        ? { ...message, streaming: false, interrupted: true }
        : message));
    }
    finally { setBusy(false); }
  };
  const decideProposal = async decision => {
    if (!proposal || busy) return;
    setBusy(true); setError('');
    try {
      if (!conversationId) await start();
      const result = await decideWorkflowProposal(proposal.proposal_id, decision, proposal.configuration_fingerprint);
      const resolved = result.proposal;
      if (decision === 'fill_only') onFillConfiguration?.(resolved);
      setProposal(resolved);
      const actualOutput = resolved.execution?.output;
      const executionText = actualOutput ? `Executed: ${resolved.status}. Actual backend result: ${JSON.stringify(actualOutput).slice(0, 1800)}` : `Executed: ${resolved.status}.`;
      setMessages(current => [...current, { id: `proposal-${Date.now()}`, role: 'assistant', content: decision === 'execute' ? executionText : decision === 'fill_only' ? 'Configuration filled into the workflow. No stage was executed.' : 'Proposal cancelled. No stage was executed.' }]);
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  };
  const savePosition = next => {
    setPosition(next);
    localStorage.setItem('ai_companion_position', JSON.stringify(next));
  };
  const onPointerDown = e => {
    if (mode === 'focus' || e.target.closest('button, input, textarea, select, a')) return;
    const rect = companionRef.current?.getBoundingClientRect();
    drag.current = {
      x: e.clientX,
      y: e.clientY,
      right: window.innerWidth - (rect?.right || window.innerWidth - position.x),
      bottom: window.innerHeight - (rect?.bottom || window.innerHeight - position.y),
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = e => {
    if (!drag.current) return;
    const rect = companionRef.current?.getBoundingClientRect();
    const width = rect?.width || 118;
    const height = rect?.height || 118;
    savePosition({
      x: Math.max(8, Math.min(window.innerWidth - width - 8, drag.current.right - (e.clientX - drag.current.x))),
      y: Math.max(8, Math.min(window.innerHeight - height - 8, drag.current.bottom - (e.clientY - drag.current.y))),
    });
  };
  const endDrag = () => { drag.current = null; };
  const handleHeaderDoubleClick = e => {
    if (e.target.closest('button, input, textarea, select, a')) return;
    setRepositioning(active => !active);
  };
  useEffect(() => {
    const element = companionRef.current;
    if (!element || !repositioning || mode === 'focus') return;
    const handleWheel = event => {
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      const horizontal = event.shiftKey ? (event.deltaY || event.deltaX) : 0;
      const vertical = event.shiftKey ? 0 : (event.deltaY || event.deltaX);
      setPosition(current => {
        const next = {
          x: Math.max(8, Math.min(window.innerWidth - rect.width - 8, current.x - horizontal)),
          y: Math.max(8, Math.min(window.innerHeight - rect.height - 8, current.y - vertical)),
        };
        localStorage.setItem('ai_companion_position', JSON.stringify(next));
        return next;
      });
    };
    element.addEventListener('wheel', handleWheel, { passive: false });
    return () => element.removeEventListener('wheel', handleWheel);
  }, [repositioning, mode]);

  const avatarState = busy ? (conversationId && messages.at(-1)?.role === 'user' ? 'thinking' : 'speaking') : attention ? 'attention' : 'idle';
  if (mode === 'minimized') return <button ref={companionRef} className={`ai-companion-orb ai-companion-minimized${repositioning ? ' is-repositioning' : ''}`} style={{ right: position.x, bottom: position.y }} onDoubleClick={() => setRepositioning(active => !active)} onClick={() => { setMode('expanded'); setAttention(false); }} aria-label="Open AI companion"><RobotAvatar state={avatarState} /></button>;
  const open = mode === 'expanded' || mode === 'focus';
  return <section ref={companionRef} className={`ai-companion ${mode}${repositioning ? ' is-repositioning' : ''}`} style={mode !== 'focus' ? { right: position.x, bottom: position.y } : undefined} aria-label="AI companion">
    <header className={`ai-companion-head ${mode !== 'focus' ? 'ai-companion-drag-handle' : ''}`} onDoubleClick={handleHeaderDoubleClick} onPointerDown={mode !== 'focus' ? onPointerDown : undefined} onPointerMove={mode !== 'focus' ? onPointerMove : undefined} onPointerUp={endDrag}>
      <button className="ai-companion-orb" onClick={() => { setMode(open ? 'companion' : 'expanded'); setAttention(false); }} aria-label={open ? 'Minimize AI companion' : 'Open AI companion'}><RobotAvatar state={avatarState} /></button>
      {mode === 'companion' && <span className="ai-companion-move-grip" title="Drag or double-click to reposition" aria-label="Move assistant">⋮⋮</span>}
      {open && <><strong>ML Assistant</strong>{repositioning && <span className="ai-companion-move-status">Repositioning · scroll to move</span>}<span className="ai-companion-actions"><button onClick={() => { savePosition(DEFAULT_POSITION); setRepositioning(false); }} aria-label="Reset assistant position" title="Reset position">⌖</button><button onClick={() => setShowMemoryManager(v => !v)} aria-label="Manage assistant memory" title="Memory">M</button><button onClick={() => setMode(mode === 'focus' ? 'expanded' : 'focus')}>{mode === 'focus' ? '↙' : '↗'}</button><button onClick={() => setMode('companion')} aria-label="Minimize assistant">−</button></span></>}
    </header>
    {open && showMemoryManager ? <div className="ai-companion-body"><AIMemoryManager workspaceId={workspaceId} conversationId={conversationId} onConversationCleared={() => setMessages([])} onClose={() => setShowMemoryManager(false)} /></div> : open && <div className="ai-companion-body">
      {workflowContext && <div className="ai-workflow-live" role="status">Workflow · {workflowContext.workspace_status} · {workflowContext.events.at(-1)?.payload?.stage_name || workflowContext.events.at(-1)?.event_type || 'state synced'} {workflowContext.events.at(-1)?.payload?.execution_status || ''}</div>}
      {proposal?.status === 'awaiting_approval' && <section className="ai-approval-card" aria-label="Workflow approval">
        <span className="ai-approval-stage">{proposal.stage_id.replaceAll('_', ' ')}</span>
        <p>{proposal.rationale}</p>
        <pre>{JSON.stringify(proposal.configuration, null, 2)}</pre>
        <p className="ai-approval-cost">Rows: {proposal.resource_usage.rows ?? 'unavailable'} · Features: {proposal.resource_usage.features ?? 'unavailable'} · Runtime: {proposal.resource_usage.estimated_runtime_seconds ?? 'unavailable'} {proposal.resource_usage.estimate_is_approximate ? '(estimate)' : ''}</p>
        {proposal.warnings?.length > 0 && <p>{proposal.warnings.join(' ')}</p>}
        <div className="ai-approval-actions"><button onClick={() => decideProposal('execute')} disabled={busy}>YES — Execute</button><button onClick={() => decideProposal('cancel')} disabled={busy}>NO — Cancel</button><button onClick={() => decideProposal('fill_only')} disabled={busy}>JUST FILL</button></div>
      </section>}
      {!conversationId ? <button className="ai-primary-button" disabled={busy} onClick={start}>{busy ? 'Connecting…' : 'Start a conversation'}</button> : <>
        <div ref={messagesRef} className="ai-messages" aria-live="polite" onScroll={e => { const element = e.currentTarget; stickToBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48; }}>
          {messages.map(m => <div key={m.id} className={`ai-message ${m.role}`}>
            {m.role === 'assistant' && <RobotAvatar state={m.streaming ? 'thinking' : 'idle'} />}
            <div className="ai-message-content">
              <b>{m.role === 'user' ? 'You' : 'AI'}</b>
              {m.content ? <MarkdownMessage content={m.content} /> : m.streaming ? <span className="ai-generating">Generating response…</span> : null}
              {m.toolEvents?.length > 0 && <div className="ai-tool-events" aria-label="Tool execution status">{m.toolEvents.map((tool, index) => <span key={`${tool.tool_name}-${index}`} data-status={tool.status}>{tool.tool_name.replaceAll('_', ' ')} · {tool.status}</span>)}</div>}
              {m.interrupted && <small className="ai-stream-interrupted">Stream interrupted. The partial answer is preserved; retry your question to continue.</small>}
            </div>
          </div>)}
          {busy && !messages.at(-1)?.streaming && <p className="ai-message assistant"><RobotAvatar state="thinking" /><span><b>AI</b>Working…</span></p>}
        </div>
        <form onSubmit={submit} className="ai-chat-form"><label className="sr-only" htmlFor="ai-message">Message AI assistant</label><textarea id="ai-message" value={draft} onChange={e => setDraft(e.target.value)} placeholder="Ask about this workflow…" rows="2"/><div className="ai-chat-submit"><label className="sr-only" htmlFor="ai-workflow-mode">Assistant mode</label><select id="ai-workflow-mode" value={workflowMode} onChange={e => setWorkflowMode(e.target.value)} aria-label="Assistant mode"><option value="manual">Manual</option><option value="auto">Auto</option></select><button disabled={busy || !draft.trim()}>Send</button></div></form>
      </>}
      {error && <p className="ai-companion-error" role="alert">{error}</p>}
    </div>}
  </section>;
}
