import React, { useEffect, useRef, useState } from 'react';
import Workspace from './Workspace';
import AIMemoryManager from './AIMemoryManager';
import { getAIAuthConfig, loginTestAdmin } from './api';

const navItems = [
  ['⌂', 'Home'], ['◷', 'Conversations'], ['◌', 'Projects'], ['▤', 'Datasets'],
];
const starters = [
  { icon: '▥', title: 'Classify a dataset', detail: 'Predict a category from your data', prompt: 'Help me build a classification model' },
  { icon: '↗', title: 'Predict a value', detail: 'Explore a regression workflow', prompt: 'Help me predict a numeric value' },
  { icon: '◉', title: 'Explore my data', detail: 'Upload a dataset to get started', prompt: 'I want to explore a dataset' },
  { icon: '⌁', title: 'Forecast a series', detail: 'Plan a time based prediction', prompt: 'Help me forecast a time series' },
];

function Orb({ small = false }) {
  return <span className={`ai-orb${small ? ' ai-orb-small' : ''}`} aria-hidden="true" />;
}

export default function AIWorkspace({ onOpenPlayground, onContextUpdate, onDatasetHandoff, externalDataset }) {
  const [started, setStarted] = useState(() => Boolean(window.localStorage.getItem('active_workspace_id')));
  const [prompt, setPrompt] = useState(() => window.localStorage.getItem('active_workspace_goal') || '');
  const [activeNav, setActiveNav] = useState('Home');
  const [datasetFile, setDatasetFile] = useState(null);
  const fileRef = useRef(null);
  const googleButtonRef = useRef(null);
  const [authConfig, setAuthConfig] = useState({ clientId: '', openAccess: false, loading: true });
  const [authError, setAuthError] = useState('');
  const [signedIn, setSignedIn] = useState(() => Boolean(
    window.sessionStorage.getItem('ai_auth_token') || window.localStorage.getItem('ai_auth_token')
  ));
  const [adminEmail, setAdminEmail] = useState('');
  const [adminPassword, setAdminPassword] = useState('');
  const [adminLoginBusy, setAdminLoginBusy] = useState(false);

  const signInTestAdmin = async e => {
    e.preventDefault(); setAdminLoginBusy(true); setAuthError('');
    try {
      const result = await loginTestAdmin(adminEmail, adminPassword);
      window.sessionStorage.setItem('ai_auth_token', result.access_token);
      setAdminPassword(''); setSignedIn(true);
    } catch (error) { setAuthError(error.message); }
    finally { setAdminLoginBusy(false); }
  };

  useEffect(() => {
    let active = true;
    getAIAuthConfig()
      .then(config => {
        if (active) {
          const openAccess = Boolean(config.open_access_mode);
          setAuthConfig({ clientId: config.google_client_id || '', openAccess, loading: false });
          if (openAccess) {
            // Confirm the backend's fixed local testing identity; no token entry
            // or client-selected identity is involved.
            fetch('/api/auth/me').then(response => {
              if (!response.ok) throw new Error('Local testing identity is unavailable.');
              return response.json();
            }).then(() => { if (active) setSignedIn(true); }).catch(error => {
              if (active) setAuthError(error.message);
            });
          }
        }
      })
      .catch(error => {
        if (active) {
          setAuthConfig({ clientId: '', openAccess: false, loading: false });
          setAuthError(error.message);
        }
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!authConfig.clientId || started || signedIn) return undefined;

    const renderGoogleButton = () => {
      const identity = window.google?.accounts?.id;
      if (!identity || !googleButtonRef.current) return;
      identity.initialize({
        client_id: authConfig.clientId,
        callback: credential => {
          if (!credential.credential) {
            setAuthError('Google sign-in did not return an ID token.');
            return;
          }
          window.sessionStorage.setItem('ai_auth_token', credential.credential);
          window.localStorage.removeItem('ai_auth_token');
          setAuthError('');
          setSignedIn(true);
        },
      });
      googleButtonRef.current.replaceChildren();
      identity.renderButton(googleButtonRef.current, {
        theme: 'outline',
        size: 'large',
        text: 'signin_with',
        shape: 'pill',
      });
    };

    let script = document.getElementById('google-identity-services');
    if (window.google?.accounts?.id) {
      renderGoogleButton();
      return undefined;
    }
    if (!script) {
      script = document.createElement('script');
      script.id = 'google-identity-services';
      script.src = 'https://accounts.google.com/gsi/client';
      script.async = true;
      script.defer = true;
      document.head.appendChild(script);
    }
    const handleLoad = () => renderGoogleButton();
    const handleError = () => setAuthError('Google sign-in could not load. Check your connection and try again.');
    script.addEventListener('load', handleLoad);
    script.addEventListener('error', handleError);
    return () => {
      script.removeEventListener('load', handleLoad);
      script.removeEventListener('error', handleError);
    };
  }, [authConfig.clientId, signedIn, started]);

  const signOut = () => {
    window.sessionStorage.removeItem('ai_auth_token');
    window.localStorage.removeItem('ai_auth_token');
    window.google?.accounts?.id?.disableAutoSelect();
    setSignedIn(false);
    setAuthError('');
  };

  const begin = (text = prompt) => {
    if (text.trim()) {
      setPrompt(text.trim());
      window.localStorage.setItem('active_workspace_goal', text.trim());
    }
    setStarted(true);
  };

  if (started) return (
    <div className="ai-flow-screen">
      <header className="ai-flow-header">
        <button className="ai-back" onClick={() => setStarted(false)}>← <span>AI home</span></button>
        <div className="ai-flow-title"><Orb small /> <span>New ML project</span></div>
        <span className="ai-flow-context">Your existing ML workflow</span>
        {onOpenPlayground && (
          <button
            type="button"
            onClick={onOpenPlayground}
            style={{ marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '7px 14px', borderRadius: '8px', border: '1px solid #29435f', background: 'transparent', color: '#9eb2c9', fontSize: '12px', fontWeight: 600, cursor: 'pointer' }}
          >
            ▤ Data Playground
          </button>
        )}
      </header>
      <Workspace initialGoal={prompt} initialFile={datasetFile} initialWorkspaceId={window.localStorage.getItem('active_workspace_id')} onOpenPlayground={onOpenPlayground} onContextUpdate={onContextUpdate} onDatasetHandoff={onDatasetHandoff} externalDataset={externalDataset} />
    </div>
  );

  return (
    <main className="ai-home">
      <aside className="ai-rail" aria-label="Workspace navigation">
        <button className="ai-brand" aria-label="No-Code ML home" onClick={() => setActiveNav('Home')}><span>✳</span></button>
        <div className="ai-rail-nav">
          {navItems.map(([icon, label]) => <button key={label} title={label} aria-label={label} aria-current={activeNav === label ? 'page' : undefined} className={`ai-rail-button${activeNav === label ? ' active' : ''}`} onClick={() => setActiveNav(label)}><span>{icon}</span></button>)}
        </div>
        <div className="ai-rail-bottom">
          <button className="ai-rail-button" title="Settings" aria-label="Settings" onClick={() => setActiveNav('Settings')}><span>⚙</span></button>
          <span className="ai-avatar" aria-hidden="true">J</span>
        </div>
      </aside>

      <section className="ai-home-main">
        <header className="ai-topbar">
          <div className="ai-mode-label"><span>✧</span> AI workspace</div>
          <div className="ai-top-actions">
            <button className="ai-outline-button" onClick={() => setActiveNav('Conversations')}><span>◷</span> Conversations</button>
            <button className="ai-primary-button" onClick={() => { window.localStorage.removeItem('active_workspace_id'); window.localStorage.removeItem('active_workspace_goal'); setPrompt(''); setDatasetFile(null); if (fileRef.current) fileRef.current.value = ''; setStarted(true); }}>＋ <span>New project</span></button>
          </div>
        </header>

        <div className="ai-content">
          {activeNav === 'Settings' ? <AIMemoryManager /> : activeNav !== 'Home' ? (
            <section className="ai-empty-section">
              <Orb small />
              <p className="ai-eyebrow">{activeNav.toUpperCase()}</p>
              <h1>{activeNav === 'Datasets' ? 'Bring your data into a workflow' : `Your ${activeNav.toLowerCase()} will appear here`}</h1>
              <p className="ai-subtitle">{activeNav === 'Datasets' ? 'Dataset management is available inside the ML workflow.' : `${activeNav} are not connected to a saved backend yet.`}</p>
              <button className="ai-primary-button" onClick={() => activeNav === 'Datasets' ? setStarted(true) : setActiveNav('Home')}>{activeNav === 'Datasets' ? 'Open ML workflow' : 'Back to home'}</button>
            </section>
          ) : <>
          <div className="ai-welcome">
            <Orb />
            <p className="ai-eyebrow">YOUR MACHINE LEARNING WORKSPACE</p>
            <h1>What would you like<br className="desktop-break" /> to <em>build</em> today?</h1>
            <p className="ai-subtitle">Describe your goal. We’ll help you shape a workflow around your data.</p>
          </div>

          <div className="ai-auth-panel" aria-live="polite">
            {authConfig.openAccess && <div className="ai-open-access-banner" role="status">Local Open Access Testing Mode active · ready to explore</div>}
            {authConfig.openAccess ? null : signedIn ? (
              <>
                <span className="ai-auth-status">Google account connected</span>
                <button className="ai-auth-signout" onClick={signOut}>Sign out</button>
              </>
            ) : authConfig.loading ? (
              <span className="ai-auth-note">Checking sign-in configuration…</span>
            ) : authConfig.clientId ? (
              <div ref={googleButtonRef} aria-label="Sign in with Google" />
            ) : (
              <>
                <span className="ai-auth-note">Google sign-in needs a client ID; the ML workflow remains available.</span>
                <form className="ai-test-admin-login" onSubmit={signInTestAdmin}>
                  <label><span className="sr-only">Test admin email</span><input type="email" value={adminEmail} onChange={e => setAdminEmail(e.target.value)} placeholder="Test admin email" autoComplete="username" required /></label>
                  <label><span className="sr-only">Test admin password</span><input type="password" value={adminPassword} onChange={e => setAdminPassword(e.target.value)} placeholder="Test admin password" autoComplete="current-password" required /></label>
                  <button className="ai-auth-signout" disabled={adminLoginBusy}>{adminLoginBusy ? 'Signing in…' : 'Test admin sign in'}</button>
                </form>
              </>
            )}
            {authError && <p className="ai-auth-error" role="alert">{authError}</p>}
          </div>

          <form className="ai-composer" onSubmit={e => { e.preventDefault(); begin(); }}>
            <div className="ai-composer-input">
              <span className="ai-sparkle">✦</span>
              <textarea aria-label="Describe your machine learning goal" value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Describe what you want to predict or explore…" rows={2} />
            </div>
            {datasetFile && <div className="ai-file-chip"><span>▤</span>{datasetFile.name}<button type="button" aria-label="Remove attached file" onClick={() => { setDatasetFile(null); if (fileRef.current) fileRef.current.value = ''; }}>×</button></div>}
            <div className="ai-composer-footer">
              <div className="ai-composer-tools">
                <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={e => setDatasetFile(e.target.files?.[0] || null)} />
                <button className="ai-tool-button" type="button" onClick={() => fileRef.current?.click()}><span>⌁</span> Add dataset</button>
                <span className="ai-tool-note">CSV · uploads when you start</span>
              </div>
              <div className="ai-composer-submit">
                <button className="ai-send" type="submit" aria-label="Start workflow" disabled={!prompt.trim() && !datasetFile}>↑</button>
              </div>
            </div>
          </form>

          <div className="ai-starters-heading"><span>START WITH A GOAL</span><i /></div>
          <div className="ai-starters">
            {starters.map(card => <button key={card.title} className="ai-starter-card" onClick={() => begin(card.prompt)}><span className="ai-starter-icon">{card.icon}</span><span className="ai-starter-title">{card.title}</span><span className="ai-starter-detail">{card.detail}</span><span className="ai-starter-arrow">↗</span></button>)}
          </div>
          </>}
        </div>
      </section>
    </main>
  );
}
