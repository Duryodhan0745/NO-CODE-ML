import React from 'react';

export default class ErrorBoundary extends React.Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('Application rendering failed', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <main role="alert" className="min-h-screen bg-slate-950 p-8 text-slate-100">
          <section className="mx-auto max-w-xl rounded-xl border border-red-500/60 bg-slate-900 p-6">
            <h1 className="text-xl font-bold">This screen could not be displayed</h1>
            <p className="mt-2 text-slate-300">Your current workflow data has not been intentionally cleared. Reload to try again.</p>
            <button type="button" className="mt-5 rounded bg-blue-600 px-4 py-2 font-semibold" onClick={() => window.location.reload()}>Reload application</button>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}
