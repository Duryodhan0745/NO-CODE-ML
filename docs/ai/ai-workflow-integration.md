# AI and ML workflow integration

The AI conversation routes now read the authoritative workspace and dataset state from the existing `services` module. `ai.workflow_context.build_workflow_context` assembles bounded metadata, a five-row preview, EDA statistics, current configuration, training split details, evaluation metrics, optimization output, historical runs, and controlled artifact URLs. The typed `WorkflowStage` envelope carries workspace, dataset, workflow and stage IDs, status, output, execution time, configuration version, artifact references, and error information.

Each assistant message receives this context before the provider generates an answer. The provider is instructed to separate completed evidence from configured options and historical runs, and to report unsupported or unexecuted stages honestly. The assistant can call targeted context tools rather than receiving a raw dataset.

## Workflow stages and evidence

| Stage | Existing source | Context status/output |
| --- | --- | --- |
| Dataset ingestion | `services.upload_dataset` | Metadata, validation checks, dtypes, missing and duplicate counts |
| EDA | `services.get_analysis` | Column summaries, numeric summaries, categorical top values and correlation matrix |
| Preprocessing | `services.configure_workspace` and training pipeline | Configured options; applied options and train/test feature details only after training |
| Model selection/configuration | `MODEL_REGISTRY`, `services.configure_workspace` | Supported algorithms, selected model, target, task and hyperparameters |
| Training | `services.train_workspace` | Run ID/status/time, split sizes, features, duration, metrics and artifacts |
| Evaluation | `services.get_evaluation` | Actual stored metrics and prediction artifact reference |
| Optimization | `services.optimize_workspace` | Existing best parameters, score, strategy and stored optimization result |
| Final summary | Current workspace state | Current model/configuration/metrics/artifacts; historical runs labeled separately |

Reconfiguration clears current metrics, artifacts, run ID and optimization. Previous runs stay in the run history and are marked historical in AI context. This prevents the assistant from presenting an earlier evaluation as current.

## Tool inventory

The existing allowlist retains dataset inspection, preview, analysis, correlation, VIF, profiling, model listing/details, workspace create/read/configure/train/optimize, evaluation, runs, predictions and artifacts. Added context tools are `get_workflow_context`, `get_stage_output`, `get_preprocessing_results`, `get_model_configuration`, `get_training_results`, `get_evaluation_results`, `get_optimization_results`, and `get_final_workflow_summary`. Every execution still passes through `ToolRegistry`, existing service validation, ownership checks and idempotency recording.

## Auto mode and recovery

The chat mode selector supports Manual and Auto. Auto asks the provider for one next allowlisted operation, executes it through the existing registry, refreshes workflow state, and then asks for the next decision. The loop is capped at eight operations, stops on repeated operations, and feeds actual failed results back into context so the assistant can explain or choose a safe correction. In local Open Access development, existing training approval checks are bypassed. Other environments preserve approval requirements.

The actual ML engine currently supports classification and regression only. Clustering and time-series forecasting are not implemented and are disclosed as unsupported. Per-stage training logs, residual summaries, target-specific EDA, and automatic candidate-target selection are also not emitted by current services; the assistant is not given fabricated replacements.

## Synchronization and endpoints

Configuration, training, optimization, workspace creation and tool operations append durable workflow events to SQLite. `GET /api/workspaces/{workspace_id}/events` supports live UI activity; the companion polls it every five seconds while the workflow is mounted. `GET /api/workspaces/{workspace_id}/ai-context` exposes the current structured context. Conversations persist both messages and tool execution records.




