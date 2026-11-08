from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from pathlib import Path

OUT = Path('No_Code_ML_Project_Research_Paper.docx')

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)

def borders(cell, color='D9D9D9', size='6'):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.first_child_found_in('w:tcBorders')
    if b is None:
        b = OxmlElement('w:tcBorders'); tcPr.append(b)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        tag = 'w:' + edge; el = b.find(qn(tag))
        if el is None: el = OxmlElement(tag); b.append(el)
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), size); el.set(qn('w:color'), color)

def set_cell_text(cell, text, bold=False, color='000000', size=9):
    cell.text = ''
    p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(2)
    r = p.add_run(str(text)); r.bold = bold; r.font.name = 'Aptos'; r.font.size = Pt(size); r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    borders(cell)

def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers)); table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, color='FFFFFF', size=9); shade(table.rows[0].cells[i], '63452F')
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, size=8.5)
            if ridx % 2 == 1: shade(cells[i], 'F5F1EB')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet' if level == 0 else 'List Bullet 2')
    p.add_run(text); return p

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(.7); sec.bottom_margin = Inches(.65); sec.left_margin = Inches(.85); sec.right_margin = Inches(.85)

styles = doc.styles
styles['Normal'].font.name = 'Aptos'; styles['Normal'].font.size = Pt(10.5); styles['Normal'].font.color.rgb = RGBColor(48,40,32)
styles['Normal'].paragraph_format.space_after = Pt(7); styles['Normal'].paragraph_format.line_spacing = 1.12
for name, size, color in [('Title', 24, '000000'), ('Heading 1', 16, '000000'), ('Heading 2', 12.5, '000000')]:
    st = styles[name]; st.font.name = 'Aptos'; st.font.size = Pt(size); st.font.bold = True; st.font.color.rgb = RGBColor.from_string(color)
    st.paragraph_format.space_before = Pt(14 if name == 'Heading 1' else 9); st.paragraph_format.space_after = Pt(5)
if 'Small Metadata' not in styles:
    st = styles.add_style('Small Metadata', WD_STYLE_TYPE.PARAGRAPH); st.font.name = 'Aptos'; st.font.size = Pt(9); st.font.color.rgb = RGBColor(89,75,64)

header = sec.header.paragraphs[0]; header.text = 'No Code Machine Learning Platform  |  Project Research Document'; header.style = 'Small Metadata'; header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer = sec.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER; footer.add_run('Research context draft  •  Repository-grounded implementation study')

p = doc.add_paragraph(style='Title'); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run('No Code Machine Learning Platform Research Paper')
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = p.add_run('Technical research context and preliminary system study'); r.italic = True; r.font.size = Pt(13); r.font.color.rgb = RGBColor(99,69,47)
p = doc.add_paragraph(style='Small Metadata'); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run('Prepared from the repository implementation and the supplied research context brief')

doc.add_paragraph('Document status: This is a project research document and technical context package. It is not a completed literature review, competitor benchmark, or claim of production readiness. Statements about the current system are based on repository inspection; proposed experiments are explicitly marked as unexecuted.')

doc.add_heading('Abstract', level=1)
doc.add_paragraph('This document studies a browser-based no-code machine learning platform implemented with a React and Vite frontend and a FastAPI, pandas, and scikit-learn backend. The platform guides a user through CSV upload, dataset inspection, target and problem selection, preprocessing, model configuration, training, evaluation, optimization, artifact export, and batch prediction. The implementation provides a unified model registry, leakage-aware preprocessing, real estimator execution, Optuna TPE Bayesian optimization, GridSearchCV, RandomizedSearchCV, local artifact storage, and a persisted JSON workspace state. The study finds a technically meaningful foundation for accessible tabular machine learning, but it also identifies clear limits: clustering and several advanced preprocessing operations are incomplete, durable database-backed multi-user persistence is absent, conversational AI and long-term memory are not implemented, and end-to-end verification is constrained by the current test environment. The document defines the evidence needed before making claims about usability, reliability, or advantage over existing platforms.')
doc.add_paragraph('Keywords: no-code machine learning, AutoML, reproducibility, preprocessing leakage, model optimization, Bayesian optimization, human-centered machine learning, workflow reliability')

doc.add_heading('1 Introduction and Problem Statement', level=1)
doc.add_paragraph('Building a reliable machine learning workflow requires more than selecting an estimator. A practitioner must validate a dataset, understand its structure, choose a target, handle missing and categorical values, configure transformations, select a model, tune parameters without contaminating the test set, evaluate the result, and preserve the exact pipeline used for inference. These tasks are particularly difficult for users who understand a domain problem but do not routinely write Python or reason about scikit-learn pipeline semantics.')
doc.add_paragraph('The platform examined here addresses this workflow problem through a visual interface for tabular classification and regression. Its research value is not established by the existence of a polished interface. It must instead be assessed through functional correctness, reproducibility, transparency of transformations, error handling, and evidence that users can complete valid workflows with fewer avoidable mistakes.')
doc.add_paragraph('The supplied context brief also proposes conversational assistance and persistent memory as a possible research direction. Repository inspection does not show an implemented LLM orchestration layer, provider integration, conversation store, or long-term memory subsystem. Those elements therefore remain research hypotheses and future work rather than current system capabilities.')

doc.add_heading('2 Research Questions and Scope', level=1)
doc.add_paragraph('The following questions organize this project study. They are intentionally phrased as questions rather than conclusions.')
for q in ['What technical and usability problem does a guided no-code tabular ML workflow address?', 'Which parts of dataset preparation, model building, optimization, evaluation, and inference are implemented and verifiable in the current repository?', 'How does the system preserve preprocessing correctness and prevent leakage during training and cross-validation?', 'What evidence would be required to compare the platform fairly with visual ML tools, AutoML libraries, and cloud products?', 'Which capabilities remain incomplete, and how do they constrain claims about the platform?', 'Could context-aware conversational assistance improve workflow continuity while preserving user visibility and control?']:
    add_bullet(doc, q)
doc.add_paragraph('The scope of this document is the current repository implementation. Literature claims, competitor capabilities, user-study results, and performance comparisons are not asserted unless directly evidenced in the repository or marked as proposed work.')

doc.add_heading('3 System Implementation', level=1)
doc.add_heading('3.1 Architecture', level=2)
doc.add_paragraph('The frontend is a React application built with Vite and Tailwind CSS. The backend is a FastAPI service with thin HTTP routes and business logic in services.py. The ML engine in ml.py owns the centralized model registry, estimator factory, parameter validation, preprocessing construction, and metric computation. optimizer.py contains cross-validation optimization logic. Uploaded datasets are stored beneath the configured upload directory; trained models, preprocessing artifacts, metadata, metrics, configurations, and prediction CSVs are stored beneath the artifact directory. Workspace and dataset metadata are persisted to a JSON state file configured through STATE_DIR.')
add_table(doc, ['Layer', 'Actual implementation', 'Evidence'], [
    ['Interface', 'React, Vite, Tailwind; ten-stage workflow in Workspace.jsx', 'frontend/package.json; frontend/src/Workspace.jsx'],
    ['API', 'FastAPI routes for upload, profiling, models, workspaces, training, optimization, evaluation, prediction, and artifacts', 'backend/main.py'],
    ['Services', 'Dataset validation, analysis, workspace state, training, optimization orchestration, artifacts, inference', 'backend/services.py'],
    ['ML engine', 'scikit-learn pipelines, XGBoost, SVM, gradient and linear estimators, metrics', 'backend/ml.py'],
    ['Optimization', 'Optuna TPE, GridSearchCV, RandomizedSearchCV', 'backend/optimizer.py'],
    ['Persistence', 'Local files plus JSON workspace state', 'backend/config.py; backend/services.py'],
])

doc.add_heading('3.2 User Workflow', level=2)
doc.add_paragraph('The current interface presents ten sequential stages: upload, preview, analysis, target and problem selection, preprocessing, model selection, parameter configuration, training, evaluation, and prediction or download. Upload creates a dataset record and workspace. Configuration is sent to the backend before training. Training splits the dataset, fits preprocessing on the training partition, transforms the test partition, trains the selected estimator, computes task-appropriate metrics, and writes artifacts. Prediction reloads the stored preprocessor and estimator and transforms new data without refitting.')

doc.add_heading('4 Machine Learning Pipeline', level=1)
doc.add_heading('4.1 Dataset handling and profiling', level=2)
doc.add_paragraph('The upload service accepts CSV files, checks the extension, enforces a configured maximum size, rejects empty input, parses the file with pandas, rejects empty datasets and duplicate column names, and stores metadata such as shape, dtypes, missing counts, and duplicate rows. Preview responses are row-limited. Analysis reports numerical and categorical columns, missingness, descriptive statistics, top categorical values, and numerical correlations. Optional profiling uses ydata-profiling, and VIF analysis uses statsmodels for numerical columns.')
doc.add_heading('4.2 Preprocessing and leakage control', level=2)
doc.add_paragraph('The implemented pipeline supports numerical imputation with mean, median, or most-frequent strategies; categorical most-frequent imputation followed by OneHotEncoder with unknown-category tolerance; StandardScaler and MinMaxScaler; IQR clipping; optional PCA; and classification SMOTE applied only after the training split. The service fits the ColumnTransformer on training data and applies transform to the held-out test data. The Optuna objective rebuilds and fits preprocessing inside every cross-validation fold. These code paths provide a concrete leakage-control design, although broader preprocessing requirements such as configurable feature selection and time-aware splitting remain incomplete.')
doc.add_heading('4.3 Model catalog and evaluation', level=2)
doc.add_paragraph('The centralized registry includes classification and regression estimators spanning logistic and linear baselines, decision trees, random forests, gradient boosting, XGBoost, SVM, KNN, SGD, Ridge, Lasso, and ElasticNet. Parameter definitions include defaults, types, ranges, and categorical options. The factory validates submitted parameters before constructing estimators. Classification metrics include accuracy, precision, recall, F1, confusion matrix, and ROC AUC where probabilities are available. Regression metrics include MAE, MSE, RMSE, and R squared.')
doc.add_heading('4.4 Optimization', level=2)
doc.add_paragraph('Optuna optimization uses a seeded TPESampler and cross-validation over the training partition. The explicit Bayesian strategy is an alias around this TPE implementation and records strategy metadata as optuna_tpe. GridSearchCV and RandomizedSearchCV use a sklearn Pipeline containing preprocessing and the estimator, so each fold learns transformations from its fold-specific training data. The held-out test partition is not passed into model selection.')

doc.add_heading('5 Verification and Implementation Status', level=1)
doc.add_paragraph('Verification distinguishes source presence from executed evidence. The frontend production build passed. The ML test subset passed 26 tests when legacy assertions requiring exactly twelve models were excluded; those assertions conflict with the expanded model catalog. Bayesian, grid, and randomized optimization smoke tests completed with deterministic small datasets. The repository e2e collector attempts a localhost request, which is denied by the current sandbox socket policy; the full API and e2e workflow therefore remain environment-limited.')
add_table(doc, ['Capability', 'Status', 'Evidence or remaining issue'], [
    ['CSV upload, validation, preview', 'Implemented', 'services.py and API routes; broader live API execution is environment-limited'],
    ['Profiling, correlations, VIF', 'Implemented', 'services.py routes and frontend actions'],
    ['Leakage-safe preprocessing', 'Verified in ML tests', '26 relevant ML tests passed'],
    ['Model registry and parameter validation', 'Implemented', 'Central registry and factory in ml.py'],
    ['Optuna and Bayesian TPE optimization', 'Smoke-tested', 'Small deterministic optimization completed'],
    ['GridSearchCV and RandomizedSearchCV', 'Smoke-tested', 'Both strategies completed on a small classification dataset'],
    ['Artifacts and inference reuse', 'Implemented, broader verification pending', 'joblib artifacts and prediction route exist'],
    ['Persistent workspace metadata', 'Implemented, restart test pending', 'JSON state file in STATE_DIR'],
    ['Conversational AI and persistent memory', 'Unsupported in current repository', 'No provider, prompt, tool orchestration, or memory store found'],
    ['Clustering and advanced feature selection', 'Incomplete', 'No complete end-to-end implementation found'],
])

doc.add_heading('6 Conversational AI and Memory Audit', level=1)
doc.add_paragraph('The research brief contains a substantial conceptual discussion of LLM-assisted workflows, tool calling, context management, and memory. Those instructions must not be mistaken for implementation evidence. Repository inspection found no LLM provider client, prompt template, conversational API, tool-calling loop, structured model response validator, conversation table, vector store, memory extraction process, or persistent retrieval mechanism. The current application is therefore a visual and API-driven ML workspace, not a conversational AutoML agent.')
doc.add_paragraph('This absence is a useful boundary for the research paper. A future conversational layer could be evaluated only after it is implemented with explicit separation between recommendations and execution, user approval for consequential actions, provenance for tool outputs, and tests against fabricated metrics, invalid parameters, and stale context.')

doc.add_heading('7 Evaluation Plan', level=1)
doc.add_paragraph('No user study, competitor benchmark, or comparative accuracy experiment is reported in this document. The following experiments are proposed and must be executed before making effectiveness claims.')
add_table(doc, ['Experiment', 'Research question', 'Core measures'], [
    ['Pipeline correctness', 'Does the system produce the same transformations and metrics as a trusted reference pipeline?', 'Metric agreement, schema consistency, artifact reload success'],
    ['Leakage prevention', 'Are imputers, encoders, scalers, clipping, PCA, and SMOTE fitted only on allowed training data?', 'Fold inspection, sentinel distribution tests, held-out score integrity'],
    ['Optimization reliability', 'Do Optuna TPE, grid, and randomized search select valid configurations without using test data?', 'Trial validity, CV score, untouched test score, reproducibility'],
    ['Inference round trip', 'Can a saved artifact bundle transform new data and produce predictions after reload?', 'Prediction equality before and after reload, unknown-category handling'],
    ['Workflow usability', 'Can intended users complete a valid workflow without undocumented API calls?', 'Completion rate, time, errors, interpretation checks'],
    ['Conversational assistance', 'If implemented, does assistance use actual tool outputs and preserve user control?', 'Recommendation validity, tool-call success, unsupported-claim rate'],
    ['Performance', 'What resource and latency costs arise for realistic datasets?', 'Upload, profiling, training, optimization, prediction latency and memory'],
])

doc.add_heading('8 Discussion and Potential Contribution', level=1)
doc.add_paragraph('The current implementation’s strongest researchable contribution is an integrated, inspectable workflow for tabular ML rather than a claim of algorithmic novelty. It combines visual progression, explicit preprocessing choices, a centralized model registry, leakage-aware cross-validation, reproducible artifacts, and multiple optimization strategies in one small-stack application. These choices could support research into whether visible pipeline configuration and artifact traceability reduce workflow errors for users with limited ML programming experience.')
doc.add_paragraph('The evidence does not yet support claims that the platform is more accurate, more usable, more reliable, or more capable than commercial products, visual analytics tools, or AutoML libraries. Such claims require fair baselines, controlled datasets, repeatable tasks, and appropriate human or technical evaluation.')

doc.add_heading('9 Limitations and Future Work', level=1)
for item in ['Workspace metadata is file-backed JSON rather than a transactional database and has no multi-user authorization model.', 'Long-running training and optimization execute in the request path; background jobs, progress persistence, and cancellation are not complete.', 'Clustering, time-aware splitting, configurable feature selection, and several advanced preprocessing operations remain incomplete.', 'The test suite contains legacy assumptions about the original model count and an e2e script that requires localhost socket access.', 'No conversational AI, prompt orchestration, user approval workflow, or persistent memory subsystem is present.', 'Literature review and competitor comparison require external source verification and should be written as a separate evidence phase.']:
    add_bullet(doc, item)

doc.add_heading('10 Conclusion', level=1)
doc.add_paragraph('The repository contains a functioning foundation for a no-code tabular machine learning workspace. Its implemented workflow reaches real training, evaluation, optimization, artifact export, and inference, with concrete safeguards against preprocessing leakage in the principal training and cross-validation paths. The project is suitable for a research prototype study and for further engineering, but its research claims must remain bounded by the verification evidence. The next defensible phase is to complete the evaluation plan, repair or isolate environment-dependent tests, conduct a documented literature and competitor review, and only then write a final academic paper with measured results.')

doc.add_heading('References and Source Materials', level=1)
for i, ref in enumerate([
    'No-Code ML Platform repository, backend/main.py, backend/services.py, backend/ml.py, backend/optimizer.py, backend/schemas.py, and frontend/src/Workspace.jsx. Inspected for this document.',
    'No-Code ML Platform repository, README.md and no_code_ml_docs/09_TESTING_SPEC.md. Project documentation and testing expectations.',
    'no_code_ml_research_context_and_codex_prompt.txt. Supplied research brief and audit directive; treated as instruction and scope context, not as evidence of implementation.',
    'scikit-learn, pandas, FastAPI, XGBoost, Optuna, statsmodels, imbalanced-learn, and ydata-profiling dependencies declared in backend/requirements.txt.',
], 1):
    doc.add_paragraph(f'[{i}] {ref}')

doc.add_heading('Appendix A Implementation Evidence Map', level=1)
add_table(doc, ['Source file', 'Relevant symbols or responsibilities'], [
    ['backend/main.py', 'FastAPI application and HTTP routes'],
    ['backend/services.py', 'Dataset lifecycle, workspace configuration, training, optimization orchestration, prediction, artifacts, JSON persistence'],
    ['backend/ml.py', 'MODEL_REGISTRY, create_model, build_preprocessing_pipeline, compute_metrics'],
    ['backend/optimizer.py', 'run_optuna_optimization, run_bayesian_optimization, run_sklearn_search'],
    ['backend/config.py', 'Upload, artifact, state, and reproducibility configuration'],
    ['frontend/src/Workspace.jsx', 'Ten-stage workflow, configuration controls, optimization selection, evaluation and prediction views'],
    ['frontend/src/index.css', 'Warm-neutral design tokens, contrast mappings, focus states, reduced-motion behavior'],
])

for para in doc.paragraphs:
    para.paragraph_format.widow_control = True
props = doc.core_properties
props.title = 'No Code Machine Learning Platform Research Paper'
props.subject = 'Repository-grounded project research documentation'
props.author = 'OpenAI Codex'
props.keywords = 'no-code ML, AutoML, reproducibility, research context'
doc.save(OUT)
print(OUT.resolve())

