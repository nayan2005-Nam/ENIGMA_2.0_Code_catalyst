// Global State Variables
let currentDataset = 'moons';
let trainingInterval = null;
let currentLosses = [];
let currentEpochSum = 0;
let isTraining = false;
let currentPaintClass = 0;

// DOM Elements
const algorithmSelect = document.getElementById('algorithmSelect');
const lrSlider = document.getElementById('lrSlider');
const lrVal = document.getElementById('lrVal');
const epochSlider = document.getElementById('epochSlider');
const epochVal = document.getElementById('epochVal');
const lossDisplay = document.getElementById('lossDisplay');
const trainBtn = document.getElementById('trainBtn');
const resetBtn = document.getElementById('resetBtn');
const datasetBtns = document.querySelectorAll('.dataset-btn');
const trainingIndicator = document.getElementById('trainingIndicator');
const scenarioBtn = document.getElementById('scenarioBtn');
const scenarioCard = document.getElementById('scenarioCard');
const paintRedBtn = document.getElementById('paintRedBtn');
const paintBlueBtn = document.getElementById('paintBlueBtn');

// new controls for import and regenerate
const importBtn = document.getElementById('importBtn');
const generateBtn = document.getElementById('generateBtn');
const fileInput = document.getElementById('fileInput');

// New Phase 2 DOM Elements
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatHistory = document.getElementById('chatHistory');
// update placeholder text to emphasise technical questions
chatInput.placeholder = "Try: learning rate, neural networks, training steps, graph explanation";
const notesContent = document.getElementById('notesContent');
const quizQuestion = document.getElementById('quizQuestion');
const quizOptions = document.getElementById('quizOptions');
const quizFeedback = document.getElementById('quizFeedback');

// --- Curriculum Content ---
const curriculum = {
    'nn': {
        title: 'Neural Network (MLP)',
        notes: `
            <p class="mb-2">A Neural Network mathematically simulates the human brain using connected "neurons".</p>
            <p class="mb-2">It excels at finding <b>non-linear boundaries</b>. Because it uses hidden layers and activation functions, it can bend its decision boundary around complex shapes like circles and moons.</p>
        `,
        moreNotes: `
            <p class="mb-2">Internally it consists of layers of weighted sums and activation functions; training adjusts weights using gradient descent and backpropagation.</p>
            <p class="mb-2">Modern nets often use dropout, batch normalization, and various optimizers, but the core idea is still the same.</p>
        `,
        docUrl: 'https://en.wikipedia.org/wiki/Artificial_neural_network',
        quiz: {
            q: "Does a Neural Network use partial derivatives to update its weights?",
            options: [
                { text: "Yes, it's called Gradient Descent.", correct: true },
                { text: "No, it just remembers points.", correct: false }
            ]
        }
    },
    'lr': {
        title: 'Linear Classifier (SGD)',
        notes: `
            <p class="mb-2">A Linear Classifier tries to find a single straight line (or hyperplane) to divide the classes.</p>
            <p class="mb-2">It's fast and interpretable, but fails entirely on datasets like "Circles" where a straight line cannot separate the inside and outside.</p>
        `,
        moreNotes: `
            <p class="mb-2">SGD stands for Stochastic Gradient Descent: the model updates weights incrementally using one datapoint at a time, which allows it to scale to large datasets.</p>
            <p class="mb-2">With appropriate regularization you can control overfitting; logistic regression is a common example of a linear classifier.</p>
        `,
        docUrl: 'https://en.wikipedia.org/wiki/Linear_classifier#Logistic_regression',
        quiz: {
            q: "Can a Linear Classifier perfectly solve the 'Moons' dataset?",
            options: [
                { text: "Yes, easily.", correct: false },
                { text: "No, it can only draw straight lines.", correct: true }
            ]
        }
    },
    'knn': {
        title: 'K-Nearest Neighbors',
        notes: `
            <p class="mb-2">KNN doesn't actually "train" iteratively. It just memorizes the data.</p>
            <p class="mb-2">To make a prediction, it looks at the <b>K nearest points</b> to that spot and takes a majority vote. It can draw very jagged, complex borders.</p>
        `,
        moreNotes: `
            <p class="mb-2">The value of K controls smoothness; small K leads to overfitting, large K behaves more like a global average.</p>
            <p class="mb-2">Distance metrics (Euclidean, Manhattan, etc.) determine neighborhood shape; standardization of features is important.</p>
        `,
        docUrl: 'https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm',
        quiz: {
            q: "If K=1, what does the decision boundary look like?",
            options: [
                { text: "A perfectly smooth line.", correct: false },
                { text: "Extremely jagged, drawing boxes around every single point.", correct: true }
            ]
        }
    }
};

// Base Plotly config for dark mode
const plotConfig = {
    responsive: true,
    displayModeBar: false
};
const plotLayoutBase = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8' },
    margin: { t: 20, r: 20, l: 40, b: 40 }
};

// --- Guided Scenarios ---
const scenarios = [
    {
        title: "Level 1: The Basics (Line of Best Fit)",
        desc: "Linear algorithms (like SGD/Logistic Regression) can easily separate simple data. Try the 'Linear' dataset and hit Train!",
        setup: () => { setDataset('linear'); algorithmSelect.value = 'lr'; }
    },
    {
        title: "Level 2: The Noise (Underfitting)",
        desc: "A linear model fails on circular data. Select 'Circles' and try to train Linear. Notice how it cannot curve. Then, switch to KNN!",
        setup: () => { setDataset('circles'); algorithmSelect.value = 'lr'; }
    },
    {
        title: "Level 3: Deep Learning (Neural Networks)",
        desc: "Neural Nets excel at complex, non-linear boundaries. Select 'Moons', choose Neural Network, turn down the Learning Rate slightly, and watch it mold!",
        setup: () => { setDataset('moons'); algorithmSelect.value = 'nn'; lrSlider.value = 0.05; updateSliderValues(); }
    }
];
let currentScenarioIdx = -1;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initPlots();
    fetchDataset(currentDataset);
    setupEventListeners();
});

function setupEventListeners() {
    // Sliders
    lrSlider.addEventListener('input', updateSliderValues);
    epochSlider.addEventListener('input', updateSliderValues);

    // Algorithm Change
    algorithmSelect.addEventListener('change', updateCurriculum);

    // Buttons
    datasetBtns.forEach(btn => {
        btn.addEventListener('click', (e) => setDataset(e.target.dataset.type));
    });

    trainBtn.addEventListener('click', toggleTraining);
    resetBtn.addEventListener('click', () => {
        stopTraining();
        fetchDataset(currentDataset);
    });

    scenarioBtn.addEventListener('click', nextScenario);

    // Paintbrush toggle
    paintRedBtn.addEventListener('click', () => setPaintClass(0));
    paintBlueBtn.addEventListener('click', () => setPaintClass(1));

    // Additional dataset controls
    importBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
    generateBtn.addEventListener('click', () => {
        // re-fetch current dataset to regenerate random sample
        fetchDataset(currentDataset);
    });

    // Plotly canvas click event for adding points
    document.getElementById('mainPlot').addEventListener('click', handlePlotClick);

    // Chatbot Submit
    chatForm.addEventListener('submit', handleChatSubmit);

    // Explain Graph Button
    const explainGraphBtn = document.getElementById('explainGraphBtn');
    if (explainGraphBtn) {
        explainGraphBtn.addEventListener('click', () => {
            chatInput.value = "Please explain the current graph and decision boundary.";
            chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        });
    }

    // Init Curriculum
    updateCurriculum();
}

// Phase 2 Logic
function updateCurriculum() {
    const algo = algorithmSelect.value;
    const content = curriculum[algo];

    if (!content) return;

    // Update Notes with toggle and documentation link
    notesContent.innerHTML = `
        <div class="flex items-center justify-between">
            <h3 class="font-bold text-white mb-2">${content.title}</h3>
            ${content.docUrl ? `<a href="${content.docUrl}" target="_blank" title="Open documentation" class="text-indigo-300 hover:text-indigo-100 text-sm">📖</a>` : ''}
        </div>
        ${content.notes}
        <div id="moreNotes" class="hidden">
            ${content.moreNotes || ''}
        </div>
        ${content.moreNotes ? '<button id="toggleNotesBtn" class="text-xs text-indigo-400 mt-1">Show more</button>' : ''}
    `;
    // attach toggle handler if applicable
    if (content.moreNotes) {
        const btn = document.getElementById('toggleNotesBtn');
        const more = document.getElementById('moreNotes');
        btn.addEventListener('click', () => {
            if (more.classList.contains('hidden')) {
                more.classList.remove('hidden');
                btn.textContent = 'Show less';
            } else {
                more.classList.add('hidden');
                btn.textContent = 'Show more';
            }
        });
    }

    // Update Quiz
    quizQuestion.textContent = content.quiz.q;
    quizOptions.innerHTML = '';
    quizFeedback.classList.add('hidden');

    content.quiz.options.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = "w-full text-left bg-slate-900 border border-slate-700 hover:bg-slate-800 p-2 rounded text-sm transition text-slate-300";
        btn.textContent = opt.text;
        btn.onclick = () => checkAnswer(opt.correct, btn);
        quizOptions.appendChild(btn);
    });
}

// Make sure global checkAnswer is available
window.checkAnswer = function (isCorrect, btn) {
    const allBtns = quizOptions.querySelectorAll('button');
    allBtns.forEach(b => {
        b.disabled = true;
        b.classList.remove('hover:bg-slate-800');
    });

    if (isCorrect) {
        btn.classList.replace('bg-slate-900', 'bg-emerald-500/20');
        btn.classList.add('border-emerald-500', 'text-emerald-400');
        quizFeedback.textContent = "✅ Correct!";
        quizFeedback.className = "mt-3 text-xs font-bold text-emerald-400 block";
    } else {
        btn.classList.replace('bg-slate-900', 'bg-red-500/20');
        btn.classList.add('border-red-500', 'text-red-400');
        quizFeedback.textContent = "❌ Incorrect. Keep trying!";
        quizFeedback.className = "mt-3 text-xs font-bold text-red-400 block";
    }
};

async function handleChatSubmit(e) {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    // 1. Append user message
    appendMessage('user', message);
    chatInput.value = '';

    // 2. Append loading buble
    const loadingId = 'loading-' + Date.now();
    appendMessage('ai', 'Thinking...', loadingId);

    // 3. Fetch from Python /api/chat
    const algo = document.getElementById('algorithmSelect').options[document.getElementById('algorithmSelect').selectedIndex].text;
    // provide algorithm context for chat prompt

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, algorithm: algo })
        });
        const data = await res.json();

        // 4. Update loading bubble with real message
        const loadingBox = document.getElementById(loadingId);
        if (loadingBox) loadingBox.textContent = data.response;

    } catch (err) {
        const loadingBox = document.getElementById(loadingId);
        if (loadingBox) loadingBox.textContent = "Error connecting to AI Tutor.";
    }
}

function appendMessage(sender, text, id = null) {
    const div = document.createElement('div');
    if (id) div.id = id;

    if (sender === 'user') {
        div.className = "bg-indigo-600 rounded-lg rounded-tr-none p-3 text-sm text-white border border-indigo-500/50 self-end w-5/6";
    } else {
        div.className = "bg-slate-800/80 rounded-lg rounded-tl-none p-3 text-sm text-slate-300 border border-slate-700 self-start w-5/6";
    }
    div.textContent = text;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function setPaintClass(cls) {
    currentPaintClass = cls;
    if (cls === 0) {
        paintRedBtn.className = "paint-btn flex-1 bg-red-500/20 text-red-400 border-2 border-red-500 rounded p-2 text-xs transition font-semibold";
        paintBlueBtn.className = "paint-btn flex-1 bg-slate-800 text-slate-400 border border-slate-600 hover:border-blue-500 rounded p-2 text-xs transition";
    } else {
        paintRedBtn.className = "paint-btn flex-1 bg-slate-800 text-slate-400 border border-slate-600 hover:border-red-500 rounded p-2 text-xs transition";
        paintBlueBtn.className = "paint-btn flex-1 bg-blue-500/20 text-blue-400 border-2 border-blue-500 rounded p-2 text-xs transition font-semibold";
    }
}

async function handlePlotClick(e) {
    const mainPlot = document.getElementById('mainPlot');
    // Ensure we are clicking inside the actual plot area (the .xy layer)
    const bg = mainPlot.querySelector('.xy');
    if (!bg) return;

    const rect = bg.getBoundingClientRect();

    // Check if click is within the drawing area bounds
    if (e.clientX >= rect.left && e.clientX <= rect.right &&
        e.clientY >= rect.top && e.clientY <= rect.bottom) {

        // Convert screen pixels to chart coordinates using Plotly's internal layout object
        const xaxis = mainPlot._fullLayout.xaxis;
        const yaxis = mainPlot._fullLayout.yaxis;

        const xData = xaxis.p2c(e.clientX - rect.left);
        const yData = yaxis.p2c(e.clientY - rect.top);

        await addDataPoint(xData, yData, currentPaintClass);
    }
}

async function addDataPoint(x, y, cls) {
    try {
        const res = await fetch('/api/add_point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x, y, cls })
        });
        const data = await res.json();

        // Re-render the points
        renderScatter(data.X, data.y);

        // If not actively training, trigger a single update so boundary adapts instantly
        if (!isTraining) {
            performTrainingStep();
        }
    } catch (err) {
        console.error("Failed to add point", err);
    }
}

// handle CSV file uploads from user device
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: form
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        // reset state and render
        currentLosses = [];
        currentEpochSum = 0;
        updateLossPlot();
        lossDisplay.textContent = "1.000";
        renderScatter(data.X, data.y);
    } catch (err) {
        console.error('Upload failed', err);
        alert('Failed to import dataset: ' + err.message);
    } finally {
        // clear file input so same file can be re-selected later
        fileInput.value = '';
    }
}

function updateSliderValues() {
    lrVal.textContent = parseFloat(lrSlider.value).toFixed(3);
    epochVal.textContent = epochSlider.value;
}

function setDataset(type) {
    if (isTraining) stopTraining();
    currentDataset = type;

    // Update active button styles
    datasetBtns.forEach(btn => {
        if (btn.dataset.type === type) {
            btn.classList.replace('bg-slate-800', 'bg-indigo-600');
            btn.classList.replace('border-slate-600', 'border-transparent');
        } else {
            btn.classList.replace('bg-indigo-600', 'bg-slate-800');
            btn.classList.replace('border-transparent', 'border-slate-600');
        }
    });

    fetchDataset(type);
}

function initPlots() {
    // Init Main Plot (Empty)
    Plotly.newPlot('mainPlot', [], {
        ...plotLayoutBase,
        xaxis: { showgrid: false, zeroline: false },
        yaxis: { showgrid: false, zeroline: false }
    }, plotConfig);

    // Init Loss Plot (Empty)
    Plotly.newPlot('lossPlot', [{
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#34d399', width: 2 }
    }], {
        ...plotLayoutBase,
        yaxis: { title: 'Loss', range: [0, 1] },
        xaxis: { title: 'Iteration Steps' }
    }, plotConfig);
}

// Communication with Python Flask Backend
async function fetchDataset(type) {
    try {
        const res = await fetch(`/api/dataset?type=${type}`);
        const data = await res.json();

        // Reset state
        currentLosses = [];
        currentEpochSum = 0;
        updateLossPlot();
        lossDisplay.textContent = "1.000";

        // Render scatter
        renderScatter(data.X, data.y);
    } catch (err) {
        console.error("Failed to load dataset", err);
    }
}

function renderScatter(X, y) {
    // Separate classes for coloring
    const class0_X = [], class0_Y = [];
    const class1_X = [], class1_Y = [];

    for (let i = 0; i < X.length; i++) {
        if (y[i] === 0) {
            class0_X.push(X[i][0]);
            class0_Y.push(X[i][1]);
        } else {
            class1_X.push(X[i][0]);
            class1_Y.push(X[i][1]);
        }
    }

    const trace0 = {
        x: class0_X, y: class0_Y,
        mode: 'markers', type: 'scatter',
        name: 'Class 0', marker: { color: '#ef4444', size: 8, line: { color: '#7f1d1d', width: 1 } }
    };

    const trace1 = {
        x: class1_X, y: class1_Y,
        mode: 'markers', type: 'scatter',
        name: 'Class 1', marker: { color: '#3b82f6', size: 8, line: { color: '#1e3a8a', width: 1 } }
    };

    Plotly.react('mainPlot', [trace0, trace1], {
        ...plotLayoutBase,
        showlegend: false
    }, plotConfig);
}

function toggleTraining() {
    if (isTraining) {
        stopTraining();
    } else {
        startTraining();
    }
}

function startTraining() {
    isTraining = true;
    trainBtn.innerHTML = `
        <svg class="w-4 h-4 animate-spin hidden" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
        Pause
    `;
    trainBtn.classList.replace('from-emerald-500', 'from-amber-500');
    trainBtn.classList.replace('to-emerald-600', 'to-amber-600');

    trainingIndicator.classList.remove('hidden');
    // Removed statusText.textContent = "Training..."; as it's not in the provided diff

    // In a real TF.js setting we'd use requestAnimationFrame
    // Since we call Py backend, we'll use an interval so we don't spam the server too hard
    // but fast enough to look like animation (e.g. 10 times a second if server handles it)
    trainingInterval = setInterval(performTrainingStep, 250);
}

function stopTraining() {
    isTraining = false;
    trainBtn.innerHTML = `
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
        Train
    `;
    trainBtn.classList.replace('from-amber-500', 'from-emerald-500');
    trainBtn.classList.replace('to-amber-600', 'to-emerald-600');

    trainingIndicator.classList.add('hidden');
    // Removed statusText.textContent = "Idle"; as it's not in the provided diff

    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
}

async function performTrainingStep() {
    // Current hyperparams
    const algo = algorithmSelect.value;
    const lr = parseFloat(lrSlider.value);
    const itersPerStep = parseInt(epochSlider.value);

    // Increase total epochs sent to Py backend to simulate moving forward
    currentEpochSum += itersPerStep;

    const payload = {
        algorithm: algo,
        learningRate: lr,
        epochs: currentEpochSum, // we feed absolute total epochs if we use warm_start or standard fit loop mapping
        k_neighbors: 5 // hardcoded for demo
    };

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Backend error");

        const data = await res.json();

        // Update Loss
        lossDisplay.textContent = data.loss.toFixed(4);
        currentLosses.push(data.loss);
        updateLossPlot();

        // Update Accuracy display
        if (data.accuracy !== undefined) {
            document.getElementById('accDisplay').textContent = (data.accuracy * 100).toFixed(2) + '%';
        }

        // Update Decision Boundary (Contour)
        renderDecisionBoundary(data.grid_x, data.grid_y, data.grid_z);

        // If KNN, auto-stop (doesn't iterate)
        if (algo === 'knn') {
            stopTraining();
        }

    } catch (err) {
        console.error("Training step failed", err);
        stopTraining();
    }
}

function renderDecisionBoundary(xx, yy, Z) {
    const contour_trace = {
        x: xx[0],  // 1D array of x coords
        y: yy.map(row => row[0]), // 1D array of y coords
        z: Z, // 2D grid
        type: 'contour',
        showscale: false,
        colorscale: [
            [0, 'rgba(239, 68, 68, 0.4)'], // Tailwind red-500 transparent
            [0.5, 'rgba(255,255,255,0)'],
            [1, 'rgba(59, 130, 246, 0.4)']  // Tailwind blue-500 transparent
        ],
        opacity: 0.5,
        line: { width: 0 },
        hoverinfo: 'skip'
    };

    // Keep existing scatter data but update base plot
    const currentPlotInfo = document.getElementById('mainPlot').data;
    const scatterTraces = currentPlotInfo.filter(t => t.type === 'scatter');

    // We put contour first so markers sit on top
    Plotly.react('mainPlot', [contour_trace, ...scatterTraces], plotLayoutBase, plotConfig);
}

function updateLossPlot() {
    Plotly.react('lossPlot', [{
        y: currentLosses,
        x: Array.from({ length: currentLosses.length }, (_, i) => i * parseInt(epochSlider.value)),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#34d399', width: 3, shape: 'spline' },
        fill: 'tozeroy',
        fillcolor: 'rgba(52, 211, 153, 0.1)'
    }], {
        ...plotLayoutBase,
        yaxis: { title: 'Loss (1 - Acc)', range: [-0.05, 1.05] },
        xaxis: { title: 'Relative Iterations' }
    }, plotConfig);
}

function nextScenario() {
    currentScenarioIdx = (currentScenarioIdx + 1) % scenarios.length;
    const scenario = scenarios[currentScenarioIdx];

    document.getElementById('scenarioTitle').textContent = scenario.title;
    document.getElementById('scenarioDesc').textContent = scenario.desc;

    scenarioCard.classList.remove('hidden');
    scenarioCard.classList.add('animate-fade-in');

    scenario.setup();
}
