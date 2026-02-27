import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
client = None
if api_key:
    client = genai.Client(api_key=api_key)
    app.logger.info("Gemini API key found; attempting to use online mode (API must be enabled in Google Cloud).")
else:
    app.logger.warning("No GEMINI_API_KEY provided; using offline fallback responses.")

# Global store for the current dataset to simulate statefulness 
# (in a real app, you'd send data back and forth or use sessions, but this is simpler for the hackathon)
current_data = {'X': None, 'y': None}

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Simple mock authentication for hackathon
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials. Try admin / password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/quiz')
def quiz():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('quiz_page.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    algorithm_context = data.get('algorithm', 'Machine Learning')

    msg = user_message.lower()

    # If the Gemini client isn't configured, provide a simple local fallback answer
    if not client:
        # simplified local fallback that always replies with something useful
        if "learning rate" in msg:
            return jsonify({'response': "Learning rate is the step size during optimization; keep it small enough to converge and large enough to learn."})
        if "neural network" in msg or "nn" in msg:
            return jsonify({'response': "A neural network is layers of neurons that adjust weights via gradient descent."})
        if "brief" in msg or "explain" in msg:
            return jsonify({'response': "This sandbox trains simple models and visualizes their decision boundaries; adjust hyperparameters and watch the curve."})
        # catch‑all generic reply
        return jsonify({'response': "Ask me about model training, learning rate, algorithms, or graph explanations."})
    # Determine some basic dataset info to send to the AI
    dataset_info = "unknown"
    num_samples = 0
    if current_data['X'] is not None:
        num_samples = len(current_data['X'])
        # just a generic descriptor for the prompt
        dataset_info = f"a 2D dataset with {num_samples} data points"

    prompt = (
        f"You are an expert Machine Learning engineer and AI researcher. "
        f"The user is using an interactive sandbox exploring the '{algorithm_context}' algorithm on {dataset_info}. "
        f"They are looking at a visualization of the decision boundary graph. "
        f"User query: '{user_message}'.\n\n"
        f"Provide a highly structured, precise, and professional explanation formatted in HTML. "
        f"Use <b> tags for emphasis and <br> for line breaks. Make sure to include the following sections:\n"
        f"1. <b>Explanation</b>: Brief intuition behind the decision boundary.\n"
        f"2. <b>Formula/Math</b>: The mathematical intuition behind the boundary (use plain text for math, NO LaTeX).\n"
        f"3. <b>Parameters/Features</b>: Key parameters affecting this graph (like weights, k-value, learning rate).\n"
        f"IMPORTANT: You MUST write your entire response using ONLY basic HTML and plain English text. Do NOT use any LaTeX math notation, symbols, or equations (e.g. no $\mathbf{{x}}$, \mathbb{{R}}, etc). Only write standard English words."
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return jsonify({'response': response.text})
    except Exception as e:
        # If API is disabled, expired, or other error, fall back to offline responses
        msg = user_message.lower()
        print(f"Gemini API Error: {e}. Falling back to offline mode.")
        if "learning rate" in msg:
            return jsonify({'response': "Learning rate is the step size during optimization; keep it small enough to converge and large enough to learn."})
        if "neural network" in msg or "nn" in msg:
            return jsonify({'response': "A neural network is layers of neurons that adjust weights via gradient descent."})
        if "graph" in msg or "decision" in msg or "boundary" in msg:
            return jsonify({'response': "This sandbox trains simple models and visualizes their decision boundaries; adjust hyperparameters and watch the curve."})
        if "brief" in msg or "explain" in msg:
            return jsonify({'response': "This sandbox trains simple models and visualizes their decision boundaries; adjust hyperparameters and watch the curve."})
        return jsonify({'response': "Ask me about model training, learning rate, algorithms, or graph explanations."})

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    dataset_type = request.args.get('type', 'moons')
    n_samples = int(request.args.get('n_samples', 200))
    noise = float(request.args.get('noise', 0.2))
    
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:
        # Linear separable classification
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=42, n_clusters_per_class=1)
        # add noise manually for classification
        X += np.random.normal(0, noise, X.shape)
        
    current_data['X'] = X
    current_data['y'] = y
    
    return jsonify({
        'X': X.tolist(),
        'y': y.tolist()
    })


# new upload route for CSV datasets
@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    # accept a file sent via multipart/form-data
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        # expect comma separated values with x,y,label per row
        arr = np.loadtxt(file, delimiter=',')
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            return jsonify({'error': 'CSV must have at least 3 columns (x,y,label)'}), 400
        X = arr[:, :2]
        y = arr[:, 2].astype(int)
        current_data['X'] = X
        current_data['y'] = y
        return jsonify({'X': X.tolist(), 'y': y.tolist()})
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400


@app.route('/api/add_point', methods=['POST'])
def add_point():
    data = request.json
    x_val = float(data.get('x', 0))
    y_val = float(data.get('y', 0))
    cls = int(data.get('cls', 0))
    
    if current_data['X'] is not None and current_data['y'] is not None:
        current_data['X'] = np.vstack((current_data['X'], [x_val, y_val]))
        current_data['y'] = np.append(current_data['y'], cls)
    else:
        current_data['X'] = np.array([[x_val, y_val]])
        current_data['y'] = np.array([cls])
        
    return jsonify({
        'X': current_data['X'].tolist(),
        'y': current_data['y'].tolist()
    })

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    algorithm = data.get('algorithm', 'nn')
    epochs = int(data.get('epochs', 1))
    learning_rate = float(data.get('learningRate', 0.01))
    
    X = current_data['X']
    y = current_data['y']
    
    if X is None or y is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    # Initialize model based on selected algorithm
    if algorithm == 'lr':
        # Scikit-learn doesn't support partial_fit for standard LogisticRegression very well, 
        # so we'll use MLPClassifier with no hidden layers and identity activation for a linear baseline, 
        # or SGDClassifier (which is basically logistic regression if loss='log_loss')
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=learning_rate, max_iter=epochs, random_state=42)
        model.fit(X, y) # For SGD we can just fit X times
        
    elif algorithm == 'knn':
        # KNN doesn't "train" iteratively in the traditional sense, so we just return the boundary
        n_neighbors = int(data.get('k_neighbors', 5))
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X, y)
        
    elif algorithm == 'nn':
        # Use a simple MLP. We can't really control the exact internal epoch step easily in sklearn 
        # without partial_fit, so we approximate by retraining with varying max_iter to simulate progress.
        model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=epochs, 
                              learning_rate_init=learning_rate, solver='adam', random_state=42, 
                              warm_start=True)
        # Note: scikit-learn will raise warnings if it doesn't converge early on, which is fine
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X, y)
    else:
        return jsonify({'error': 'Unknown algorithm'}), 400

    # Create grid to evaluate model and generate decision boundary
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .05  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Calculate simple loss metric (1 - accuracy for simplicity)
    accuracy = model.score(X, y)
    loss = 1.0 - accuracy

    return jsonify({
        'grid_x': xx.tolist(),
        'grid_y': yy.tolist(),
        'grid_z': Z.tolist(),
        'loss': loss,
        'accuracy': accuracy
    })

if __name__ == '__main__':
    # Use debug mode only if FLASK_ENV=development is set
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, port=5000)
