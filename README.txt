RUN THE FRONTEND LOCALLY
1. Install Python.
2. Open Command Prompt / Terminal in this folder.
3. Run: python -m pip install -r requirements.txt
4. Run: python -m streamlit run app.py
5. Open http://localhost:8501 if the browser does not open.

CONNECT YOUR COLAB BACKEND
- Export your Colab notebook as a .py file or copy its reusable functions.
- Put those functions in backend.py.
- Implement run_model(dataframe, horizon, n_paths) as shown in backend_template.py.
- Keep app.py and backend.py in the same folder.

WHERE TO RUN?
Recommended: run Streamlit on your computer using VS Code or Command Prompt.
Google Colab is suitable for developing/testing the Python model, but it is not the simplest place to host a persistent interactive Streamlit app. Later you can deploy the app using Streamlit Community Cloud.
