# Customer Churn Prediction (ANN)

A Streamlit app that predicts whether a bank customer is likely to churn, using an Artificial Neural Network trained on the classic `Churn_Modelling` dataset.

## How it works
- Preprocesses inputs (credit score, geography, gender, age, tenure, balance, etc.) with saved `scikit-learn` encoders/scaler
- - Feeds the processed features into a trained Keras ANN (`res/annchurn.keras`)
  - - Returns a churn probability in a simple web UI
   
    - ## Tech stack
    - Python · TensorFlow/Keras · scikit-learn · Streamlit · pandas
   
    - ## Run locally
    - ```bash
      pip install -r requirements.txt
      streamlit run app.py
      ```

      ## Files
      - `processing_training.ipynb` — data preprocessing and model training
      - - `parameter_tuning.ipynb` — hyperparameter tuning experiments
        - - `app.py` — Streamlit inference app
          - - `res/` — saved model, scaler, and encoders
            - 
