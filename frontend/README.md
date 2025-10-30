App accesses Flask API at `http://localhost:5000`. React runs at `http://localhost:3000` (or `3001`).

## Data & Model

- **Dataset:** cleaned_data.csv
- **Selected Features:**

- "Number_of_casualties", "Cause_of_accident", "Day_of_week", "Type_of_vehicle",
  "Area_accident_occured", "Age_band_of_driver", "Driving_experience",
  "Number_of_vehicles_involved", "Time_Period", "Types_of_Junction",
  "Lanes_or_Medians", "Vehicle_movement","Light_conditions"

- **Model:** Random Forest Classifier
- **Performance:** Evaluated by accuracy, precision, recall, F1-score

## How to Run

**Backend:**
cd backend
pip install -r requirements.txt
python app.py

**Frontend:**
cd frontend
npm install
npm start
