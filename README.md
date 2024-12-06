# Circuit Design Simulation Project

This project simulates the Piezoelectric Bipolar Junction Transistor (PBJT) response to sound inputs and trains an AI model to classify voices as male or female.

## Setup:
1. Clone the repository.
2. Create a virtual environment and install dependencies:

python -m venv .venv

### Linux/macOS  
source .venv/bin/activate 
### Windows
.venv\Scripts\activate

3. Install requirements

pip install -r requirements.txt
-----------------------------------------------
## if it doesnt work. install all manually(one by one): 

pip install numpy pandas scipy

pip install scikit-learn

pip install matplotlib
------------------------------------------------
## To run the Simulation and AI:
1. Create Training Set: 
python scripts/pbjt_simulation.py

2. Load Training into Ai Model: 
python scripts/voice_ai_model.py 

3. Test the Simulator: 
python scripts/pbjt_ai_testing.py


