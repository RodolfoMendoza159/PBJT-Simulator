# Circuit Design Simulation Project

This project simulates the Piezoelectric Bipolar Junction Transistor (PBJT) response to sound inputs and trains an AI model to classify voices as male or female.

## Setup:
1. Clone the repository.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

### Linux/macOS 
```bash 
source .venv/bin/activate 
```
### Windows
```bash
.venv\Scripts\activate
```

3. Install requirements

```bash
pip install -r requirements.txt
```

----
## if it doesnt work then install all manually(one by one): 
```bash
pip install numpy
```

```bash
pip install pandas
```

```bash
pip install scipy
```

```bash
pip install scikit-learn
```

```bash
pip install matplotlib
```

---
## To run the Simulation:
On Terminal: 

```bash
python scripts/pbjt_simulation.py
```

### Options Select Menu: 
1. Generate DataSet: Begin with this always for AI usage, creates a dataset for training.
2. Dynamic Range Simulator: Shows Dynamic Change graph.
3. Sensitivity Calculation: Shows the sensitivity calculation.
4. Signal Transformation Test: Generates sample test and graphs. 

## Loads Training into Ai Model: 
On Terminal:

```bash
python scripts/voice_ai_model.py 
```

## Test the Simulator: 
On Terminal:

```bash
python scripts/pbjt_ai_testing.py
```

