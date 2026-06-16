# BCI-Controlled Intelligent Wheelchair System

MSc dissertation project: an EEG-based Brain-Computer Interface that classifies
motor-imagery signals and maps them to control commands for a simulated
intelligent wheelchair.

Dataset: BCI Competition IV Dataset 2a (9 subjects, 22 EEG channels, 4 classes).

## Class -> wheelchair command mapping

| EEG motor imagery | Wheelchair command |
|--------------------|---------------------|
| Left hand           | Turn left           |
| Right hand          | Turn right          |
| Feet                | Move forward        |
| Tongue              | Stop                |

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Run

\`\`\`bash
python scripts/quick_check.py
\`\`\`
