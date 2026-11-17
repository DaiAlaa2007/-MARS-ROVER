MARS ROVER PATHFINDER - MEMBER 3 PACKAGE

FILES
- rover_simulation.py : Member 3 - Pygame Mars environment, rover movement,
  simulated camera, and dynamic obstacle interaction.
- mars_rover_astar.py : Member 1 - A* map, path planning, terrain cost,
  and obstacle re-planning.
- rock_detector.py : Member 2 - image feature extraction and rock classifier.

IMPORTANT
1. Keep these three Python files in the SAME folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python rover_simulation.py

CONTROLS
SPACE = pause/resume
O = place a new obstacle ahead and re-plan
C = capture a simulated camera image
R = restart
Click an empty cell on the map = add obstacle
ESC = quit

OPTIONAL ROCK AI MODEL
The rock_detector.py file expects rock_model.pkl in the same folder.
That model file was not included in the uploaded files. The camera simulation
still works without it, but AI rock classification will remain unavailable
until the trained model is supplied by Member 2.

NOTE
The rover simulation uses the A* module supplied in the project files.
Do not rename mars_rover_astar.py or rock_detector.py unless the import
statements in rover_simulation.py are updated too.
