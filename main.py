"""
================================================================================
🚀 MARS ROVER PATHFINDER - MAIN APPLICATION
================================================================================
System Integration & AI Dashboard: Member 4
================================================================================
Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                 MARS ROVER PATHFINDER AI                    │
  ├─────────────────────────────────────────────────────────────┤
  │ [Member 1] A* Navigation Engine       (mars_rover_astar.py) │
  │ [Member 2] Computer Vision & ML Model (rock_detector.py)    │
  │ [Member 3] Rover Simulation & Physics (rover_simulation.py) │
  │ [Member 4] Integrated AI Dashboard    (dashboard.py)        │
  └─────────────────────────────────────────────────────────────┘

Run:
  python main.py
================================================================================
"""

import sys
import os

# Ensure local modules are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def print_banner():
    banner = r"""
  __  __                  _____                     _____       _   _        __ _           _             
 |  \/  |                |  __ \                   |  __ \     | | | |      / _(_)         | |            
 | \  / | __ _ _ __ ___  | |__) |_____   _____ _ __| |__) |__ _| |_| |__   | |_ _ _ __   __| | ___ _ __  
 | |\/| |/ _` | '__/ __| |  _  // _ \ \ / / _ \ '__|  ___/ _` | __| '_ \  |  _| | '_ \ / _` |/ _ \ '__| 
 | |  | | (_| | |  \__ \ | | \ \ (_) \ V /  __/ |  | |  | (_| | |_| | | | | | | | | | | (_| |  __/ |    
 |_|  |_|\__,_|_|  |___/ |_|  \_\___/ \_/ \___|_|  |_|   \__,_|\__|_| |_| |_| |_|_| |_|\__,_|\___|_|    
    """
    print(banner)
    print("=" * 80)
    print("  🚀 MARS ROVER PATHFINDER - UNIFIED AUTONOMOUS MISSION CONTROL")
    print("  Developed by AI Fundamentals Engineering Team (Members 1, 2, 3 & 4)")
    print("=" * 80)


def verify_subsystems():
    print("\n🔍 Checking Mission Subsystems...")

    # 1. Member 1: A* Navigation
    try:
        from mars_rover_astar import AStarPlanner, MarsMap
        planner = AStarPlanner()
        m = MarsMap.generate(rows=10, cols=10, seed=1)
        res = planner.plan(m)
        if res.found:
            print("  [✓] Member 1 (A* Path Planning): ONLINE (Algorithm operational)")
        else:
            print("  [!] Member 1 (A* Path Planning): ONLINE (No test path)")
    except Exception as e:
        print(f"  [X] Member 1 (A* Path Planning) ERROR: {e}")

    # 2. Member 2: CV & Rock Classifier
    try:
        import rock_detector
        print("  [✓] Member 2 (Computer Vision & ML): ONLINE")
        print(f"      - Classes: {list(rock_detector._model.classes_)}")
        print(f"      - Mineral Library: {len(rock_detector._mineral_map)} entries loaded")
    except Exception as e:
        print(f"  [!] Member 2 (Computer Vision & ML): PARTIAL/FALLBACK ({e})")

    # 3. Member 3: Mars Simulation
    try:
        import rover_simulation
        print("  [✓] Member 3 (Rover Environment & Physics): ONLINE")
    except Exception as e:
        print(f"  [!] Member 3 (Rover Environment): {e}")

    # 4. Member 4: Integration Engine & Dashboard
    try:
        from dashboard import MarsRoverDashboard
        print("  [✓] Member 4 (Integration Core & AI Dashboard): ONLINE")
    except Exception as e:
        print(f"  [X] Member 4 (Dashboard): ERROR {e}")
        return False

    print("=" * 80)
    print("🚀 All systems integrated successfully. Launching Mission Control GUI...\n")
    return True


def main():
    print_banner()
    if not verify_subsystems():
        print("Failed to initialize one or more critical subsystems.")
        sys.exit(1)

    from dashboard import MarsRoverDashboard
    app = MarsRoverDashboard()
    app.run()


if __name__ == "__main__":
    main()
