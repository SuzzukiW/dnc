import traci
import os
import sys

# Add SUMO tools to path
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# Start SUMO
traci.start(['sumo', '-n', 'Version1/2024-11-05-18-42-37/osm.net.xml', 
             '-r', 'Version1/2024-11-05-18-42-37/osm.passenger.trips.xml'])

# Get traffic lights
tls = traci.trafficlight.getIDList()

# Print info for first traffic light
for tl in tls[:1]:
    print(f'Traffic Light {tl}:')
    programs = traci.trafficlight.getAllProgramLogics(tl)
    for p in programs:
        print(f'Program: {p.programID}, phases: {len(p.phases)}')
        for i, phase in enumerate(p.phases):
            print(f'Phase {i}: state={phase.state}, duration={phase.duration}')

traci.close()
