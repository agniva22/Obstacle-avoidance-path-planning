# Obstacle-avoidance-path-planning

## Directory Structure

```
├── MPC_OUT.py              # For nomral MPC based path tracking 
├── PPO_TRAIN.py            # To train the PPO algorithm on same environment and vehicle dynamics
├── PPO_TEST.py             # For testing PPO algorithm
├── MPC_FUSION.py           # Run first to execute the fusion algorithm
├── PPO_FUSION.py           # Then run it, to pair with socket connection, to exchange the data and execute the fusion algorithm 
├── Dynamics.py             # Here vehicle dynamics is defined
├── ENV.py                  # Here designed environment is defined (not necessary for running any python file)
├── .pth                    # Contain trained model weights

```

