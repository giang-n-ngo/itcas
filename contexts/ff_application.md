# Application 1: Spacecraft Position Keeping
## Description
Spacecraft position keeping involves maintaining the relative position of a spacecraft along the orbit. 
The goal is to ensure that the spacecraft does not drift away too far from its intended path.

To control a spacecraft, we use an LQR controller that tries to minimize the deviation from the desired trajectory while also minimizing the control effort.
An LQR controller is defined by 10 hyperparameters, which are the weights for the state and control variables in the cost function, and control frequency.
The state of a spacecraft can be represented by its position and velocity in three-dimensional space, resulting in a 6-dimensional state vector.

## Performance Metrics
Given a specific LQR controller and an initial state of the spacecraft, we can simulate the trajectory of the spacecraft over a predetermined time.
The simulation will result in some performance metrics, including:
- **Root Mean Square Error (RMSE)**: This metric measures the average deviation of the spacecraft's position from the desired trajectory over the simulation time. A lower RMSE indicates better performance.
- **Fuel Consumption**: This metric measures the total amount of fuel used by the spacecraft to maintain its position. A lower fuel consumption indicates a more efficient controller.
- **Maximum Deviation**: This metric measures the maximum distance the spacecraft deviates from the desired trajectory at any point during the simulation. A lower maximum deviation indicates better performance.
- **Proportion of Time Within Threshold**: This metric measures the proportion of time during the simulation that the spacecraft's position is within a certain threshold distance from the desired trajectory. A higher proportion indicates better performance.

## Hyperparameter and State Ranges
The LQR hyperparameters, denoted as a vector x, range:
- State weights: from 1e-16 to 1e-1
- Control weights: from 1e-2 to 10
- Control frequency: from 1 second to 1000 seconds

The state vector, denoted as c, range:
- Radial position: from -100m to 100 m
- Along-track position: from -200m to 200m
- Cross-track position: from -200m to 200m
- Radial velocity: from -0.1m/s to 0.1m/s
- Along-track velocity: from -0.1 m/s to 0.1 m/s
- Cross-track velocity: from -0.1 m/s to 0.1 m/s

## Performance Constraints
To ensure that the spacecraft maintains a safe and efficient trajectory, we impose the following performance constraints:
- The RMSE must be less than 150 meters.
- The fuel consumption must be less than 100 grams.
- The maximum deviation must be less than 300 meters.
- The proportion of time within the threshold must be greater than 0.5.

## Objective
This application happens during the development phase of the LQR controller, where we aim to find controllers that satisfy the performance constraints for a wide range of initial states.
In addition, given a specific initial state, we also want controllers with diverse performance metrics in order to understand the trade-offs between different controllers and to select the most suitable one for a given mission.

# Application 2: Placeholder