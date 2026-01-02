# Hybrid ML Contact Dynamics

## Motivation and Overview

This project is a proof of concept research investigation into how machine learning (ML) models can support and accelerate physics calculations in complex and computationally demanding real-time simulations.

While the ML models are not intended to replace numerical computations entirely, the models are intended to improve computational performance for simulating physics phenomena which are notoriously difficult to implement in real-time due to noise, stiffness and computational constraints.

The current achieved milestone investigates whether ML can improve the estimation of contact restitution under noisy, discretely sampled dynamics.

This project has produced an encouraging baseline from which further ML and physics systems can be simulated and evaluated. While modern analytic estimators degrade rapidly under noise and temporal uncertainty, the model created has proven ML systems can successfully exploit local temporal structure around contact events to recover physically meaningful parameters.

Building upon this foundation, the developed baseline will be extended to study hybrid ML potential for more challenging physical systems such as fluid dynamics simulations.

## Physical System and Ground Truth

The physical system of this milestone simulates 2D rigidbody collision. Specifically, multiple configurable runs are performed involving a 2D circle rigidbody under the force of gravity freefalling and colliding with an infinite plane.

Each run has a fixed coefficient of restitution $e \in (0, 1)$. The governing equations of the simulation involve physically accurate freefall equations, the impact law and realistic height decay used as ground truth values for evaluation of the ML model. These equations are as follows:

### Dynamics Equations

$g = -9.81$

$\dot{v} = g$

$\dot{y} = v$

### Impact Law

$v^+ = -ev^-$

### Height Decay

$\frac{h_k+1}{h_k} = e^2$

### Visual Plots

Figure 1: Position vs Time

Figure 2: Velocity vs Time

Figure 3: Phase Plot (Velocity vs Position)

## Data Generation and Trajectory Logging

## Noise Model and Temporal Windowing

## Baseline Estimators

## Machine Learning Model

## Training and Evaluation

## Results

## Discussion

## Next Steps

## How to Run
