from physics.types import vec2
from physics.rigidbody import RigidBody2D
from physics.primitives.circle import Circle 
from physics.sim import Simulation

should_update = True

def main():
    circle = Circle(vec2(100.0, 1000.0), 5.0)
    circle.set_mass(5.0)
    circle.set_velocity(vec2(0.0, 0.0))
    circle.add_force(vec2(0.0, -9.81))
    world = Simulation()
    world.add_rigidbody(circle.get_rigidbody())

    while(should_update):
        world.step(1.0/60.0)
        if circle.get_position()[1] <= 0.0:
            break
    

if __name__ == "__main__":
    main()