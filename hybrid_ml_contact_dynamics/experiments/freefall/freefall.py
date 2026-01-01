from hybrid_ml_contact_dynamics.physics.types import vec2
from hybrid_ml_contact_dynamics.physics.rigidbody import RigidBody2D
from hybrid_ml_contact_dynamics.physics.primitives.circle import Circle
from hybrid_ml_contact_dynamics.physics.primitives.plane import Plane 
from hybrid_ml_contact_dynamics.physics.collision.collisioncheck import CollisionCheck, ContactData, CollisionResolver
from hybrid_ml_contact_dynamics.physics.sim import Simulation
from hybrid_ml_contact_dynamics.experiments.freefall.trajectory_buffer import Trajectory_Buffer

should_update = True

gravity = vec2(0.0, -9.81)

def main():
    traj = Trajectory_Buffer()
    collision_check = CollisionCheck()
    circle = Circle(vec2(100.0, 10000.0), 5.0)
    circle.set_mass(5.0)
    circle.set_velocity(vec2(0.0, 0.0))
    plane = Plane(vec2(0.0, 1.0), 0.0)
    world = Simulation()
    world.add_rigidbody(circle.get_rigidbody())
    total_time = 0
    contact = ContactData(None, None, None)
    collision_resolver = CollisionResolver()
    dt = 1.0/60.0
    while(should_update):
        total_time = total_time + dt

        circle.add_force(circle.get_mass() * gravity)
        world.step(dt)

        contact = collision_check.circle_and_plane(circle, plane)
        if (contact.collided):
            collision_resolver.resolve_collision_circle_plane(circle, plane, contact)
            print("Collided!")
            print(circle.get_position())
            print(circle.get_acceleration())
            print(circle.get_velocity())
        else:
            print(circle.get_position())
            print(circle.get_acceleration())
            print(circle.get_velocity())
            print("Not colliding")

        if (total_time >= 300.0):
            print("Closing")
            break

        traj.record(total_time, circle.get_position(), circle.get_velocity())

    traj.finalise()
    traj.save()