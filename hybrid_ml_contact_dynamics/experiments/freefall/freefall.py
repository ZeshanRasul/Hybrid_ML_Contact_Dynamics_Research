import datetime
import json
import numpy as np

from hybrid_ml_contact_dynamics.physics.types import vec2
from hybrid_ml_contact_dynamics.physics.rigidbody import RigidBody2D
from hybrid_ml_contact_dynamics.physics.primitives.circle import Circle
from hybrid_ml_contact_dynamics.physics.primitives.plane import Plane 
from hybrid_ml_contact_dynamics.physics.collision.collisioncheck import CollisionCheck, ContactData, CollisionResolver
from hybrid_ml_contact_dynamics.physics.sim import Simulation
from hybrid_ml_contact_dynamics.experiments.freefall.trajectory_buffer import Trajectory_Buffer
from hybrid_ml_contact_dynamics.experiments.freefall.validation import validate_restitution

should_update = True

def main():
    config = list()
    with open("hybrid_ml_contact_dynamics/experiments/freefall/config.json") as f:
        config = json.load(f)
    collision_check = CollisionCheck()
    gravity_force = vec2(config['Force X'], config['Force Y'])
    gravity = gravity_force
    start_pos = vec2(config['Start Pos X'], config['Start Pos Y'])
    circle = Circle(start_pos, 5.0)
    circle.set_mass(5.0)
    circle.set_velocity(vec2(0.0, 0.0))
    plane = Plane(vec2(0.0, 1.0), 0.0)
    world = Simulation()
    world.add_rigidbody(circle.get_rigidbody())
    total_time = 0
    contact = ContactData(None, None, None)
    collision_resolver = CollisionResolver()
    dt = config['Delta Time']
    run_count = config['Runs']
    runtimestamp = datetime.datetime.now().strftime("%d-%m-%Y,%H:%M:%S")


    for i in range(run_count):
        traj = Trajectory_Buffer()
        e = np.random.random()
        # e = max(0.1, min(e, 0.9))
        print(e)
        circle.set_restitution(e)
        should_update = True
        circle.set_position(start_pos)
        circle.set_velocity(vec2(0.0, 0.0))

        while(should_update):
            total_time = total_time + dt
            circle.add_force(circle.get_mass() * gravity)
            world.step(dt)

            contact = collision_check.circle_and_plane(circle, plane)
            if (contact.collided):
                collision_resolver.resolve_collision_circle_plane(circle, plane, contact)

            if (total_time >= config["End Time"]):
                print("Closing")
                total_time = 0
                should_update = False

            traj.record(total_time, circle.get_position(), circle.get_velocity(), circle.get_restitution())
        traj.finalise()
        traj.save(i, runtimestamp)
        validate_restitution(i, traj.load(i, runtimestamp), circle, plane, dt, run_count, runtimestamp)
        print(f'RUN_DIR={i}/{runtimestamp}')

