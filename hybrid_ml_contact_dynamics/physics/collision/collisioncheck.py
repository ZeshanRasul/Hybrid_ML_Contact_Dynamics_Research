import numpy as np
from hybrid_ml_contact_dynamics.physics.types import Vec2
from hybrid_ml_contact_dynamics.physics.primitives.circle import Circle
from hybrid_ml_contact_dynamics.physics.primitives.plane import Plane

class ContactData:
    collided: bool
    penetration: float
    contact_normal: Vec2
    
    def __init__(self, collided: bool, penetration: float, contact_normal: Vec2):
        self.collided = collided
        self.penetration = penetration
        self.contact_normal = contact_normal

class CollisionCheck:
    contact_data: ContactData


    def __init__(self):
        self.contact_data = ContactData(None, None, None)

    def circle_and_plane(self, circle: Circle, plane: Plane):
        n = plane.get_normal()
        c = circle.get_position()
        d = plane.get_offset()

        signed_distance = float(np.dot(n, c) - d)
        penetration = circle.get_radius() - signed_distance
        if (penetration >= 0):
            self.contact_data = ContactData(True, penetration, n) 
        else:
            self.contact_data = ContactData(False, 0.0, n)
        return self.contact_data
    

class CollisionResolver:
    def __init__(self):
        pass

    def resolve_collision_circle_plane(self, circle: Circle, plane: Plane, contact_data: ContactData):
        if (not contact_data.collided):
            return
        
        n = contact_data.contact_normal
        v = circle.get_velocity()

        c_pos = circle.get_position()

        circle.set_position(c_pos + n * contact_data.penetration)

        vn = float(np.dot(v, n))
        if (vn >= 0.0):
            return
        
        inv_mass = 1.0 / circle.get_mass()
        j = -(1.0 + circle.get_restitution()) * vn / inv_mass
        impulse = j * n

        circle.set_velocity(v + impulse * inv_mass)