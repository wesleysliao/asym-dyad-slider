#!/usr/bin/env python3


import numpy as np


class DynamicObject:
    POS = 0
    VEL = 1
    ACC = 2

    def __init__(self, mass):
        self.mass = mass
        self.state = np.zeros((3,))
        self.force = 0.0


    def step(self, dt_s):
        if self.mass != 0.0:
            self.state[self.ACC] = (self.force / self.mass)
        self.force = 0.0
        self.state[self.VEL] += self.state[self.ACC] * dt_s
        self.state[self.POS] += self.state[self.VEL] * dt_s


    def add_force(self, force):
        self.force += force


class Constraint:
    def __init__(self, target):
        self.target = target
        self.references = []

    def apply(self):
        pass



class Damping(Constraint):
    def __init__(self, b, target):
        self.b = b
        super().__init__(target)

    def apply(self):
        force = (self.target.state[1] * -self.b)
        self.target.add_force(force)



class CompressionSpring(Constraint):
    def __init__(self, target, reference, spring_coeff, resting_length):
        self.spring_coeff = spring_coeff
        self.resting_length = resting_length
        super().__init__(target)
        self.references.append(reference)

    def apply(self):

        pos1 = self.target.state[0]
        pos2 = self.references[0].state[0]

        compression = (pos2 + self.resting_length) - pos1
        if compression > 0:
           force = compression * self.spring_coeff

           self.target.add_force(force)


class TensionSpring(Constraint):
    def __init__(self, target, reference, spring_coeff, resting_length):
        self.spring_coeff = spring_coeff
        self.resting_length = resting_length
        super().__init__(target)
        self.references.append(reference)

    def apply(self):

        pos1 = self.target.state[0]
        pos2 = self.references[0].state[0]

        tension = pos1 - (pos2 + self.resting_length)
        if tension > 0:
           force = tension * -self.spring_coeff

           self.target.add_force(force)


class BindPosition(Constraint):
    def __init__(self, target, reference, proportion=1.0):
        super().__init__(target)
        self.references.append(reference)
        self.proportion = proportion

    def apply(self):
        self.target.state[0] = self.references[0].state[0] * self.proportion


class PositionLimits(Constraint):
    def __init__(self, target, pos, neg):
        self.bound_pos = pos
        self.bound_neg = neg

        super().__init__(target)

    def apply(self):
       if(self.target.state[0] > self.bound_pos):
           self.target.state[0] = self.bound_pos
       elif(self.target.state[0] < self.bound_neg):
           self.target.state[0] = self.bound_neg
