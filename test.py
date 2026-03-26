import math

#All measurements are in mm 

L1 = 101
L2 = 94
L3_OPEN = 145
L3_CLOSED = 170

Z_LOWER_LIMIT = -155
RHO_LOWER_LIMIT = 100

t1 = math.radians(50)
t2 = math.radians(120)
t3 = math.radians(100)

theta1 = t1
theta2 = t1 - (math.pi - t2)
theta3 = theta2 - (math.pi - t3)

rho = L1*math.cos(theta1) + L2*math.cos(theta2) + L3_CLOSED*math.cos(theta3)
z = L1*math.sin(theta1) + L2*math.sin(theta2) + L3_CLOSED*math.sin(theta3)
(3 * math.pi) / 2 = t1+t2+t3 

print(rho,z)