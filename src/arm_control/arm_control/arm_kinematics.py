import math
from dataclasses import dataclass


MIN_RHO = 160.0
MAX_RHO = 190.0
DEFAULT_PICKUP_Z = 15.0

L1 = 101.0
L2 = 94.0

BASE_SERVO_CENTER = 120.0
WRIST_DOWN_ORIENTATION_DEG = -90.0

BASE_LIMITS = (75.0, 165.0)     # P5: straight at 120°, initial: 120
SHOULDER_LIMITS = (30.0, 120.0) # P4: straight at 120°         initial: 180
ELBOW_LIMITS = (150.0, 210.0)   # P3: DONT USE TOO MUCH 200-220    initial: 220
WRIST_LIMITS = (40.0, 170.0)    # P2: straight at 120°      initial:30
ORIENTATION_WRIST = (17.0, 140.0)   #P1: straight at 120°   initial: 120
GRIPPER_LIMITS = (10.0, 100.0)  #P0 open: 10°, close: 100°  initial: 40

INITIAL_POSITION = [40, 120, 30, 220, 180, 120]
START_SAFE_POSITION = [40, 120, 166.4, 220, 180, 120]
IDLE_POSE = {'p2': 16.6, 'p4': 89.8}
HOLDING_POSE = {'p2': 30.0, 'p3': 170.0, 'p4': 120.0}
DROP_POSE = {'p2': 90.0, 'p4': 80.0}


@dataclass(frozen=True)
class PlanarTarget:
    rho: float
    alpha_deg: float
    z: float


@dataclass(frozen=True)
class ArmJointTarget:
    base: float
    shoulder: float
    elbow: float
    wrist: float


def clamp_rho(rho: float) -> float:
    return max(MIN_RHO, min(MAX_RHO, rho))


def is_rho_safe(rho: float) -> bool:
    return MIN_RHO <= rho <= MAX_RHO


def is_joint_value_in_limits(value: float, limits) -> bool:
    return limits[0] <= value <= limits[1]


def is_joint_triplet_safe(shoulder: float, elbow: float, wrist: float) -> bool:
    return (
        is_joint_value_in_limits(shoulder, SHOULDER_LIMITS)
        and is_joint_value_in_limits(elbow, ELBOW_LIMITS)
        and is_joint_value_in_limits(wrist, WRIST_LIMITS)
    )


def is_base_safe(base: float) -> bool:
    return is_joint_value_in_limits(base, BASE_LIMITS)


def inverse_kinematics_2d(rho: float, z: float, orientation_deg: float = WRIST_DOWN_ORIENTATION_DEG):
    if not is_rho_safe(rho):
        raise ValueError(f'rho {rho:.2f} outside safe range [{MIN_RHO}, {MAX_RHO}]')

    orientation = math.radians(orientation_deg)
    r2 = rho * rho + z * z
    cos_t2 = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)

    if abs(cos_t2) > 1.0:
        raise ValueError(f'planar target rho={rho:.2f}, z={z:.2f} is unreachable')

    t2 = -math.acos(cos_t2)
    k1 = L1 + L2 * math.cos(t2)
    k2 = L2 * math.sin(t2)
    t1 = math.atan2(z, rho) - math.atan2(k2, k1)
    t3 = orientation - t1 - t2

    shoulder = 30.0 + math.degrees(t1)
    elbow = 120.0 - math.degrees(t2)
    wrist = 120.0 + math.degrees(t3)

    if not is_joint_triplet_safe(shoulder, elbow, wrist):
        raise ValueError(
            'joint target outside limits: '
            f'shoulder={shoulder:.2f}, elbow={elbow:.2f}, wrist={wrist:.2f}'
        )

    return shoulder, elbow, wrist


def alpha_to_base_servo(alpha_deg: float) -> float:
    return BASE_SERVO_CENTER + alpha_deg


def make_planar_target(rho: float, alpha_deg: float, z: float = DEFAULT_PICKUP_Z) -> PlanarTarget:
    return PlanarTarget(rho=clamp_rho(rho), alpha_deg=alpha_deg, z=z)


def planar_to_joint_target(target: PlanarTarget) -> ArmJointTarget:
    shoulder, elbow, wrist = inverse_kinematics_2d(target.rho, target.z)
    base = alpha_to_base_servo(target.alpha_deg)

    if not is_base_safe(base):
        raise ValueError(
            f'base target outside limits: base={base:.2f}, allowed={BASE_LIMITS}'
        )

    return ArmJointTarget(
        base=base,
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
    )


def pixel_y_to_rho_step(pixel_error_y: float, pixel_to_mm: float) -> float:
    return pixel_error_y * pixel_to_mm


def rho_midpoint() -> float:
    return (MIN_RHO + MAX_RHO) / 2.0
