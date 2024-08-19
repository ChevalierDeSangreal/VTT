import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion
# qua: x y z w
# eul: x y z
def qua2euler(qua):
    rotation_matrices = quaternion_to_matrix(
        qua[:, [3, 0, 1, 2]])
    euler_angles = matrix_to_euler_angles(
        rotation_matrices, "ZYX")[:, [2, 1, 0]]
    return euler_angles

def euler2qua(euler):
    rotation_matrices = euler_angles_to_matrix(euler, "ZYX")
    qua = matrix_to_quaternion(rotation_matrices)[:, [3, 2, 1, 0]]
    return qua

# euler_angles = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Example Euler angles
# quaternions = euler2qua(euler_angles)


quaternions = torch.tensor([[0.0641, 0.0912, 0.1534, 0.9819], [0.2556, 0.1748, 0.3276, 0.8927]])
# print(quaternions)
euler_angles2 = qua2euler(quaternions)
print(euler_angles2)
quaternions2 = euler2qua(euler_angles2)
print(quaternions2)