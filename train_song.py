import numpy as np

# =====================
# === Utility funcs ===
# =====================

def save_parameters(parameters, filename):
    np.savez_compressed(filename, **parameters)
    print(f"Parameters saved to {filename}.npz")

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# ================================
# === Parameter Initialization ===
# ================================

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

# =====================
# === Forward Pass ===
# =====================

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linear_forward_for_res(A, W, b, X):
    Z = W.dot(A) + b + X
    cache = (A, W, b, X)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation, X=None):
    if X is not None and A_prev.shape == X.shape:
        Z, linear_cache = linear_forward_for_res(A_prev, W, b, X)
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'relu':
        A, activation_cache = relu(Z)
    else:
        raise ValueError("Only 'relu' activation allowed in hidden layers.")
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    skip = None

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters[f'W{l}'],
            parameters[f'b{l}'],
            activation='relu',
            X=skip
        )
        caches.append(cache)
        skip = A_prev if A_prev.shape == A.shape else None

    ZL, linear_cache = linear_forward(A, parameters[f'W{L}'], parameters[f'b{L}'])
    AL = ZL
    caches.append((linear_cache, None))
    return AL, caches

# ====================
# === Cost & Back ===
# ====================

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1 / (2 * m)) * np.sum((AL - Y) ** 2)
    return cost

def compute_cost_per_label(AL, Y):
    return 0.5 * (AL - Y) ** 2

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dA_prev = W.T.dot(dZ)
    dW = (1 / m) * dZ.dot(A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db

def linear_backward_for_res(dZ, cache):
    A_prev, W, b, X = cache
    m = A_prev.shape[1]
    dA_main = W.T.dot(dZ)
    dW = (1 / m) * dZ.dot(A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_skip = dZ
    dA_prev = dA_main + dA_skip
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    else:
        raise ValueError("Only 'relu' activation allowed in hidden layers.")
    if len(linear_cache) == 4:
        dA_prev, dW, db = linear_backward_for_res(dZ, linear_cache)
    else:
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)

    # last layer (linear)
    linear_cache, _ = caches[L-1]
    dZL = (AL - Y) / Y.shape[1]
    dA_prev, dW, db = linear_backward(dZL, linear_cache)
    grads[f'dA{L-1}'] = dA_prev
    grads[f'dW{L}'] = dW
    grads[f'db{L}'] = db

    # hidden layers
    for l in reversed(range(L-1)):
        dA_curr = grads[f'dA{l+1}']
        cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(dA_curr, cache, activation='relu')
        grads[f'dA{l}'] = dA_prev
        grads[f'dW{l+1}'] = dW
        grads[f'db{l+1}'] = db

    return grads

# ====================
# === Parameter Upd ===
# ====================

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return parameters

# ====================
# ====== TRAIN ======
# ====================

if __name__ == '__main__':
    # ⬇⬇⬇ 입력 차원만 3 -> 4로 수정 (출력 4는 그대로)
    layer_dims = [4, 32, 64, 128, 256, 128, 64, 32, 4]

    # (초기 실행 시에는 아래 한 줄 주석 해제해서 가중치 초기화)
    parameters = initialize_parameters_deep(layer_dims)

    # 기존처럼 파라미터 파일을 불러쓰는 경우:
    # param_data = np.load("model_parameters_resnet.npz")
    # parameters = {k: param_data[k] for k in param_data.files}

    # 내가 만들어둔 NPZ를 그대로 사용 (X: (N,4), Y: (N,4))
    data = np.load('robot_hand_data.npz')
    X_data, Y_data = data['camera_data'], data['joint_data']

    # ⬇ 기존 코드의 정규화 부분은 건드리지 말라는 의도로, 스케일 1.0만 적용
    X_data = X_data.astype(np.float64) / 650.0
    Y_data = Y_data.astype(np.float64) / 4100.0

    # 학습 루프 형태를 유지하기 위해 (features, samples)로 전치
    X_data, Y_data = X_data.T, Y_data.T  # (4, m), (4, m)
    m = X_data.shape[1]

    num_epochs = 2000
    lr = 0.0002
    total_iters = num_epochs * m

    for i in range(total_iters):
        idx = i % m
        X_input = X_data[:, idx].reshape(4, 1)  # ⬅ 3 -> 4로 수정
        Y_label = Y_data[:, idx].reshape(4, 1)

        AL, caches = L_model_forward(X_input, parameters)
        cost = compute_cost(AL, Y_label)
        grads = L_model_backward(AL, Y_label, caches)
        parameters = update_parameters(parameters, grads, lr)

        if idx == 0:
            epoch = i // m
            loss = cost * 100
            print(f"[Epoch {epoch}] Cost: {cost:.6f}, Loss: {loss:.2f}%")
            print(f" Predicted: {AL.flatten()} | Label: {Y_label.flatten()}")

    save_parameters(parameters, "model_parameters_resnet.npz")