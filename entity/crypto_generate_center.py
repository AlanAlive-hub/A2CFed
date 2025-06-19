import os
import torch
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pickle

class KeyGenerateCenter:
    def __init__(self, num_users):
        self.num_users = num_users
        self.masks = []
        self.key_pairs = []

    def generate_keys(self):
        for _ in range(self.num_users):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            self.key_pairs.append((public_key, private_key))

    def generate_aes_key(self):
        key = os.urandom(32)  # AES key size can be 16, 24, or 32 bytes
        iv = os.urandom(16)   # AES block size for CFB mode
        return key, iv

    def encrypt_and_sign_model(self, model, user_index):
        serialized_model = pickle.dumps(model.state_dict())
        public_key, private_key = self.key_pairs[user_index]
        aes_key, iv = self.generate_aes_key()

        # Encrypt the model state_dict with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_model = encryptor.update(serialized_model) + encryptor.finalize()

        # Encrypt the AES key with RSA
        encrypted_aes_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Sign the encrypted model
        signature = private_key.sign(
            encrypted_model,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return encrypted_model, encrypted_aes_key, iv, signature

    def decrypt_and_verify_model(self, encrypted_model, encrypted_aes_key, iv, signature, user_index):
        public_key, private_key = self.key_pairs[user_index]

        # Decrypt the AES key
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Decrypt the model
        cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_model_bytes = decryptor.update(encrypted_model) + decryptor.finalize()
        decrypted_model = pickle.loads(decrypted_model_bytes)

        # Verify the signature
        try:
            public_key.verify(
                signature,
                encrypted_model,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            verification = True
        except Exception:
            verification = False

        return decrypted_model, verification

# Test the class
if __name__ == "__main__":
    # num_users = 3
    # pg = KeyGenerateCenter(num_users)
    # pg.generate_keys()

    # # Sample model
    # model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    # import copy 
    # ori_m = copy.deepcopy(model)
    
    # pg.generate_masks()

    # # Blind model for user 0
    # pg.blind_model(model, 0)

    # # Encrypt and sign
    # encrypted_model, encrypted_aes_key, iv, signature = pg.encrypt_and_sign_model(model, 0)

    # # Decrypt and verify
    # decrypted_state_dict, is_verified = pg.decrypt_and_verify_model(encrypted_model, encrypted_aes_key,iv, signature, 0)
    # model.load_state_dict(decrypted_state_dict)  # Load decrypted model
    # pg.unblind_model(model, 0)                   # Unblind the model
    # print(ori_m == model)
    # print("Verification successful:", is_verified)

    import torch
    import torch.nn as nn

    # 定义一个简单的CNN模型
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # 实例化模型
    model = SimpleCNN()

    # 生成和应用盲化因子
    blind_factors = {}
    for name, param in model.named_parameters():
        # 生成与权重形状相同的随机盲化因子
        blind_factor = torch.randn_like(param.data)
        blind_factors[name] = blind_factor
        # 盲化权重
        param.data += blind_factor

    # 存储盲化后的权重，模拟发送到另一个地方
    blind_weights = {name: param.data.clone() for name, param in model.named_parameters()}

    # 去盲化模型权重
    for name, param in model.named_parameters():
        # 使用相同的盲化因子去盲化权重
        param.data = blind_weights[name] - blind_factors[name]

    # 验证权重恢复正确性
    for name, param in model.named_parameters():
        original = blind_weights[name] - blind_factors[name]
        restored = param.data
        if not torch.allclose(original, restored):
            print(f"Difference detected in layer: {name}")
        else:
            print(f"No difference in weights for layer {name}")

