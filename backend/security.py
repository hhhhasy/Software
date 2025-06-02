from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import os

# 建议通过环境变量配置密钥
AES_KEY = os.getenv("AES_KEY", "ThisIsASecretKey").encode("utf-8")  # 必须是 16, 24, 32 字节
AES_IV = os.getenv("AES_IV", "ThisIsInitVector").encode("utf-8")   # 必须是 16 字节 在 CBC 模式下，IV 用来防止相同的明文每次加密都产生相同的密文。

def encrypt_password(plain_text: str) -> str:
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
    ct_bytes = cipher.encrypt(pad(plain_text.encode('utf-8'), AES.block_size))
    return base64.b64encode(ct_bytes).decode('utf-8')

def decrypt_password(cipher_text: str) -> str:
    try:
        cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
        ct = base64.b64decode(cipher_text)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        return pt.decode('utf-8')
    except Exception as e:
        raise ValueError("解密失败") from e
