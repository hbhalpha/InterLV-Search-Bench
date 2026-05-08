#!/usr/bin/env python3
"""
InterLV Data Encryption/Decryption Tool
对 data/level1.json, level2.json, level3.json 进行 AES-256-GCM 加密/解密
"""

import os
import json
import argparse
import getpass
import base64
import hashlib
from pathlib import Path
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

# ─────────────────────────────────────────────
# 若没有 cryptography 库，使用纯标准库 AES-CTR 实现
# ─────────────────────────────────────────────
if not HAS_CRYPTOGRAPHY:
    import struct
    import hmac as _hmac

    def _aes_encrypt_block(key: bytes, block: bytes) -> bytes:
        """使用 hashlib 模拟 AES 单块加密（仅用于无 cryptography 库的降级方案）"""
        # 警告：这不是真正的 AES，仅作为无依赖降级方案
        # 推荐安装 cryptography: pip install cryptography
        raise RuntimeError(
            "请安装 cryptography 库: pip install cryptography\n"
            "然后重新运行此脚本。"
        )


DATA_DIR = Path(__file__).parent / "data"
LEVEL_FILES = ["level1.json", "level2.json", "level3.json"]
ENCRYPTED_EXT = ".enc"

ITERATIONS = 390_000   # PBKDF2 迭代次数（NIST 推荐 ≥ 310000）
SALT_LEN   = 16        # bytes
NONCE_LEN  = 12        # bytes（GCM 标准 nonce）
KEY_LEN    = 32        # bytes（AES-256）


# ═══════════════════════════════════════════
#  核心加密 / 解密函数
# ═══════════════════════════════════════════

def _derive_key(password: str, salt: bytes) -> bytes:
    """从密码派生 AES-256 密钥（PBKDF2-HMAC-SHA256）"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=salt,
        iterations=ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt_data(plaintext: bytes, password: str) -> bytes:
    """
    加密格式（二进制）:
        [4B magic][1B version][16B salt][12B nonce][ciphertext+16B tag]
    """
    magic   = b"ILVD"   # InterLV Data
    version = b"\x01"
    salt    = os.urandom(SALT_LEN)
    nonce   = os.urandom(NONCE_LEN)
    key     = _derive_key(password, salt)
    aesgcm  = AESGCM(key)
    ct      = aesgcm.encrypt(nonce, plaintext, None)   # GCM tag 自动附加
    return magic + version + salt + nonce + ct


def decrypt_data(ciphertext: bytes, password: str) -> bytes:
    """解密并验证完整性，密码错误或数据损坏时抛出异常"""
    magic   = b"ILVD"
    if ciphertext[:4] != magic:
        raise ValueError("文件格式错误：不是有效的加密文件（magic bytes 不匹配）")
    version = ciphertext[4:5]
    if version != b"\x01":
        raise ValueError(f"不支持的加密版本: {version!r}")
    offset  = 5
    salt    = ciphertext[offset : offset + SALT_LEN];   offset += SALT_LEN
    nonce   = ciphertext[offset : offset + NONCE_LEN];  offset += NONCE_LEN
    ct      = ciphertext[offset:]
    key     = _derive_key(password, salt)
    aesgcm  = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ct, None)
    except Exception:
        raise ValueError("解密失败：密码错误或文件已损坏")
    return plaintext


# ═══════════════════════════════════════════
#  文件级别操作
# ═══════════════════════════════════════════

def encrypt_file(src: Path, dst: Path, password: str) -> None:
    """加密单个文件"""
    raw = src.read_bytes()
    enc = encrypt_data(raw, password)
    dst.write_bytes(enc)
    size_ratio = len(enc) / len(raw)
    print(f"  [加密] {src.name}  →  {dst.name}  "
          f"({len(raw):,} B → {len(enc):,} B, ×{size_ratio:.2f})")


def decrypt_file(src: Path, dst: Path, password: str) -> None:
    """解密单个文件"""
    enc = src.read_bytes()
    raw = decrypt_data(enc, password)
    dst.write_bytes(raw)
    # 快速验证：确认解密结果是合法 JSON
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        dst.unlink(missing_ok=True)
        raise ValueError(f"解密后的内容不是有效 JSON，已删除输出文件: {dst}")
    print(f"  [解密] {src.name}  →  {dst.name}  "
          f"({len(enc):,} B → {len(raw):,} B)")


# ═══════════════════════════════════════════
#  批量操作
# ═══════════════════════════════════════════

def encrypt_all(password: str, output_dir: Optional[Path] = None) -> None:
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n开始加密（输出目录: {out}）")
    for name in LEVEL_FILES:
        src = DATA_DIR / name
        if not src.exists():
            print(f"  [跳过] {name} 不存在")
            continue
        dst = out / (name + ENCRYPTED_EXT)
        encrypt_file(src, dst, password)
    print("加密完成 ✓\n")


def decrypt_all(password: str, input_dir: Optional[Path] = None,
                output_dir: Optional[Path] = None) -> None:
    inp = input_dir or DATA_DIR
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n开始解密（来源目录: {inp}，输出目录: {out}）")
    for name in LEVEL_FILES:
        src = inp / (name + ENCRYPTED_EXT)
        if not src.exists():
            print(f"  [跳过] {name + ENCRYPTED_EXT} 不存在")
            continue
        dst = out / name
        decrypt_file(src, dst, password)
    print("解密完成 ✓\n")


# ═══════════════════════════════════════════
#  单文件操作（支持任意 .json 或 .enc 文件）
# ═══════════════════════════════════════════

def encrypt_single(src: str, dst: Optional[str], password: str) -> None:
    sp = Path(src)
    dp = Path(dst) if dst else sp.with_suffix(sp.suffix + ENCRYPTED_EXT)
    encrypt_file(sp, dp, password)


def decrypt_single(src: str, dst: Optional[str], password: str) -> None:
    sp = Path(src)
    if dst:
        dp = Path(dst)
    else:
        # 移除 .enc 后缀，若无则加 _decrypted
        stem = sp.stem if sp.suffix == ENCRYPTED_EXT else sp.name + "_decrypted"
        dp = sp.parent / stem
    decrypt_file(sp, dp, password)


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

def get_password(confirm: bool = False) -> str:
    pwd = getpass.getpass("请输入密码: ")
    if not pwd:
        raise ValueError("密码不能为空")
    if confirm:
        pwd2 = getpass.getpass("再次确认密码: ")
        if pwd != pwd2:
            raise ValueError("两次密码不一致")
    return pwd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto_tool",
        description="InterLV level1/2/3 数据文件 AES-256-GCM 加密/解密工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量加密所有 level 文件（交互式输入密码）
  python crypto_tool.py encrypt-all

  # 批量解密
  python crypto_tool.py decrypt-all

  # 加密单个文件，指定输出路径
  python crypto_tool.py encrypt-file data/level1.json -o data/level1_enc.enc

  # 解密单个文件
  python crypto_tool.py decrypt-file data/level1.json.enc -o data/level1_out.json

  # 通过 --password 直接传入密码（不推荐，会出现在 shell 历史）
  python crypto_tool.py encrypt-all --password mysecret
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pwd_kwargs = dict(help="加密/解密密码（不传则交互式输入）", default=None)

    # ── encrypt-all ──
    p_ea = sub.add_parser("encrypt-all", help="批量加密 level1/2/3.json")
    p_ea.add_argument("--password", "-p", **pwd_kwargs)
    p_ea.add_argument("--output-dir", "-d", help="加密文件输出目录（默认 data/）")

    # ── decrypt-all ──
    p_da = sub.add_parser("decrypt-all", help="批量解密 level1/2/3.json.enc")
    p_da.add_argument("--password", "-p", **pwd_kwargs)
    p_da.add_argument("--input-dir",  "-i", help="加密文件所在目录（默认 data/）")
    p_da.add_argument("--output-dir", "-d", help="解密文件输出目录（默认 data/）")

    # ── encrypt-file ──
    p_ef = sub.add_parser("encrypt-file", help="加密单个文件")
    p_ef.add_argument("--password", "-p", **pwd_kwargs)
    p_ef.add_argument("src", help="源文件路径")
    p_ef.add_argument("--output", "-o", help="输出路径（默认: <src>.enc）")

    # ── decrypt-file ──
    p_df = sub.add_parser("decrypt-file", help="解密单个文件")
    p_df.add_argument("--password", "-p", **pwd_kwargs)
    p_df.add_argument("src", help="加密文件路径（.enc）")
    p_df.add_argument("--output", "-o", help="输出路径（默认: 去掉 .enc 后缀）")

    return parser


def main() -> None:
    if not HAS_CRYPTOGRAPHY:
        print("错误: 缺少 cryptography 库，请先安装:")
        print("  pip install cryptography")
        raise SystemExit(1)

    parser = build_parser()
    args = parser.parse_args()

    # 获取密码
    try:
        if args.password:
            pwd = args.password
        elif args.command in ("encrypt-all", "encrypt-file"):
            pwd = get_password(confirm=True)
        else:
            pwd = get_password(confirm=False)
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n错误: {e}")
        raise SystemExit(1)

    # 执行操作
    try:
        if args.command == "encrypt-all":
            out = Path(args.output_dir) if args.output_dir else None
            encrypt_all(pwd, output_dir=out)

        elif args.command == "decrypt-all":
            inp = Path(args.input_dir)  if args.input_dir  else None
            out = Path(args.output_dir) if args.output_dir else None
            decrypt_all(pwd, input_dir=inp, output_dir=out)

        elif args.command == "encrypt-file":
            encrypt_single(args.src, args.output, pwd)

        elif args.command == "decrypt-file":
            decrypt_single(args.src, args.output, pwd)

    except (ValueError, FileNotFoundError) as e:
        print(f"\n错误: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
