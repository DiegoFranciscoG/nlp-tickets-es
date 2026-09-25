"""Genera el hash Argon2id de una contraseña para DEMO_PASSWORD_HASH o REVIEWER_PASSWORD_HASH.

Uso: python demo/hash_password.py   (la contraseña se pide sin mostrarla en pantalla)
"""

from getpass import getpass

from auth import hash_password

if __name__ == "__main__":
    password = getpass("Contraseña: ")
    if len(password) < 12:
        raise SystemExit("usa al menos 12 caracteres")
    print(hash_password(password))
