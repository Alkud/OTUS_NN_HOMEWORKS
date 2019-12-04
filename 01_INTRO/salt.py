import hashlib

def salt(text):
    m = hashlib.md5()
    m.update(bytes(text, 'utf-8'))
    return m.hexdigest()[-2::-2]