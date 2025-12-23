import random

# Fake encryption (XOR shift)
def simple_encrypt(text, key=7):
    return ''.join([chr(ord(c) ^ key) for c in text])

# Redact some keywords
def redact_text(text):
    words = text.split()
    num_to_redact = random.randint(1, len(words)//2)
    for i in random.sample(range(len(words)), num_to_redact):
        words[i] = '[REDACTED]'
    return ' '.join(words)
