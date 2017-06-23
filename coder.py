import numpy as np

_a = 26
ORD_LOOKUP = {" ": _a, ".": _a + 1, ",": _a + 2, ";": _a + 3, "'": _a + 4}
CHAR_LOOKUP = [" ", ".", ",", ";", "'"]
ONE_HOT_SIZE = _a + len(ORD_LOOKUP)
ONE_HOT_BASE = np.zeros(ONE_HOT_SIZE)

def one_hot(i):
    arr = np.copy(ONE_HOT_BASE)
    arr[i] = 1
    return np.array(arr)

def encode(text):
    encoded = []
    for c in text:
        o = ord(c)
        if (o >= ord("a") and o <= ord("z")):
            o -= ord("a")
        elif (o >= ord("A") and o <= ord("Z")):
            o -= ord("A")
        elif c in ORD_LOOKUP:
            o = ORD_LOOKUP[c]
        else:
            continue
        encoded.append(one_hot(o))
    return np.array(encoded)

def decode(text):
    decoded = []
    for i in text:
        if i < 0:
            continue
        elif i < _a:
            c = chr(i + ord("A"))
        elif i < (ONE_HOT_SIZE):
            c = CHAR_LOOKUP[i - _a]
        else:
            continue
        decoded.append(c)
    return decoded
    
