import numpy as np

_a = 26
CHAR_LOOKUP = {" ": _a, ".": _a + 1, ",": _a + 2, ";": _a + 3, "'": _a + 4}
ONE_HOT_SIZE = _a + len(CHAR_LOOKUP)
ONE_HOT_BASE = np.zeros(ONE_HOT_SIZE)

def one_hot(i):
    arr = np.copy(ONE_HOT_BASE)
    arr[i] = 1
    return np.array([arr])

def encode(text):
    encoded = []
    for c in text:
        o = ord(c)
        if (o >= ord("a") and o <= ord("z")):
            o -= ord("a")
        elif (o >= ord("A") and o <= ord("Z")):
            o -= ord("A")
        elif c in CHAR_LOOKUP:
            o = CHAR_LOOKUP[c]
        else:
            continue
        encoded.append(one_hot(o))
    return np.array(encoded)
    
    
