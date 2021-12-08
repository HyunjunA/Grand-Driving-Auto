from keys import Keys
from time import sleep

def press_key(key) :
    keys = Keys()
    keys.direct_keys(key)

def release_key(key) :
    keys = Keys()
    keys.directKey(key, keys.key_release)