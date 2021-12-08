from keys import PressKey, ReleaseKey
import time
from constants import W_HEX,A_HEX,S_HEX,D_HEX

def straight():
    PressKey(W_HEX)
    ReleaseKey(A_HEX)
    ReleaseKey(D_HEX)
    ReleaseKey(S_HEX)
    print('W')

def left():
    PressKey(A_HEX)
    ReleaseKey(W_HEX)
    ReleaseKey(S_HEX)
    ReleaseKey(D_HEX)
    print('A')
def right():
    PressKey(D_HEX)
    ReleaseKey(W_HEX)
    ReleaseKey(A_HEX)
    ReleaseKey(S_HEX)
    print('D')
def acc_left():
    PressKey(W_HEX)
    PressKey(A_HEX)
    ReleaseKey(S_HEX)
    ReleaseKey(D_HEX)
    print('WA')
def acc_right():
    PressKey(W_HEX)
    PressKey(D_HEX)
    ReleaseKey(S_HEX)
    ReleaseKey(A_HEX)
    print('WD')
def brake():
    PressKey(S_HEX)
    ReleaseKey(A_HEX)
    ReleaseKey(D_HEX)
    ReleaseKey(W_HEX)
    print('S')
def reverse_left():
    PressKey(S_HEX)
    PressKey(A_HEX)
    ReleaseKey(W_HEX)
    ReleaseKey(D_HEX)
    print('SA')
def reverse_right():
    PressKey(S_HEX)
    PressKey(D_HEX)
    ReleaseKey(W_HEX)
    ReleaseKey(A_HEX)
    print('SD')
def do_nothing():
    time.sleep(1)
    ReleaseKey(W_HEX)
    ReleaseKey(A_HEX)
    ReleaseKey(S_HEX)
    ReleaseKey(D_HEX)
    print('NK')
