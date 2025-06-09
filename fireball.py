# *Guide*

#charge - gets "energy" to use other attacks
#shield - shields you from any attack except megaball; costs 0 charges
#fireball - requires 1 charge to use and you beat the opponent only if they use charge
#iceball - requires 2 charges to use and it can beat fireball and charge
#megaball - requires 5 charges and when you use it you instantly win (unless the other person uses megaball too)
#if two people use the same attack (eg. iceball vs iceball), nothing happens except they both lose the charges

#click the "run" button to start the game|
#what the computer does is COMPLETELY randomized (no machine learning or optimization YET)
#it will keep asking you for input (where you enter your move) and it keeps going until someone wins 

#tell me if you find any bugs

# Made by Dhruv Sheth 

import random
from colorama import Fore
import math

mycount = 0
compcount = 0

# possible moves
moves = ["charge", "fireball", "iceball", "shield", "megaball"]
moves1 = ["charge" , "fireball", "iceball", "shield"]
moves2 = ["charge" , "fireball", "shield"]
moves3 = ["charge" , "shield"]


n = moves3 #computer possible moveset
m = moves3 #player possible moveset
gameStart = ("Game started!")
print(f'\033[32;1m{gameStart}\033[0m')
for i in range(0,100):
    a = input() #player move
    b = random.choice(n) #computer move
    if a == "end":
        exit()
    #legal move
    if a in moves:
        if a not in m:
            print ("not enough charges!")
            exit() 
    #spellcheck
    if a not in moves:
        print("Error")
        exit()

    #charge counting
    if a == "fireball": 
        mycount -= 1
    elif a == "iceball":
        mycount -= 2
    if b == "fireball":
        compcount -= 1
    elif b == "iceball":
        compcount -= 2
    if a == "charge":
        mycount += 1
    if b == "charge":
        compcount += 1

    #legal move checker
    if compcount == 0:
        n = moves3
    if compcount == 1:
        n = moves2
    if 2 <= compcount < 5:
        n = moves1
    if compcount >= 5:
        n = moves
    if mycount == 0:
        m = moves3
    if mycount == 1:
        m = moves2
    if 2 <= mycount < 5:
        m = moves1
    if mycount >= 5:
        m = moves

    #prints
    compprint = (f"The Computer chose {b} as their move!") 
    youprint = (f"You chose {a} as your move!")
    print(f'\033[94;1m{compprint}\033[0m')
    print(f'\033[94;1m{youprint}\033[0m')
    print(f'\033[36;1m{a}\033[0m',f'\033[31;1m{b}\033[0m',f'\033[36;1m{mycount}\033[0m',f'\033[31;1m{compcount}\033[0m')

    #comp vs player logic
    if a == "shield" and b != "megaball":
        continue
    elif b == "shield" and a != "megaball":
        continue
    elif a == "iceball" and b == "iceball":
        continue
    if a == "megaball" and b == "megaball":
        continue
    elif a == "charge" and b == "fireball" or b == "iceball" or b == "megaball":
        print ("You lost :(")
        exit()
    if a == "fireball" and b == "iceball" or b == "megaball":
        print ("You lost :(")
        exit()
    elif b == "charge" and a == "fireball" or a == "iceball" or a == "megaball":
        print ("You won :)")
        exit()
    if a == "fireball" and b == "charge":
        print ("You won :)")
        exit()
    elif a == "iceball" and b == "charge":
        print ("You won :)")
        exit()
    if b == "fireball" and a == "iceball" or a == "megaball":
        print ("You won :)")
        exit()
    elif b == "megaball" and a != "megaball": 
        print ("You lost")
        exit()     
    elif a == "megaball" and b != "megaball": 
        print ("You won :)")
        exit()
