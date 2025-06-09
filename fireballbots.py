



import random
from colorama import Fore
import math
from spellchecker import SpellChecker

comp1count = 0
comp2count = 0

# possible moves
moves = ["charge", "fireball", "iceball", "shield", "megaball"]
moves1 = ["charge" , "fireball", "iceball", "shield"]
moves2 = ["charge" , "fireball", "shield"]
moves3 = ["charge" , "shield"]


n = moves3 #comp1 possible moveset
m = moves3 #comp2 possible moveset
gameStart = ("Game started!")
print(f'\033[32;1m{gameStart}\033[0m')
for i in range(0,100):
    a = random.choice(n) #comp1 move
    b = random.choice(n) #comp2 move

    #charge counting
    if a == "fireball": 
        comp1count -= 1
    elif a == "iceball":
        comp1count -= 2
    if b == "fireball":
        comp2count -= 1
    elif b == "iceball":
        comp2count -= 2
    if a == "charge":
        comp1count += 1
    if b == "charge":
        comp2count += 1

    #legal move checker
    if comp2count == 0:
        n = moves3
    if comp2count == 1:
        n = moves2
    if 2 <= comp2count < 5:
        n = moves1
    if comp2count >= 5:
        n = moves
    if comp1count == 0:
        m = moves3
    if comp1count == 1:
        m = moves2
    if 2 <= comp1count < 5:
        m = moves1
    if comp1count >= 5:
        m = moves

    #prints
    youprint = (f"Comp1 chose {a} as your move!")
    compprint = (f"Comp2 chose {b} as their move!")
    print(f'\033[94;1m{compprint}\033[0m')
    print(f'\033[94;1m{youprint}\033[0m')
    print(f'\033[36;1m{a}\033[0m',f'\033[31;1m{b}\033[0m',f'\033[36;1m{comp1count}\033[0m',f'\033[31;1m{comp2count}\033[0m')

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
        print ("Comp2 Wins")
        exit()
    if a == "fireball" and b == "iceball" or b == "megaball":
        print ("Comp2 Wins")
        exit()
    elif b == "charge" and a == "fireball" or a == "iceball" or a == "megaball":
        print ("Comp1 Wins")
        exit()
    if a == "fireball" and b == "charge":
        print ("Comp1 Wins")
        exit()
    elif a == "iceball" and b == "charge":
        print ("Comp1 Wins")
        exit()
    if b == "fireball" and a == "iceball" or a == "megaball":
        print ("Comp1 Wins")
        exit()
    elif b == "megaball" and a != "megaball": 
        print ("Comp2 Wins")
        exit()     
    elif a == "megaball" and b != "megaball": 
        print ("Comp1 Wins")
        exit()
