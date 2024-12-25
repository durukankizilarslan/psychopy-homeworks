from random import randint
import time

def rock_paper_scissors():
    # Setting up the variables for the game
    counter = 0
    winner = None
    choices = ["rock", "paper", "scissors"]
    cp_win = 0
    usr_win = 0

    print("Welcome to Rock-Paper-Scissors! The game will be played for 3 turns.")

    while counter < 3:
        print(f"\nTurn {counter + 1}:")

        # Get valid user input
        while True:
            user_choice = input("Enter your choice in a written form (rock, paper, or scissors): ").lower()
            if user_choice in choices:
                break
            else:
                print("Invalid input. Please write rock, paper, or scissors.")

        # Random computer choice
        computer_choice = choices[randint(0, 2)]
        print(f"Computer chose: {computer_choice}")

        # Determine the winner of the round
        if user_choice == computer_choice:
            print("It's a tie this round!")
        elif (user_choice == "rock" and computer_choice == "scissors") or \
             (user_choice == "paper" and computer_choice == "rock") or \
             (user_choice == "scissors" and computer_choice == "paper"):
            print("You win this round!")
            usr_win += 1
        else:
            print("Computer wins this round!")
            cp_win += 1

        counter += 1

    # Determine the overall winner
    if cp_win == usr_win:
        winner = "no one"
    elif cp_win > usr_win:
        winner = "the computer"
    else:
        winner = "the user"

    print(f"\nThe game has ended, {winner} has won!")
    time.sleep(5)

rock_paper_scissors()