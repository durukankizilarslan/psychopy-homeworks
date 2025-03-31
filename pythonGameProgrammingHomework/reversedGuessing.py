def reversed_guessing():
    import random

    print("Welcome to the Reverse Number Guessing Game!")
    print("Think of a number between 1 and 50, and the computer will try to guess it.")

    lower_bound = 1
    upper_bound = 50
    guessed_numbers = set()
    correct_guess = False

    while not correct_guess:
        # Computer makes a guess within the current bounds, avoiding previous guesses
        while True:
            computer_guess = random.randint(lower_bound, upper_bound)
            if computer_guess not in guessed_numbers:
                break

        guessed_numbers.add(computer_guess)
        print(f"The computer guesses: {computer_guess}")

        # Get feedback from the user
        feedback = input("Is the guess too high (H), too low (L), or correct (C)? ").lower()

        if feedback == 'c':
            print(f"The computer guessed your number {computer_guess} correctly!")
            correct_guess = True
        elif feedback == 'h':
            upper_bound = computer_guess - 1
        elif feedback == 'l':
            lower_bound = computer_guess + 1
        else:
            print("Invalid input. Please enter 'H' for too high, 'L' for too low, or 'C' for correct.")

        # Check if bounds are still valid
        if lower_bound > upper_bound:
            print("The feedback seems inconsistent. Please start the game again.")
            break

# Run the game
reversed_guessing()
