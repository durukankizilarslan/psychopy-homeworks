from psychopy import visual, event, core, gui
import random

# Set up the window
win = visual.Window(
    size=[800, 600], 
    color="gray", 
    units="norm"  # Use normalized units (-1 to 1 range)
)

# Step 1: Explanatory text
explanation_text = visual.TextStim(
    win, 
    text="Welcome to the Posner Task.\n\n"
         "In this task, you will see a fixation cross, followed by a cue and a target.\n\n"
         "But first, we will require your participant information.\n\n"
         "Press the SPACE key to start the task.",
    color="white",
    height=0.08,
    wrapWidth=1.5
)
explanation_text.draw()
win.flip()

# Wait for the user to press the space key to start
while True:
    keys = event.getKeys()
    if 'space' in keys:
        break

# Step 2: Collect participant information
info = {"Participant ID": "", "Age": "", "Gender": ["Male", "Female", "Other"]}
dlg = gui.DlgFromDict(info, title="Participant Information")
if not dlg.OK:  # If the user clicks 'Cancel', exit the program
    core.quit()

# Step 3: Display explanatory text for the cueing task
explanation_text = visual.TextStim(
    win,
    text=f"Welcome to the Posner Task, {info['Participant ID']}!\n\n"
         "In this task, you will see a fixation cross, followed by a cue and a target.\n\n"
         "The target will either be a coffee cup or a gun.\n"
         "The cue will predict the target's location most of the time, but not always.\n\n"
         "Press the arrow key corresponding to the target's location as quickly as possible.\n\n"
         "Press the SPACE key to start the task.",
    color="white",
    height=0.08,  # Text height
    wrapWidth=1.5  # Allow text wrapping
)
explanation_text.draw()
win.flip()

# Wait for the user to press the space key to start the task
while True:
    keys = event.getKeys()
    if 'space' in keys:  # Check if space key is pressed
        break

# Stimuli details
fixation = visual.TextStim(win, text='+', color='white', height=0.1)
cue = visual.TextStim(win, text='', color='yellow', height=0.1)
target_image = visual.ImageStim(win, size=(0.3, 0.3))  # Placeholder for the stimuli image

# Trial parameters
directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']  # Possible target directions
positions = {'UP': (0, 0.5), 'DOWN': (0, -0.5), 'LEFT': (-0.5, 0), 'RIGHT': (0.5, 0)}
stimuli_images = {'gun': 'path_to_gun_image.png', 'cup': 'path_to_cup_image.png'}
n_trials = 10  # Total number of trials
cue_validity = 0.8  # Probability of the cue being correct

# Step 4: Run trials
for trial in range(n_trials):
    # Random fixation duration (0.5 to 1.5 seconds)
    fixation_duration = random.uniform(0.5, 1.5)

    # Randomly select the target direction and stimulus type
    target_direction = random.choice(directions)
    stimulus_type = random.choice(['gun', 'cup'])

    # Determine if the cue is valid or invalid
    is_valid_cue = random.random() < cue_validity
    cue_direction = target_direction if is_valid_cue else random.choice([d for d in directions if d != target_direction])

    # Step 1: Show fixation
    fixation.draw()
    win.flip()
    core.wait(fixation_duration)

    # Step 2: Show cue
    cue.text = '⬆' if cue_direction == 'UP' else \
               '⬇' if cue_direction == 'DOWN' else \
               '⬅' if cue_direction == 'LEFT' else \
               '➡'
    cue.draw()
    win.flip()
    core.wait(0.5)  # Display cue for 500 ms

    # Step 3: Show target (stimuli)
    target_image.image = stimuli_images[stimulus_type]
    target_image.pos = positions[target_direction]
    target_image.draw()
    win.flip()

    # Step 4: Wait for response
    timer = core.Clock()
    keys = event.waitKeys(maxWait=1.5, keyList=['up', 'down', 'left', 'right'], timeStamped=timer)

    # Step 5: Log response
    participant_info = f"ParticipantID={info['Participant ID']}, Age={info['Age']}, Gender={info['Gender']}"
    if keys:
        response, rt = keys[0]
        correct_direction = target_direction.lower()
        correct = (response == correct_direction)
        print(f"{participant_info}, Trial {trial+1}: Stimulus={stimulus_type}, Cue={cue_direction}, "
              f"Target={target_direction}, Response={response}, RT={rt:.3f}, Correct={correct}, CueValid={is_valid_cue}")
    else:
        print(f"{participant_info}, Trial {trial+1}: Stimulus={stimulus_type}, Cue={cue_direction}, "
              f"Target={target_direction}, No response, CueValid={is_valid_cue}")


    # Clear the screen briefly between trials
    win.flip()
    core.wait(0.5)

# Step 5: Show experiment completion message
completion_text = visual.TextStim(
    win, 
    text="The experiment is now complete.\n\nThank you for participating!\n\nThe program will close automatically.",
    color="white",
    height=0.1,
    wrapWidth=1.5
)
completion_text.draw()
win.flip()

# Wait for 5 seconds before closing the program
core.wait(5)
# Close the window
win.close()
core.quit()
