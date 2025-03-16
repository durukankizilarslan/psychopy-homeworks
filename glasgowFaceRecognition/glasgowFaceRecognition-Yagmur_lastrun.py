#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Aralık 31, 2024, at 14:49
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'glasgowFaceRecognition-Yagmur'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'Participant ID': f"{randint(0, 999999):06.0f}",
    'Age': '',
    'Gender': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s_%s_%s' % (expInfo['Participant ID'], expInfo['Age'], expInfo['Gender'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\duruk\\Desktop\\Odev\\StroopOdev\\glasgowFaceRecognition\\glasgowFaceRecognition-Yagmur\\glasgowFaceRecognition-Yagmur_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='norm',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'norm'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_space') is None:
        # initialise key_space
        key_space = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_space',
        )
    if deviceManager.getDevice('keyResp') is None:
        # initialise keyResp
        keyResp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyResp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    text = visual.TextStim(win=win, name='text',
        text='In this task, you will decide whether two images belong to the same person or different people.\n\n\n\nPress the LEFT arrow key if the faces are the SAME.  \n\nPress the RIGHT arrow key if the faces are DIFFERENT.  \n\n\n\nPress the SPACE bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_space = keyboard.Keyboard(deviceName='key_space')
    
    # --- Initialize components for Routine "trial" ---
    same_text = visual.TextStim(win=win, name='same_text',
        text='SAME',
        font='Arial',
        units='norm', pos=[-0.35, -0.8], draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    different_text = visual.TextStim(win=win, name='different_text',
        text='DIFFERENT',
        font='Arial',
        units='norm', pos=(0.42, -0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    image = visual.ImageStim(
        win=win,
        name='image', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.7, 0.7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    keyResp = keyboard.Keyboard(deviceName='keyResp')
    arrow_left = visual.ShapeStim(
        win=win, name='arrow_left',units='norm', 
        size=(0.1, 0.1), vertices='triangle',
        ori=270.0, pos=(-0.2, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    arrow_right = visual.ShapeStim(
        win=win, name='arrow_right',units='norm', 
        size=(0.1, 0.1), vertices='triangle',
        ori=90.0, pos=(0.2, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback" ---
    textCorr = visual.TextStim(win=win, name='textCorr',
        text='',
        font='Arial',
        units='norm', pos=(0, -0.5), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    same_text_cont = visual.TextStim(win=win, name='same_text_cont',
        text='SAME',
        font='Arial',
        units='norm', pos=[-0.35, -0.8], draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    different_text_cont = visual.TextStim(win=win, name='different_text_cont',
        text='DIFFERENT',
        font='Arial',
        units='norm', pos=(0.42, -0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    image_cont = visual.ImageStim(
        win=win,
        name='image_cont', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.7, 0.7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    arrow_left_cont = visual.ShapeStim(
        win=win, name='arrow_left_cont',units='norm', 
        size=(0.1, 0.1), vertices='triangle',
        ori=270.0, pos=(-0.2, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    arrow_right_cont = visual.ShapeStim(
        win=win, name='arrow_right_cont',units='norm', 
        size=(0.1, 0.1), vertices='triangle',
        ori=90.0, pos=(0.2, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    
    # --- Initialize components for Routine "thanks_message" ---
    thank_you_message = visual.TextStim(win=win, name='thank_you_message',
        text='The experiment has ended. Thank you for your participation!\n\nThis window will close in 5 seconds',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[text, key_space],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_space
    key_space.keys = []
    key_space.rt = []
    _key_space_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_space* updates
        waitOnFlip = False
        
        # if key_space is starting this frame...
        if key_space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_space.frameNStart = frameN  # exact frame index
            key_space.tStart = t  # local t and not account for scr refresh
            key_space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_space, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_space.started')
            # update status
            key_space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_space.status == STARTED and not waitOnFlip:
            theseKeys = key_space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_space_allKeys.extend(theseKeys)
            if len(_key_space_allKeys):
                key_space.keys = _key_space_allKeys[-1].name  # just the last key pressed
                key_space.rt = _key_space_allKeys[-1].rt
                key_space.duration = _key_space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if key_space.keys in ['', [], None]:  # No response was made
        key_space.keys = None
    thisExp.addData('key_space.keys',key_space.keys)
    if key_space.keys != None:  # we had a response
        thisExp.addData('key_space.rt', key_space.rt)
        thisExp.addData('key_space.duration', key_space.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_faceRecognition = data.TrialHandler2(
        name='trials_faceRecognition',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_faceRecognition)  # add the loop to the experiment
    thisTrials_faceRecognition = trials_faceRecognition.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_faceRecognition.rgb)
    if thisTrials_faceRecognition != None:
        for paramName in thisTrials_faceRecognition:
            globals()[paramName] = thisTrials_faceRecognition[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_faceRecognition in trials_faceRecognition:
        currentLoop = trials_faceRecognition
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_faceRecognition.rgb)
        if thisTrials_faceRecognition != None:
            for paramName in thisTrials_faceRecognition:
                globals()[paramName] = thisTrials_faceRecognition[paramName]
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[same_text, different_text, image, keyResp, arrow_left, arrow_right],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image.setImage(imageFile)
        # create starting attributes for keyResp
        keyResp.keys = []
        keyResp.rt = []
        _keyResp_allKeys = []
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        # if trial has changed, end Routine now
        if isinstance(trials_faceRecognition, data.TrialHandler2) and thisTrials_faceRecognition.thisN != trials_faceRecognition.thisTrial.thisN:
            continueRoutine = False
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *same_text* updates
            
            # if same_text is starting this frame...
            if same_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                same_text.frameNStart = frameN  # exact frame index
                same_text.tStart = t  # local t and not account for scr refresh
                same_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(same_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'same_text.started')
                # update status
                same_text.status = STARTED
                same_text.setAutoDraw(True)
            
            # if same_text is active this frame...
            if same_text.status == STARTED:
                # update params
                pass
            
            # *different_text* updates
            
            # if different_text is starting this frame...
            if different_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                different_text.frameNStart = frameN  # exact frame index
                different_text.tStart = t  # local t and not account for scr refresh
                different_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(different_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'different_text.started')
                # update status
                different_text.status = STARTED
                different_text.setAutoDraw(True)
            
            # if different_text is active this frame...
            if different_text.status == STARTED:
                # update params
                pass
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # *keyResp* updates
            waitOnFlip = False
            
            # if keyResp is starting this frame...
            if keyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                keyResp.frameNStart = frameN  # exact frame index
                keyResp.tStart = t  # local t and not account for scr refresh
                keyResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'keyResp.started')
                # update status
                keyResp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keyResp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if keyResp.status == STARTED and not waitOnFlip:
                theseKeys = keyResp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _keyResp_allKeys.extend(theseKeys)
                if len(_keyResp_allKeys):
                    keyResp.keys = _keyResp_allKeys[-1].name  # just the last key pressed
                    keyResp.rt = _keyResp_allKeys[-1].rt
                    keyResp.duration = _keyResp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *arrow_left* updates
            
            # if arrow_left is starting this frame...
            if arrow_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrow_left.frameNStart = frameN  # exact frame index
                arrow_left.tStart = t  # local t and not account for scr refresh
                arrow_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrow_left.started')
                # update status
                arrow_left.status = STARTED
                arrow_left.setAutoDraw(True)
            
            # if arrow_left is active this frame...
            if arrow_left.status == STARTED:
                # update params
                pass
            
            # *arrow_right* updates
            
            # if arrow_right is starting this frame...
            if arrow_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrow_right.frameNStart = frameN  # exact frame index
                arrow_right.tStart = t  # local t and not account for scr refresh
                arrow_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrow_right.started')
                # update status
                arrow_right.status = STARTED
                arrow_right.setAutoDraw(True)
            
            # if arrow_right is active this frame...
            if arrow_right.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # check responses
        if keyResp.keys in ['', [], None]:  # No response was made
            keyResp.keys = None
        trials_faceRecognition.addData('keyResp.keys',keyResp.keys)
        if keyResp.keys != None:  # we had a response
            trials_faceRecognition.addData('keyResp.rt', keyResp.rt)
            trials_faceRecognition.addData('keyResp.duration', keyResp.duration)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[textCorr, same_text_cont, different_text_cont, image_cont, arrow_left_cont, arrow_right_cont],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from setMsg
        # Check for keyboard input
        keys = event.getKeys()
        if 'left' in keys:
            response = 'left'  # User pressed the left arrow key
        elif 'right' in keys:
            response = 'right'  # User pressed the right arrow key
            
        # Compare the user's response with the correct answer from the conditions file
        if response == corrAns:
            msgColor = "green"
            msg = "Correct response!"
            print(msg)  # Message for correct response
            
        else:
            msgColor = "red"
            msg = " Wrong response. Try again!"
            print(msg)  # Message for incorrect response
            
        textCorr.setColor(msgColor, colorSpace='rgb')
        textCorr.setText(msg)
        image_cont.setImage(imageFile)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials_faceRecognition, data.TrialHandler2) and thisTrials_faceRecognition.thisN != trials_faceRecognition.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.2:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textCorr* updates
            
            # if textCorr is starting this frame...
            if textCorr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textCorr.frameNStart = frameN  # exact frame index
                textCorr.tStart = t  # local t and not account for scr refresh
                textCorr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textCorr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textCorr.started')
                # update status
                textCorr.status = STARTED
                textCorr.setAutoDraw(True)
            
            # if textCorr is active this frame...
            if textCorr.status == STARTED:
                # update params
                pass
            
            # if textCorr is stopping this frame...
            if textCorr.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textCorr.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    textCorr.tStop = t  # not accounting for scr refresh
                    textCorr.tStopRefresh = tThisFlipGlobal  # on global time
                    textCorr.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textCorr.stopped')
                    # update status
                    textCorr.status = FINISHED
                    textCorr.setAutoDraw(False)
            
            # *same_text_cont* updates
            
            # if same_text_cont is starting this frame...
            if same_text_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                same_text_cont.frameNStart = frameN  # exact frame index
                same_text_cont.tStart = t  # local t and not account for scr refresh
                same_text_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(same_text_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'same_text_cont.started')
                # update status
                same_text_cont.status = STARTED
                same_text_cont.setAutoDraw(True)
            
            # if same_text_cont is active this frame...
            if same_text_cont.status == STARTED:
                # update params
                pass
            
            # if same_text_cont is stopping this frame...
            if same_text_cont.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > same_text_cont.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    same_text_cont.tStop = t  # not accounting for scr refresh
                    same_text_cont.tStopRefresh = tThisFlipGlobal  # on global time
                    same_text_cont.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'same_text_cont.stopped')
                    # update status
                    same_text_cont.status = FINISHED
                    same_text_cont.setAutoDraw(False)
            
            # *different_text_cont* updates
            
            # if different_text_cont is starting this frame...
            if different_text_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                different_text_cont.frameNStart = frameN  # exact frame index
                different_text_cont.tStart = t  # local t and not account for scr refresh
                different_text_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(different_text_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'different_text_cont.started')
                # update status
                different_text_cont.status = STARTED
                different_text_cont.setAutoDraw(True)
            
            # if different_text_cont is active this frame...
            if different_text_cont.status == STARTED:
                # update params
                pass
            
            # if different_text_cont is stopping this frame...
            if different_text_cont.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > different_text_cont.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    different_text_cont.tStop = t  # not accounting for scr refresh
                    different_text_cont.tStopRefresh = tThisFlipGlobal  # on global time
                    different_text_cont.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'different_text_cont.stopped')
                    # update status
                    different_text_cont.status = FINISHED
                    different_text_cont.setAutoDraw(False)
            
            # *image_cont* updates
            
            # if image_cont is starting this frame...
            if image_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_cont.frameNStart = frameN  # exact frame index
                image_cont.tStart = t  # local t and not account for scr refresh
                image_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_cont.started')
                # update status
                image_cont.status = STARTED
                image_cont.setAutoDraw(True)
            
            # if image_cont is active this frame...
            if image_cont.status == STARTED:
                # update params
                pass
            
            # if image_cont is stopping this frame...
            if image_cont.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_cont.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    image_cont.tStop = t  # not accounting for scr refresh
                    image_cont.tStopRefresh = tThisFlipGlobal  # on global time
                    image_cont.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_cont.stopped')
                    # update status
                    image_cont.status = FINISHED
                    image_cont.setAutoDraw(False)
            
            # *arrow_left_cont* updates
            
            # if arrow_left_cont is starting this frame...
            if arrow_left_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrow_left_cont.frameNStart = frameN  # exact frame index
                arrow_left_cont.tStart = t  # local t and not account for scr refresh
                arrow_left_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow_left_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrow_left_cont.started')
                # update status
                arrow_left_cont.status = STARTED
                arrow_left_cont.setAutoDraw(True)
            
            # if arrow_left_cont is active this frame...
            if arrow_left_cont.status == STARTED:
                # update params
                pass
            
            # if arrow_left_cont is stopping this frame...
            if arrow_left_cont.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrow_left_cont.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    arrow_left_cont.tStop = t  # not accounting for scr refresh
                    arrow_left_cont.tStopRefresh = tThisFlipGlobal  # on global time
                    arrow_left_cont.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrow_left_cont.stopped')
                    # update status
                    arrow_left_cont.status = FINISHED
                    arrow_left_cont.setAutoDraw(False)
            
            # *arrow_right_cont* updates
            
            # if arrow_right_cont is starting this frame...
            if arrow_right_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrow_right_cont.frameNStart = frameN  # exact frame index
                arrow_right_cont.tStart = t  # local t and not account for scr refresh
                arrow_right_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow_right_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrow_right_cont.started')
                # update status
                arrow_right_cont.status = STARTED
                arrow_right_cont.setAutoDraw(True)
            
            # if arrow_right_cont is active this frame...
            if arrow_right_cont.status == STARTED:
                # update params
                pass
            
            # if arrow_right_cont is stopping this frame...
            if arrow_right_cont.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrow_right_cont.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    arrow_right_cont.tStop = t  # not accounting for scr refresh
                    arrow_right_cont.tStopRefresh = tThisFlipGlobal  # on global time
                    arrow_right_cont.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrow_right_cont.stopped')
                    # update status
                    arrow_right_cont.status = FINISHED
                    arrow_right_cont.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.200000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_faceRecognition'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thanks_message" ---
    # create an object to store info about Routine thanks_message
    thanks_message = data.Routine(
        name='thanks_message',
        components=[thank_you_message],
    )
    thanks_message.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thanks_message
    thanks_message.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thanks_message.tStart = globalClock.getTime(format='float')
    thanks_message.status = STARTED
    thisExp.addData('thanks_message.started', thanks_message.tStart)
    thanks_message.maxDuration = None
    # keep track of which components have finished
    thanks_messageComponents = thanks_message.components
    for thisComponent in thanks_message.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks_message" ---
    thanks_message.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thank_you_message* updates
        
        # if thank_you_message is starting this frame...
        if thank_you_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thank_you_message.frameNStart = frameN  # exact frame index
            thank_you_message.tStart = t  # local t and not account for scr refresh
            thank_you_message.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thank_you_message, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thank_you_message.started')
            # update status
            thank_you_message.status = STARTED
            thank_you_message.setAutoDraw(True)
        
        # if thank_you_message is active this frame...
        if thank_you_message.status == STARTED:
            # update params
            pass
        
        # if thank_you_message is stopping this frame...
        if thank_you_message.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thank_you_message.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                thank_you_message.tStop = t  # not accounting for scr refresh
                thank_you_message.tStopRefresh = tThisFlipGlobal  # on global time
                thank_you_message.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thank_you_message.stopped')
                # update status
                thank_you_message.status = FINISHED
                thank_you_message.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thanks_message.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanks_message.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks_message" ---
    for thisComponent in thanks_message.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thanks_message
    thanks_message.tStop = globalClock.getTime(format='float')
    thanks_message.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thanks_message.stopped', thanks_message.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thanks_message.maxDurationReached:
        routineTimer.addTime(-thanks_message.maxDuration)
    elif thanks_message.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
