#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Ocak 10, 2025, at 12:32
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
expName = 'Yagmur_Final_308'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'Participant': f"{randint(0, 999999):06.0f}",
    'Age': '',
    'Gender': '',
    'Group': 'A',
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
    filename = u'data/%s_%s_%s_%s_%s_%s' % (expInfo['Participant'], expInfo['Age'], expInfo['Gender'], expInfo['Group'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\duruk\\Desktop\\Odev\\PSK_308_FINAL\\Yagmur_Final_308\\Yagmur_Final_308_lastrun.py',
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
            units='deg',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
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
    if deviceManager.getDevice('continue_resp') is None:
        # initialise continue_resp
        continue_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue_resp',
        )
    if deviceManager.getDevice('resp') is None:
        # initialise resp
        resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='resp',
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
    
    # --- Initialize components for Routine "intro_text" ---
    text = visual.TextStim(win=win, name='text',
        text='Welcome\n\nIn this experiment you will be presented with a worded, colored Posner task. The fixation image in the center will change in each block. Keep your eyes fixated to the fixation image.\n\nWords will appear at left or right sides of the screen. Press the arrow keys LEFT or RIGHT to indicate where the word appears.\n\nPress SPACE to continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    continue_resp = keyboard.Keyboard(deviceName='continue_resp')
    
    # --- Initialize components for Routine "ITI" ---
    fixationITI = visual.ImageStim(
        win=win,
        name='fixationITI', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "trial" ---
    fixationITI_continue = visual.ImageStim(
        win=win,
        name='fixationITI_continue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    target = visual.TextStim(win=win, name='target',
        text='',
        font='Arial',
        units='deg', pos=[0,0], draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cue = visual.ShapeStim(
        win=win, name='cue',
        size=(1.5, 1.5), vertices='triangle',
        ori=1.0, pos=(0, -2), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    resp = keyboard.Keyboard(deviceName='resp')
    
    # --- Initialize components for Routine "feedback" ---
    fixationITI_continue_feedback = visual.ImageStim(
        win=win,
        name='fixationITI_continue_feedback', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='',
        font='Arial',
        pos=(0, 2), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "exit" ---
    text_thanks = visual.TextStim(win=win, name='text_thanks',
        text='The experiment has ended. Thank you for your participation.\n\nThis window will close automatically in 5 seconds.',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
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
    
    # --- Prepare to start Routine "intro_text" ---
    # create an object to store info about Routine intro_text
    intro_text = data.Routine(
        name='intro_text',
        components=[text, continue_resp],
    )
    intro_text.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for continue_resp
    continue_resp.keys = []
    continue_resp.rt = []
    _continue_resp_allKeys = []
    # store start times for intro_text
    intro_text.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_text.tStart = globalClock.getTime(format='float')
    intro_text.status = STARTED
    thisExp.addData('intro_text.started', intro_text.tStart)
    intro_text.maxDuration = None
    # keep track of which components have finished
    intro_textComponents = intro_text.components
    for thisComponent in intro_text.components:
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
    
    # --- Run Routine "intro_text" ---
    intro_text.forceEnded = routineForceEnded = not continueRoutine
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
        
        # *continue_resp* updates
        waitOnFlip = False
        
        # if continue_resp is starting this frame...
        if continue_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_resp.frameNStart = frameN  # exact frame index
            continue_resp.tStart = t  # local t and not account for scr refresh
            continue_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_resp.started')
            # update status
            continue_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_resp.status == STARTED and not waitOnFlip:
            theseKeys = continue_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _continue_resp_allKeys.extend(theseKeys)
            if len(_continue_resp_allKeys):
                continue_resp.keys = _continue_resp_allKeys[-1].name  # just the last key pressed
                continue_resp.rt = _continue_resp_allKeys[-1].rt
                continue_resp.duration = _continue_resp_allKeys[-1].duration
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
            intro_text.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_text.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_text" ---
    for thisComponent in intro_text.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_text
    intro_text.tStop = globalClock.getTime(format='float')
    intro_text.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_text.stopped', intro_text.tStop)
    # check responses
    if continue_resp.keys in ['', [], None]:  # No response was made
        continue_resp.keys = None
    thisExp.addData('continue_resp.keys',continue_resp.keys)
    if continue_resp.keys != None:  # we had a response
        thisExp.addData('continue_resp.rt', continue_resp.rt)
        thisExp.addData('continue_resp.duration', continue_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_text" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions("Group" + expInfo["Group"] + ".xlsx"), 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('conditionsPosner.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "ITI" ---
            # create an object to store info about Routine ITI
            ITI = data.Routine(
                name='ITI',
                components=[fixationITI],
            )
            ITI.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            fixationITI.setImage(imagePath)
            # store start times for ITI
            ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ITI.tStart = globalClock.getTime(format='float')
            ITI.status = STARTED
            thisExp.addData('ITI.started', ITI.tStart)
            ITI.maxDuration = None
            # keep track of which components have finished
            ITIComponents = ITI.components
            for thisComponent in ITI.components:
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
            
            # --- Run Routine "ITI" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            ITI.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixationITI* updates
                
                # if fixationITI is starting this frame...
                if fixationITI.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixationITI.frameNStart = frameN  # exact frame index
                    fixationITI.tStart = t  # local t and not account for scr refresh
                    fixationITI.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixationITI, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationITI.started')
                    # update status
                    fixationITI.status = STARTED
                    fixationITI.setAutoDraw(True)
                
                # if fixationITI is active this frame...
                if fixationITI.status == STARTED:
                    # update params
                    pass
                
                # if fixationITI is stopping this frame...
                if fixationITI.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixationITI.tStartRefresh + random()+1-frameTolerance:
                        # keep track of stop time/frame for later
                        fixationITI.tStop = t  # not accounting for scr refresh
                        fixationITI.tStopRefresh = tThisFlipGlobal  # on global time
                        fixationITI.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixationITI.stopped')
                        # update status
                        fixationITI.status = FINISHED
                        fixationITI.setAutoDraw(False)
                
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
                    ITI.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ITI.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ITI" ---
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ITI
            ITI.tStop = globalClock.getTime(format='float')
            ITI.tStopRefresh = tThisFlipGlobal
            thisExp.addData('ITI.stopped', ITI.tStop)
            # the Routine "ITI" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[fixationITI_continue, target, cue, resp],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            fixationITI_continue.setImage(imagePath)
            target.setColor(color, colorSpace='rgb')
            target.setPos((targetX, 0))
            target.setText(word)
            cue.setOri(cueOri)
            # create starting attributes for resp
            resp.keys = []
            resp.rt = []
            _resp_allKeys = []
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
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.2:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixationITI_continue* updates
                
                # if fixationITI_continue is starting this frame...
                if fixationITI_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixationITI_continue.frameNStart = frameN  # exact frame index
                    fixationITI_continue.tStart = t  # local t and not account for scr refresh
                    fixationITI_continue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixationITI_continue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationITI_continue.started')
                    # update status
                    fixationITI_continue.status = STARTED
                    fixationITI_continue.setAutoDraw(True)
                
                # if fixationITI_continue is active this frame...
                if fixationITI_continue.status == STARTED:
                    # update params
                    pass
                
                # if fixationITI_continue is stopping this frame...
                if fixationITI_continue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixationITI_continue.tStartRefresh + 2.2-frameTolerance:
                        # keep track of stop time/frame for later
                        fixationITI_continue.tStop = t  # not accounting for scr refresh
                        fixationITI_continue.tStopRefresh = tThisFlipGlobal  # on global time
                        fixationITI_continue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixationITI_continue.stopped')
                        # update status
                        fixationITI_continue.status = FINISHED
                        fixationITI_continue.setAutoDraw(False)
                
                # *target* updates
                
                # if target is starting this frame...
                if target.status == NOT_STARTED and frameN >= 12:
                    # keep track of start time/frame for later
                    target.frameNStart = frameN  # exact frame index
                    target.tStart = t  # local t and not account for scr refresh
                    target.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target.started')
                    # update status
                    target.status = STARTED
                    target.setAutoDraw(True)
                
                # if target is active this frame...
                if target.status == STARTED:
                    # update params
                    pass
                
                # if target is stopping this frame...
                if target.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        target.tStop = t  # not accounting for scr refresh
                        target.tStopRefresh = tThisFlipGlobal  # on global time
                        target.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target.stopped')
                        # update status
                        target.status = FINISHED
                        target.setAutoDraw(False)
                
                # *cue* updates
                
                # if cue is starting this frame...
                if cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue.frameNStart = frameN  # exact frame index
                    cue.tStart = t  # local t and not account for scr refresh
                    cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue.started')
                    # update status
                    cue.status = STARTED
                    cue.setAutoDraw(True)
                
                # if cue is active this frame...
                if cue.status == STARTED:
                    # update params
                    pass
                
                # if cue is stopping this frame...
                if cue.status == STARTED:
                    if frameN >= 12:
                        # keep track of stop time/frame for later
                        cue.tStop = t  # not accounting for scr refresh
                        cue.tStopRefresh = tThisFlipGlobal  # on global time
                        cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue.stopped')
                        # update status
                        cue.status = FINISHED
                        cue.setAutoDraw(False)
                
                # *resp* updates
                waitOnFlip = False
                
                # if resp is starting this frame...
                if resp.status == NOT_STARTED and frameN >= 12:
                    # keep track of start time/frame for later
                    resp.frameNStart = frameN  # exact frame index
                    resp.tStart = t  # local t and not account for scr refresh
                    resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp.started')
                    # update status
                    resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if resp is stopping this frame...
                if resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > resp.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        resp.tStop = t  # not accounting for scr refresh
                        resp.tStopRefresh = tThisFlipGlobal  # on global time
                        resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'resp.stopped')
                        # update status
                        resp.status = FINISHED
                        resp.status = FINISHED
                if resp.status == STARTED and not waitOnFlip:
                    theseKeys = resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _resp_allKeys.extend(theseKeys)
                    if len(_resp_allKeys):
                        resp.keys = _resp_allKeys[-1].name  # just the last key pressed
                        resp.rt = _resp_allKeys[-1].rt
                        resp.duration = _resp_allKeys[-1].duration
                        # was this correct?
                        if (resp.keys == str(corrAns)) or (resp.keys == corrAns):
                            resp.corr = 1
                        else:
                            resp.corr = 0
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
            if resp.keys in ['', [], None]:  # No response was made
                resp.keys = None
                # was no response the correct answer?!
                if str(corrAns).lower() == 'none':
                   resp.corr = 1;  # correct non-response
                else:
                   resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('resp.keys',resp.keys)
            trials.addData('resp.corr', resp.corr)
            if resp.keys != None:  # we had a response
                trials.addData('resp.rt', resp.rt)
                trials.addData('resp.duration', resp.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.200000)
            
            # --- Prepare to start Routine "feedback" ---
            # create an object to store info about Routine feedback
            feedback = data.Routine(
                name='feedback',
                components=[fixationITI_continue_feedback, text_feedback],
            )
            feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            fixationITI_continue_feedback.setImage(imagePath)
            # Run 'Begin Routine' code from code
            #Check responses
            if resp.corr:
                #Display response time, correct answers out of all trials and correct responses out of last 5 trials
                msgColor = "green"
                msg = "Correct!"
            else:
                msgColor = "red"
                msg = "Incorrect."
            text_feedback.setColor(msgColor, colorSpace='rgb')
            text_feedback.setText(msg)
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
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixationITI_continue_feedback* updates
                
                # if fixationITI_continue_feedback is starting this frame...
                if fixationITI_continue_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixationITI_continue_feedback.frameNStart = frameN  # exact frame index
                    fixationITI_continue_feedback.tStart = t  # local t and not account for scr refresh
                    fixationITI_continue_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixationITI_continue_feedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationITI_continue_feedback.started')
                    # update status
                    fixationITI_continue_feedback.status = STARTED
                    fixationITI_continue_feedback.setAutoDraw(True)
                
                # if fixationITI_continue_feedback is active this frame...
                if fixationITI_continue_feedback.status == STARTED:
                    # update params
                    pass
                
                # if fixationITI_continue_feedback is stopping this frame...
                if fixationITI_continue_feedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixationITI_continue_feedback.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        fixationITI_continue_feedback.tStop = t  # not accounting for scr refresh
                        fixationITI_continue_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                        fixationITI_continue_feedback.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixationITI_continue_feedback.stopped')
                        # update status
                        fixationITI_continue_feedback.status = FINISHED
                        fixationITI_continue_feedback.setAutoDraw(False)
                
                # *text_feedback* updates
                
                # if text_feedback is starting this frame...
                if text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_feedback.frameNStart = frameN  # exact frame index
                    text_feedback.tStart = t  # local t and not account for scr refresh
                    text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_feedback.started')
                    # update status
                    text_feedback.status = STARTED
                    text_feedback.setAutoDraw(True)
                
                # if text_feedback is active this frame...
                if text_feedback.status == STARTED:
                    # update params
                    pass
                
                # if text_feedback is stopping this frame...
                if text_feedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_feedback.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_feedback.tStop = t  # not accounting for scr refresh
                        text_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                        text_feedback.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_feedback.stopped')
                        # update status
                        text_feedback.status = FINISHED
                        text_feedback.setAutoDraw(False)
                
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
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'blocks'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "exit" ---
    # create an object to store info about Routine exit
    exit = data.Routine(
        name='exit',
        components=[text_thanks],
    )
    exit.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for exit
    exit.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    exit.tStart = globalClock.getTime(format='float')
    exit.status = STARTED
    thisExp.addData('exit.started', exit.tStart)
    exit.maxDuration = None
    # keep track of which components have finished
    exitComponents = exit.components
    for thisComponent in exit.components:
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
    
    # --- Run Routine "exit" ---
    exit.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_thanks* updates
        
        # if text_thanks is starting this frame...
        if text_thanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_thanks.frameNStart = frameN  # exact frame index
            text_thanks.tStart = t  # local t and not account for scr refresh
            text_thanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_thanks, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_thanks.started')
            # update status
            text_thanks.status = STARTED
            text_thanks.setAutoDraw(True)
        
        # if text_thanks is active this frame...
        if text_thanks.status == STARTED:
            # update params
            pass
        
        # if text_thanks is stopping this frame...
        if text_thanks.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_thanks.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                text_thanks.tStop = t  # not accounting for scr refresh
                text_thanks.tStopRefresh = tThisFlipGlobal  # on global time
                text_thanks.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_thanks.stopped')
                # update status
                text_thanks.status = FINISHED
                text_thanks.setAutoDraw(False)
        
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
            exit.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in exit.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "exit" ---
    for thisComponent in exit.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for exit
    exit.tStop = globalClock.getTime(format='float')
    exit.tStopRefresh = tThisFlipGlobal
    thisExp.addData('exit.stopped', exit.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if exit.maxDurationReached:
        routineTimer.addTime(-exit.maxDuration)
    elif exit.forceEnded:
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
