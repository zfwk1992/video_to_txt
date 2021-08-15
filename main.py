import tempfile
from pytube import YouTube
from pytube import helpers
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import speech_recognition as sr
from os import path

#import ffmpeg
#import textract
import math
import scipy.io.wavfile as wav

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

url='https://www.youtube.com/watch?v=NHopJHSlVo4&ab'
yt=YouTube(url)
title=helpers.safe_filename(yt.title)
print("Downloading ...")
yt = yt.streams.filter(only_audio=True,progressive=False, file_extension='mp4').first()
#yt = yt.streams.filter(only_audio=True,progressive=False, file_extension='mp4').order_by('resolution').desc().first()
yt = yt.download()
print("Converting ...")
print(title + ".mp4")

mp4_version = AudioSegment.from_file( title + '.mp4', 'mp4')
mp4_version.set_channels(1)
mp4_version.export(title+".mp3",format="mp3",parameters=["-ac", "1", "-vol", "150"])

mp3_version = AudioSegment.from_file(title+".mp3","mp3")
channel_count = mp3_version.channels    #Get channels
sample_width = mp3_version.sample_width #Get sample width

mp3_version.set_sample_width(2) 
mp3_version.set_channels(1)
mp3_version.export(title+".wav",format="wav",bitrate="16k")

wav_version = AudioSegment.from_wav(title+".wav")
channel_count = wav_version.channels    #Get channels
sample_width = wav_version.sample_width #Get sample width
duration_in_sec = len(wav_version) / 1000 #Length of audio in sec
sample_rate = wav_version.frame_rate
bit_rate=16

print ("sample_width=", sample_width)
print ("channel_count=", channel_count)
print ("duration_in_sec=", duration_in_sec)
print ("frame_rate=", sample_rate)



AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)),  title+".wav")

f= open(title+'.txt',"w+")

r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file

# recognize speech using Sphinx
try:
    print("Sphinx thinks you said " + r.recognize_sphinx(audio))
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))

# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said: "+ r.recognize_google(audio))
    f.write(r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# recognize speech using Google Cloud Speech
GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""INSERT THE CONTENTS OF THE GOOGLE CLOUD SPEECH JSON CREDENTIALS FILE HERE"""
try:
    print("Google Cloud Speech thinks you said " + r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
except sr.UnknownValueError:
    print("Google Cloud Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Cloud Speech service; {0}".format(e))

# recognize speech using Wit.ai
WIT_AI_KEY = "INSERT WIT.AI API KEY HERE"  # Wit.ai keys are 32-character uppercase alphanumeric strings
try:
    print("Wit.ai thinks you said " + r.recognize_wit(audio, key=WIT_AI_KEY))
except sr.UnknownValueError:
    print("Wit.ai could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Wit.ai service; {0}".format(e))

# recognize speech using Microsoft Azure Speech
AZURE_SPEECH_KEY = "INSERT AZURE SPEECH API KEY HERE"  # Microsoft Speech API keys 32-character lowercase hexadecimal strings
try:
    print("Microsoft Azure Speech thinks you said " + r.recognize_azure(audio, key=AZURE_SPEECH_KEY))
except sr.UnknownValueError:
    print("Microsoft Azure Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Microsoft Azure Speech service; {0}".format(e))

# recognize speech using Microsoft Bing Voice Recognition
BING_KEY = "INSERT BING API KEY HERE"  # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
try:
    print("Microsoft Bing Voice Recognition thinks you said " + r.recognize_bing(audio, key=BING_KEY))
except sr.UnknownValueError:
    print("Microsoft Bing Voice Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))

# recognize speech using Houndify
HOUNDIFY_CLIENT_ID = "INSERT HOUNDIFY CLIENT ID HERE"  # Houndify client IDs are Base64-encoded strings
HOUNDIFY_CLIENT_KEY = "INSERT HOUNDIFY CLIENT KEY HERE"  # Houndify client keys are Base64-encoded strings
try:
    print("Houndify thinks you said " + r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY))
except sr.UnknownValueError:
    print("Houndify could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Houndify service; {0}".format(e))

# recognize speech using IBM Speech to Text
IBM_USERNAME = "INSERT IBM SPEECH TO TEXT USERNAME HERE"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
IBM_PASSWORD = "INSERT IBM SPEECH TO TEXT PASSWORD HERE"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
try:
    print("IBM Speech to Text thinks you said " + r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD))
except sr.UnknownValueError:
    print("IBM Speech to Text could not understand audio")
except sr.RequestError as e:
    print("Could not request results from IBM Speech to Text service; {0}".format(e))