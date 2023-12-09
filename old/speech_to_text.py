import speech_recognition as sr
import pyttsx3 
 
# Initialize the recognizer 
     
     
# Loop infinitely for user to
# speak
def speech_to_text():
    r = sr.Recognizer() 
    while(1):   
     
        print("Speak now")
        
        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input 
                audio2 = r.listen(source2)
                
                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
    
                return MyText
                
        except sr.RequestError as e:
            return ("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            return ("unknown error occured")
        