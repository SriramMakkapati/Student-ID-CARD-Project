import smtplib
import imghdr
from email.message import EmailMessage

Sender_Email = "studentidcardproject@gmail.com"
Password = "studentidcard"


Reciever_Email = "sriramblaze44@gmail.com"

def sendmail(image):
    try:
        print("Sending Mail...", end = "")
        newMessage = EmailMessage()                         
        newMessage['Subject'] = "Person without ID Card Detected..." 
        newMessage['From'] = Sender_Email                   
        newMessage['To'] = Reciever_Email                   
        newMessage.set_content('Find the attached Image of Voilation and Approve.') 

        with open(image, 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = f.name

        newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Sender_Email, Password)              
            smtp.send_message(newMessage)
        
        print(" Done...")
    except:
        print("Failed")

if __name__ == "__main__":
    sendmail("livemail.png")
        