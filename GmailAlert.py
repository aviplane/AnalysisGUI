import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apiclient import errors, discovery
import base64


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/gmail.compose',
          ]


class GmailAlert():
    def __init__(self):
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        self.service = build('gmail', 'v1', credentials=creds)

    def send_error_message(self, to, text):
        """
        Does exactly what you would guess

        Params:
            to: str, email address you want to send to
            message_text: str, the message to be sent
        """
        message = self.create_message(sender='sslab.emergency@gmail.com',
                                      to=to,
                                      subject='Cavity Lab Shot Problem',
                                      message_text=text)
        self.send_message("me", message)
        return

    def create_message(self, sender: str, to: str, subject: str, message_text: str):
        """
        Send a message.
        """
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject
        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    def send_message(self, user_id, message):
        try:
            message = (self.service.users().messages().send(
                userId=user_id,
                body=message).execute())
            print(f"Message id: {message['id']}")
            return message
        except errors.HttpError as error:
            print(error)
        return


def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """

    # Call the Gmail API
    gmail_alert = GmailAlert()
    gmail_alert.send_error_message('3016054695@tmomail.net', "what up")


if __name__ == '__main__':
    main()
