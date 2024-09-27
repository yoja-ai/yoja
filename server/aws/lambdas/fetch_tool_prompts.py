# fetch_tool_prompts.py

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def get_sheet_id_by_name(sheet_name, creds):
    """
    Finds the Google Sheet ID based on the sheet name using Google Drive API.
    """
    try:
        # Build the Drive API client
        service = build('drive', 'v3', credentials=creds)
        
        # Search for the file by name
        results = service.files().list(
            q=f"name='{sheet_name}' and mimeType='application/vnd.google-apps.spreadsheet'",
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        items = results.get('files', [])
        
        if not items:
            print(f"No file found with name: {sheet_name}")
            return None
        else:
            # Return the first file's ID that matches the name
            print(f"Found file: {items[0]['name']} with ID: {items[0]['id']}")
            return items[0]['id']
    
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

def fetch_tool_prompts(user_input):
    """
    Fetch tool prompts from Google Sheets based on user input in the format: %sheet_name/tab_name.
    Returns a list of dictionaries with 'Tool Name' and 'Description'.
    """
    try:
        # Extract the sheet name and tab name from user input: %sheet_name/tab_name
        if not user_input.startswith('%'):
            raise ValueError("User input must start with '%' to indicate a command.")

        sheet_and_tab = user_input[1:].split('/')
        sheet_name = sheet_and_tab[0]  # Get the sheet name
        tab_name = sheet_and_tab[1] if len(sheet_and_tab) > 1 else 'Sheet1'  # Default to 'Sheet1' if not provided

        # Define the scope for Google Sheets and Google Drive
        scope = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]

        # Authenticate using service account credentials
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)

        # Dynamically find the sheet ID based on the sheet name
        sheet_id = get_sheet_id_by_name(sheet_name, creds)
        if not sheet_id:
            raise ValueError(f"Sheet name '{sheet_name}' not found in Google Drive.")

        # Open the Google Sheet by ID and select the worksheet by name
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).worksheet(tab_name)

        # Define the range for tool names and descriptions, you can modify this as needed
        range_name = 'A1:B10'
        data = sheet.get(range_name)  # Get the specified range of cells as a list of lists

        # Parse tools into a list of dictionaries with 'Tool Name' and 'Description'
        tool_list = []
        for row in data:
            if len(row) >= 2:  # Ensure that both tool name and description are present
                tool_name = row[0]
                description = row[1]
                tool_list.append({
                    'Tool Name': tool_name,
                    'Description': description
                })

        print(f"Fetched tools from {sheet_name}/{tab_name}: {tool_list}")
        return tool_list

    except HttpError as error:
        print(f"An error occurred while accessing Google Drive API: {error}")
        return []
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet '{tab_name}' not found in sheet '{sheet_name}'.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
