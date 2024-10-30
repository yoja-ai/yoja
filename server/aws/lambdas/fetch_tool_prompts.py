# fetch_tool_prompts.py

from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.oauth2.credentials 
import google.auth.transport.requests
import googleapiclient.discovery
import gspread
from utils import get_user_table_entry, refresh_user_google


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

def construct_full_path(file_id, file_name, service):
    current_id = file_id
    path_segments = [file_name]
    while True:
        file = service.files().get(fileId=current_id, fields="id, name, parents").execute()
        if 'parents' not in file:
            break  # Reached the root
        parent_id = file['parents'][0]
        parent = service.files().get(fileId=parent_id, fields="id, name, parents").execute()
        path_segments.insert(0, parent['name'])
        current_id = parent_id
    return '/' + '/'.join(path_segments)

def _get_sheet_fileid(creds, full_file_name):
    try:
        lst = full_file_name.rfind('/')
        if lst == -1:
            print(f"_get_sheet_fileid: Error. Could not find last /")
            return None
        file_name = full_file_name[lst+1:]
        desired_path = full_file_name[:lst]
        print(f"_get_sheet_fileid: full_file_name={full_file_name}, filename={file_name}, parentdir={desired_path}")
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(
            q=f"name='{file_name}'",
            spaces='drive',
            fields="files(id, name, parents)"
        ).execute()
        items = results.get('files', [])
        for item in items:
            full_path = construct_full_path(item['id'], file_name, service)
            print(f"Checking path: {full_path}")
            if full_path == desired_path + '/' + file_name:
                print(f"Match found: {full_path}. id={item['id']}")
                return item['id']
        print(f"_get_sheet_fileid: Error. full path match not found")
        return None
    except Exception as ex:
        print(f"_get_sheet_fileid: Caught {ex}")
        return None

def fetch_tool_prompts(email, user_input):
    """
    Fetch tool prompts from Google Sheets based on user input in the format: %sheet_name/tab_name.
    Returns a list of dictionaries with 'Tool Name' and 'Description'.
    """
    try:
        item = get_user_table_entry(email)
        creds:google.oauth2.credentials.Credentials = refresh_user_google(item)

        if not user_input.startswith('%'):
            print(f"fetch_tool_prompts: Error. must start with %. Instead it is {user_input}")
            return None
        user_input = user_input[1:]
        if user_input.startswith('[gdrive]'):
            sheet_id = _get_sheet_fileid(creds, user_input[8:])
        elif user_input.startswith('[dropbox]'):
            print(f"fetch_tool_prompts: ERROR!! [dropbox] tool prompts sheet not yet implemented")
            return None
        else:
            print(f"fetch_tool_prompts: sheet name should start with [gdrive] or [dropbox]. Instead it is {user_input}")
            return None
        tab_name = 'Sheet1'
        if not sheet_id:
            print(f"fetch_tool_prompts: Error. could not find sheet id. Not using tool prompts")
            return None

        service = build("sheets", "v4", credentials=creds)
        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=sheet_id, range='A1:Z10')
            .execute()
        )
        values = result.get("values", [])
        if not values:
            print("fetch_tool_prompts: No data found in spreadsheet.")
            return None
        tool_list = []
        for row in values:
            if len(row) >= 2:  # Ensure that both tool name and description are present
                print(f"eeeeeeeeee len(row)={len(row)}")
                tool_name = row[0]
                description = row[1]
                tool = {
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'description': description
                    }
                }
                if len(row) >= 5:
                    tool['function']['parameters'] = {"type": "object", "properties": {}}
                    required = []
                    for ind in range(int((len(row) - 2)/3)):
                        property_name = row[2 + ((ind * 3) + 0)]
                        property_type = row[2 + ((ind * 3) + 1)]
                        property_description = row[2 + ((ind * 3) + 2)]
                        tool['function']['parameters']['properties'][property_name] = {'type': property_type, 'description': property_description}
                        required.append(property_name)
                    tool['function']['parameters']['required'] = required
            tool_list.append(tool)
        if len(tool_list) == 0:
            print(f"fetch_tool_prompts: Error tool_list is 0 len. Using default tool prompts")
            return None
        print(f"Fetched tools from {user_input}: {tool_list}")
        return tool_list
    except HttpError as error:
        print(f"An error occurred while accessing Google Drive API: {error}")
        return None
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet '{tab_name}' not found in sheet '{sheet_name}'.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
